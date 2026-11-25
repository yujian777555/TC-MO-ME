from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from tc_mo_me.genome import CONTENT_COMPONENTS, Genome, validate_genome
from tc_mo_me.operators.base import (
    STRUCTURAL_OPERATIONS,
    EvolutionOperator,
    MutationSpec,
    OperatorContext,
    OperatorResult,
    ParentInput,
    _OperatorFailure,
    build_success,
    call_model,
    content_payload,
    failure_result,
    normalize_parents,
    parent_id,
    parse_strict_json,
    validate_context,
    zero_usage,
)

_SYSTEM_PROMPT = (
    "Return exactly one strict JSON graph patch, never a list of multiple operations. "
    "Use one of these exact shapes: "
    "{operation: insert, node, source, target}; "
    "{operation: delete, node_id}; "
    "{operation: replace, node_id, node}; "
    "{operation: reorder, node_ids}. "
    "The operation must match the request. Do not modify any non-workflow Genome field."
)

_PATCH_KEYS = {
    "insert": {"operation", "node", "source", "target"},
    "delete": {"operation", "node_id"},
    "replace": {"operation", "node_id", "node"},
    "reorder": {"operation", "node_ids"},
}
_PROTECTED_FIELDS = set(CONTENT_COMPONENTS) | {"schema_version", "identity", "lineage"}


def _prompt(parent: Genome, operation: str) -> str:
    payload = {
        "operation": operation,
        "parent_hash": parent.genome_hash(),
        "workflow_graph": parent.workflow_graph.model_dump(mode="json"),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_patch(content: str, requested_operation: str) -> dict[str, Any]:
    payload = parse_strict_json(content)
    if isinstance(payload, list):
        if len(payload) != 1:
            code = "multiple_operations" if payload else "invalid_patch"
            raise _OperatorFailure(code, "O2 accepts exactly one graph operation")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise _OperatorFailure("invalid_patch", "O2 patch must be one JSON object")
    if "operations" in payload:
        operations = payload["operations"]
        code = (
            "multiple_operations"
            if isinstance(operations, list) and len(operations) > 1
            else "invalid_patch"
        )
        raise _OperatorFailure(code, "O2 accepts exactly one graph operation")
    if "path" in payload:
        path = payload["path"]
        if not isinstance(path, str) or not path.startswith("/workflow_graph"):
            raise _OperatorFailure(
                "unauthorized_modification", "O2 cannot patch outside workflow_graph"
            )
        raise _OperatorFailure(
            "invalid_patch", "generic JSON Patch paths are not supported; use a graph patch"
        )
    if set(payload) & (_PROTECTED_FIELDS - {"workflow_graph"}):
        raise _OperatorFailure(
            "unauthorized_modification", "O2 cannot include non-workflow Genome fields"
        )
    operation = payload.get("operation")
    if operation not in STRUCTURAL_OPERATIONS:
        raise _OperatorFailure(
            "invalid_patch",
            "O2 operation must be insert, delete, replace, or reorder",
        )
    if operation != requested_operation:
        raise _OperatorFailure(
            "operation_mismatch",
            "O2 patch operation must match the MutationSpec operation",
            details={"requested": requested_operation, "received": operation},
        )
    if set(payload) != _PATCH_KEYS[operation]:
        raise _OperatorFailure(
            "invalid_patch",
            f"O2 {operation} patch has an invalid set of fields",
            details={"required_fields": sorted(_PATCH_KEYS[operation])},
        )
    return payload


def _node_index(nodes: list[dict[str, Any]], node_id: str) -> int:
    matches = [index for index, node in enumerate(nodes) if node.get("id") == node_id]
    if len(matches) != 1:
        raise _OperatorFailure(
            "invalid_patch", f"patch must identify exactly one existing node: {node_id!r}"
        )
    return matches[0]


def _insert(graph: dict[str, Any], patch: dict[str, Any]) -> None:
    node = patch["node"]
    source = patch["source"]
    target = patch["target"]
    if not isinstance(node, dict) or not isinstance(source, str) or not isinstance(target, str):
        raise _OperatorFailure("invalid_patch", "insert requires a node and string endpoints")
    node_id = node.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise _OperatorFailure("invalid_patch", "inserted node requires a non-empty id")
    nodes = graph["nodes"]
    edges = graph["edges"]
    matching_edges = [
        index
        for index, edge in enumerate(edges)
        if edge["source"] == source and edge["target"] == target
    ]
    if len(matching_edges) != 1:
        raise _OperatorFailure(
            "invalid_patch", "insert source and target must identify one existing edge"
        )
    edge_index = matching_edges[0]
    target_index = _node_index(nodes, target)
    nodes.insert(target_index, deepcopy(node))
    edges[edge_index : edge_index + 1] = [
        {"source": source, "target": node_id},
        {"source": node_id, "target": target},
    ]


def _delete(graph: dict[str, Any], patch: dict[str, Any]) -> None:
    node_id = patch["node_id"]
    if not isinstance(node_id, str):
        raise _OperatorFailure("invalid_patch", "delete node_id must be a string")
    nodes = graph["nodes"]
    edges = graph["edges"]
    _node_index(nodes, node_id)
    incoming = [(index, edge) for index, edge in enumerate(edges) if edge["target"] == node_id]
    outgoing = [(index, edge) for index, edge in enumerate(edges) if edge["source"] == node_id]
    if len(incoming) != 1 or len(outgoing) != 1:
        raise _OperatorFailure(
            "invalid_patch",
            "delete requires one incoming and one outgoing edge for safe reconnection",
        )
    source = incoming[0][1]["source"]
    target = outgoing[0][1]["target"]
    incident_indexes = {incoming[0][0], outgoing[0][0]}
    insertion_index = min(incident_indexes)
    remaining_edges = [edge for index, edge in enumerate(edges) if index not in incident_indexes]
    reconnection = {"source": source, "target": target}
    if reconnection not in remaining_edges:
        position = sum(1 for index in range(insertion_index) if index not in incident_indexes)
        remaining_edges.insert(position, reconnection)
    graph["nodes"] = [node for node in nodes if node.get("id") != node_id]
    graph["edges"] = remaining_edges


def _replace(graph: dict[str, Any], patch: dict[str, Any]) -> None:
    node_id = patch["node_id"]
    replacement = patch["node"]
    if not isinstance(node_id, str) or not isinstance(replacement, dict):
        raise _OperatorFailure("invalid_patch", "replace requires node_id and a node object")
    if replacement.get("id") != node_id:
        raise _OperatorFailure(
            "unauthorized_modification", "replace cannot change a node id or rewire the graph"
        )
    index = _node_index(graph["nodes"], node_id)
    graph["nodes"][index] = deepcopy(replacement)


def _reorder(graph: dict[str, Any], patch: dict[str, Any]) -> None:
    node_ids = patch["node_ids"]
    if not isinstance(node_ids, list) or not all(isinstance(item, str) for item in node_ids):
        raise _OperatorFailure("invalid_patch", "reorder node_ids must be a list of strings")
    nodes = graph["nodes"]
    existing_ids = [node["id"] for node in nodes]
    if (
        len(node_ids) != len(existing_ids)
        or len(set(node_ids)) != len(node_ids)
        or set(node_ids) != set(existing_ids)
    ):
        raise _OperatorFailure(
            "invalid_patch", "reorder must contain every existing node id exactly once"
        )
    by_id = {node["id"]: node for node in nodes}
    graph["nodes"] = [by_id[node_id] for node_id in node_ids]


def _apply_patch(parent: Genome, patch: dict[str, Any]) -> Genome:
    graph = parent.workflow_graph.model_dump(mode="json")
    operation = patch["operation"]
    if operation == "insert":
        _insert(graph, patch)
    elif operation == "delete":
        _delete(graph, patch)
    elif operation == "replace":
        _replace(graph, patch)
    else:
        _reorder(graph, patch)
    candidate_payload = content_payload(parent)
    candidate_payload["workflow_graph"] = graph
    try:
        return validate_genome(candidate_payload)
    except (TypeError, ValueError) as error:
        raise _OperatorFailure(
            "invalid_genome",
            "O2 patch produced an invalid Genome workflow graph",
            details={"validation_error": str(error)},
        ) from error


class O2StructuralOperator(EvolutionOperator):
    operator_id = "O2"

    def generate(
        self,
        parents: ParentInput,
        spec: MutationSpec,
        context: OperatorContext,
    ) -> OperatorResult:
        usage = zero_usage(context.model_id)
        try:
            validate_context(context)
            if (
                spec.operation not in STRUCTURAL_OPERATIONS
                or spec.component not in (None, "workflow_graph")
                or spec.component_sources
            ):
                raise _OperatorFailure(
                    "invalid_spec",
                    "O2 requires one structural operation and no component sources",
                    details={"allowed_operations": list(STRUCTURAL_OPERATIONS)},
                )
            normalized = normalize_parents(parents, minimum=1, maximum=1)
            parent = normalized[0]
            operation = spec.operation
            if operation is None:
                raise AssertionError("operation was checked by spec validation")
            response = call_model(context, _prompt(parent, operation), _SYSTEM_PROMPT)
            usage = response.usage
            patch = _parse_patch(response.content, operation)
            candidate = _apply_patch(parent, patch)
            return build_success(
                operator_id=self.operator_id,
                parents=normalized,
                primary_parent=parent,
                candidate=candidate,
                component_sources={name: parent_id(parent) for name in CONTENT_COMPONENTS},
                modified_paths=["workflow_graph"],
                spec=spec,
                context=context,
                usage=usage,
            )
        except _OperatorFailure as error:
            return failure_result(error, usage)


O2StructuralMutationOperator = O2StructuralOperator

__all__ = ["O2StructuralMutationOperator", "O2StructuralOperator"]
