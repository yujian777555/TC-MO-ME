from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from tc_mo_me.genome import CONTENT_COMPONENTS, Genome, validate_genome
from tc_mo_me.operators.base import (
    LOCAL_COMPONENTS,
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
    "Return only one strict JSON object with exactly the keys 'component' and 'value'. "
    "The component must equal the requested local component. Never return or modify "
    "workflow_graph, identity, lineage, schema_version, or any other Genome component."
)


def _prompt(parent: Genome, component: str) -> str:
    payload = {
        "component": component,
        "current_value": content_payload(parent)[component],
        "parent_hash": parent.genome_hash(),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_response(content: str, requested_component: str) -> Any:
    payload = parse_strict_json(content)
    if not isinstance(payload, dict):
        raise _OperatorFailure(
            "invalid_response", "O1 output must be a JSON object with component and value"
        )
    if set(payload) != {"component", "value"}:
        if set(payload) & (set(CONTENT_COMPONENTS) | {"schema_version", "identity", "lineage"}):
            raise _OperatorFailure(
                "unauthorized_modification",
                "O1 output cannot contain a whole Genome or additional components",
            )
        raise _OperatorFailure(
            "invalid_response", "O1 output must contain exactly component and value"
        )
    component = payload["component"]
    if not isinstance(component, str):
        raise _OperatorFailure("invalid_response", "O1 component must be a string")
    if component not in LOCAL_COMPONENTS:
        raise _OperatorFailure(
            "unauthorized_modification",
            f"O1 cannot modify component {component!r}",
            details={"requested_component": requested_component},
        )
    if component != requested_component:
        raise _OperatorFailure(
            "component_mismatch",
            "O1 response component must match the MutationSpec component",
            details={"requested": requested_component, "received": component},
        )
    return payload["value"]


class O1LocalOperator(EvolutionOperator):
    operator_id = "O1"

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
                spec.component not in LOCAL_COMPONENTS
                or spec.operation is not None
                or spec.component_sources
            ):
                raise _OperatorFailure(
                    "invalid_spec",
                    "O1 requires exactly one allowed local component and no other operation",
                    details={"allowed_components": list(LOCAL_COMPONENTS)},
                )
            normalized = normalize_parents(parents, minimum=1, maximum=1)
            parent = normalized[0]
            component = spec.component
            if component is None:
                raise AssertionError("component was checked by spec validation")
            response = call_model(context, _prompt(parent, component), _SYSTEM_PROMPT)
            usage = response.usage
            value = _parse_response(response.content, component)
            candidate_payload = content_payload(parent)
            candidate_payload[component] = deepcopy(value)
            try:
                candidate = validate_genome(candidate_payload)
            except (TypeError, ValueError) as error:
                raise _OperatorFailure(
                    "invalid_component_value",
                    f"O1 value is invalid for {component}",
                    details={"validation_error": str(error)},
                ) from error
            for other in CONTENT_COMPONENTS:
                if other != component and getattr(candidate, other) != getattr(parent, other):
                    raise _OperatorFailure(
                        "unauthorized_modification",
                        "O1 changed a component outside the MutationSpec",
                        details={"component": other},
                    )
            return build_success(
                operator_id=self.operator_id,
                parents=normalized,
                primary_parent=parent,
                candidate=candidate,
                component_sources={name: parent_id(parent) for name in CONTENT_COMPONENTS},
                modified_paths=[component],
                spec=spec,
                context=context,
                usage=usage,
            )
        except _OperatorFailure as error:
            return failure_result(error, usage)


O1LocalMutationOperator = O1LocalOperator

__all__ = ["O1LocalMutationOperator", "O1LocalOperator"]
