from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tc_mo_me.model import UsageRecord

GENOME_SCHEMA_VERSION = "0.1"
MAX_WORKFLOW_NODES = 64
CONTENT_COMPONENTS = (
    "system_prompt",
    "planning_policy",
    "code_generation_policy",
    "workflow_graph",
    "verification_policy",
)
CONTENT_FIELDS = ("schema_version", *CONTENT_COMPONENTS)

Metadata: TypeAlias = dict[str, JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class WorkflowNodeType(StrEnum):
    PLANNER = "planner"
    GENERATOR = "generator"
    EXECUTOR = "executor"
    VERIFIER = "verifier"


class WorkflowNode(StrictModel):
    id: str = Field(min_length=1)
    type: WorkflowNodeType
    config: Metadata = Field(default_factory=dict)


class WorkflowEdge(StrictModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class WorkflowGraph(StrictModel):
    entry_id: str = Field(min_length=1)
    exit_ids: list[str] = Field(min_length=1)
    max_nodes: int = Field(default=MAX_WORKFLOW_NODES, ge=1, le=MAX_WORKFLOW_NODES)
    nodes: list[WorkflowNode] = Field(min_length=1)
    edges: list[WorkflowEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_structure(self) -> WorkflowGraph:
        node_ids = [node.id for node in self.nodes]
        node_id_set = set(node_ids)
        if len(node_id_set) != len(node_ids):
            raise ValueError("Node IDs must be unique")
        if len(self.nodes) > self.max_nodes:
            raise ValueError(f"Node count exceeds max_nodes={self.max_nodes}")
        if len(set(self.exit_ids)) != len(self.exit_ids):
            raise ValueError("Exit IDs must be unique")
        if self.entry_id not in node_id_set:
            raise ValueError(f"Unknown entry_id: {self.entry_id}")
        missing_exits = set(self.exit_ids) - node_id_set
        if missing_exits:
            raise ValueError(f"Unknown exit_ids: {sorted(missing_exits)}")

        adjacency = {node_id: [] for node_id in node_ids}
        reverse_adjacency = {node_id: [] for node_id in node_ids}
        edge_pairs = []
        for edge in self.edges:
            if edge.source not in node_id_set:
                raise ValueError(f"Unknown edge source: {edge.source}")
            if edge.target not in node_id_set:
                raise ValueError(f"Unknown edge target: {edge.target}")
            pair = (edge.source, edge.target)
            edge_pairs.append(pair)
            adjacency[edge.source].append(edge.target)
            reverse_adjacency[edge.target].append(edge.source)
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("Edges must be unique")

        reachable_from_entry = _reachable([self.entry_id], adjacency)
        unreachable = node_id_set - reachable_from_entry
        if unreachable:
            raise ValueError(
                f"Nodes must be reachable from entry_id; unreachable: {sorted(unreachable)}"
            )

        reachable_to_exit = _reachable(self.exit_ids, reverse_adjacency)
        without_exit = node_id_set - reachable_to_exit
        if without_exit:
            raise ValueError(
                f"Every node must reach an exit; nodes without a path: {sorted(without_exit)}"
            )
        return self


def _reachable(start_ids: list[str], adjacency: dict[str, list[str]]) -> set[str]:
    visited = set(start_ids)
    pending = list(start_ids)
    while pending:
        current = pending.pop()
        for target in adjacency[current]:
            if target not in visited:
                visited.add(target)
                pending.append(target)
    return visited


class Genome(StrictModel):
    schema_version: Literal[GENOME_SCHEMA_VERSION] = GENOME_SCHEMA_VERSION
    system_prompt: str = Field(min_length=1)
    planning_policy: Metadata
    code_generation_policy: Metadata
    workflow_graph: WorkflowGraph
    verification_policy: Metadata
    identity: Metadata = Field(default_factory=dict)
    lineage: Metadata = Field(default_factory=dict)

    def canonical_genome_bytes(self) -> bytes:
        return canonical_genome_bytes(self)

    def genome_hash(self) -> str:
        return genome_hash(self)


def validate_genome(genome: Genome | Mapping[str, Any] | str | bytes) -> Genome:
    if isinstance(genome, Genome):
        return Genome.model_validate_json(genome.model_dump_json())
    if isinstance(genome, Mapping):
        return Genome.model_validate_json(
            json.dumps(genome, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
    if isinstance(genome, str | bytes):
        return Genome.model_validate_json(genome)
    raise TypeError("genome must be a Genome, mapping, JSON string, or UTF-8 JSON bytes")


def canonical_genome_bytes(genome: Genome | Mapping[str, Any] | str | bytes) -> bytes:
    validated = validate_genome(genome)
    content = validated.model_dump(mode="json", include=set(CONTENT_FIELDS))
    return json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def genome_hash(genome: Genome | Mapping[str, Any] | str | bytes) -> str:
    return hashlib.sha256(canonical_genome_bytes(genome)).hexdigest()


class LineageRecord(StrictModel):
    operator_id: str = Field(min_length=1)
    parent_ids: list[str] = Field(min_length=1)
    primary_parent_id: str = Field(min_length=1)
    child_id: str = Field(min_length=1)
    modified_paths: list[str] = Field(min_length=1)
    component_sources: dict[str, str]
    parent_hashes: dict[str, str]
    child_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=1)
    usage: UsageRecord
    created_at_utc: datetime

    @model_validator(mode="after")
    def validate_lineage(self) -> LineageRecord:
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("parent_ids must be unique")
        if self.primary_parent_id not in self.parent_ids:
            raise ValueError("primary_parent_id must be included in parent_ids")
        if self.child_id in self.parent_ids:
            raise ValueError("child_id must differ from all parent_ids")
        if len(set(self.modified_paths)) != len(self.modified_paths):
            raise ValueError("modified_paths must be unique")
        for path in self.modified_paths:
            if path.split(".", 1)[0] not in CONTENT_COMPONENTS:
                raise ValueError("modified_paths must reference Genome content components")

        if set(self.component_sources) != set(CONTENT_COMPONENTS):
            raise ValueError("component_sources must cover every Genome content component")
        unknown_sources = set(self.component_sources.values()) - set(self.parent_ids)
        if unknown_sources:
            raise ValueError(f"component_sources reference unknown parents: {sorted(unknown_sources)}")

        if set(self.parent_hashes) != set(self.parent_ids):
            raise ValueError("parent_hashes must contain exactly the parent_ids")
        for parent_id, parent_hash in self.parent_hashes.items():
            if not _is_sha256(parent_hash):
                raise ValueError(f"parent_hashes[{parent_id!r}] must be a SHA256 hex digest")
        if self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError("created_at_utc must include the UTC timezone")
        return self


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
