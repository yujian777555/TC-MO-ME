from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

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


class UsageRecord(StrictModel):
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    price_id: str | None = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    cost: Decimal | None = None
    cost_basis: str = Field(default="unknown", min_length=1)

    @model_validator(mode="after")
    def validate_usage(self) -> UsageRecord:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        if self.cost is None and self.cost_basis != "unknown":
            raise ValueError("Unknown cost requires cost_basis='unknown'")
        if self.cost is not None and self.cost_basis == "unknown":
            raise ValueError("Unknown cost must use cost=None, not a numeric value")
        return self


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
    cost: Decimal | None
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
        if self.cost != self.usage.cost:
            raise ValueError("Lineage cost must match usage.cost")
        if self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError("created_at_utc must include the UTC timezone")
        return self


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class ModelRequest(StrictModel):
    model_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    metadata: Metadata = Field(default_factory=dict)


class ModelResponse(StrictModel):
    text: str
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    usage: UsageRecord
    provider_response_id: str | None = None
    finish_reason: str | None = None


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


class MockModelAdapter(ModelAdapter):
    def __init__(
        self,
        response_text: str = "",
        *,
        responses: Mapping[str, str] | None = None,
        model_version: str | None = "mock-v1",
        price_id: str | None = None,
    ) -> None:
        self._response_text = response_text
        self._responses = dict(responses or {})
        self._model_version = model_version
        self._price_id = price_id

    def generate(self, request: ModelRequest) -> ModelResponse:
        text = self._responses.get(request.prompt, self._response_text)
        input_tokens = len(request.prompt.split())
        if request.system_prompt is not None:
            input_tokens += len(request.system_prompt.split())
        output_tokens = len(text.split())
        usage = UsageRecord(
            model_id=request.model_id,
            model_version=self._model_version,
            price_id=self._price_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            call_count=1,
            retry_count=0,
            elapsed_seconds=0.0,
            cost=None,
            cost_basis="unknown",
        )
        return ModelResponse(
            text=text,
            model_id=request.model_id,
            model_version=self._model_version,
            usage=usage,
            finish_reason="stop",
        )


class OpenAICompatibleModelAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_version: str | None = None,
        price_id: str | None = None,
        timeout_seconds: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if api_key is None:
                raise ValueError("api_key is required when no client is supplied")
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError("Install the 'api' extra to use OpenAI-compatible models") from error
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=0,
            )
        self._client = client
        self._model_version = model_version
        self._price_id = price_id

    def generate(self, request: ModelRequest) -> ModelResponse:
        messages = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        kwargs = {
            "model": request.model_id,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens

        started = time.perf_counter()
        raw_response = self._client.chat.completions.create(**kwargs)
        elapsed_seconds = time.perf_counter() - started
        if raw_response.usage is None:
            raise ValueError("OpenAI-compatible response must include token usage")

        raw_usage = raw_response.usage
        provider_cost = getattr(raw_usage, "cost", None)
        cost = None if provider_cost is None else Decimal(str(provider_cost))
        cost_basis = "unknown" if provider_cost is None else "provider_reported"
        usage = UsageRecord(
            model_id=raw_response.model,
            model_version=raw_response.system_fingerprint or self._model_version,
            price_id=self._price_id,
            input_tokens=raw_usage.prompt_tokens,
            output_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
            call_count=1,
            retry_count=0,
            elapsed_seconds=elapsed_seconds,
            cost=cost,
            cost_basis=cost_basis,
        )
        choice = raw_response.choices[0]
        return ModelResponse(
            text=choice.message.content or "",
            model_id=raw_response.model,
            model_version=raw_response.system_fingerprint or self._model_version,
            usage=usage,
            provider_response_id=raw_response.id,
            finish_reason=choice.finish_reason,
        )
