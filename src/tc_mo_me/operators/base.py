from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tc_mo_me.genome import (
    CONTENT_FIELDS,
    Genome,
    LineageRecord,
    genome_hash,
    validate_genome,
)
from tc_mo_me.model import (
    CostBasis,
    ModelAdapter,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    UsageRecord,
)

LOCAL_COMPONENTS = (
    "system_prompt",
    "planning_policy",
    "code_generation_policy",
    "verification_policy",
)
STRUCTURAL_OPERATIONS = ("insert", "delete", "replace", "reorder")
REPRODUCIBLE_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

ParentInput: TypeAlias = Genome | Sequence[Genome]


@dataclass(frozen=True)
class OperatorContext:
    model: ModelAdapter | None = None
    model_id: str = "none"
    generation: int = 1
    child_id: str | None = None
    created_at_utc: datetime = REPRODUCIBLE_EPOCH
    seed: int | None = None
    temperature: float | None = 0.0
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    price_id: str | None = None

    @property
    def model_adapter(self) -> ModelAdapter | None:
        return self.model


class MutationSpec(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    component: str | None = None
    operation: str | None = None
    component_sources: dict[str, str] = Field(default_factory=dict)


class OperatorError(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class OperatorResult(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    success: bool
    child: Genome | None = None
    lineage: LineageRecord | None = None
    usage: UsageRecord
    modified_paths: list[str] = Field(default_factory=list)
    error: OperatorError | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.success:
            if self.child is None or self.lineage is None or self.error is not None:
                raise ValueError("successful results require child and lineage without an error")
            if not self.modified_paths:
                raise ValueError("successful results require modified_paths")
            if self.lineage.child_hash != genome_hash(self.child):
                raise ValueError("lineage child_hash must match the child content")
            if self.lineage.usage != self.usage:
                raise ValueError("lineage usage must match the result usage")
        else:
            if self.error is None:
                raise ValueError("failed results require a structured error")
            if self.child is not None or self.lineage is not None:
                raise ValueError("failed results cannot contain a child or lineage")
            if self.modified_paths:
                raise ValueError("failed results cannot report applied modifications")
        return self

    @property
    def ok(self) -> bool:
        return self.success

    @property
    def status(self) -> str:
        return "succeeded" if self.success else "generation_failed"

    @property
    def modifications(self) -> list[str]:
        return self.modified_paths

    @property
    def usage_records(self) -> list[UsageRecord]:
        return [self.usage]


class EvolutionOperator(ABC):
    operator_id: str

    @abstractmethod
    def generate(
        self,
        parents: ParentInput,
        spec: MutationSpec,
        context: OperatorContext,
    ) -> OperatorResult:
        raise NotImplementedError


class _OperatorFailure(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, JsonValue] | None = None,
        usage: UsageRecord | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = {} if details is None else details
        self.usage = usage


def zero_usage(model_id: str = "deterministic") -> UsageRecord:
    return UsageRecord(
        model_id=model_id,
        phase="generation",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        calls=0,
        retries=0,
        elapsed_seconds=0.0,
        api_cost=Decimal(0),
        cost_basis=CostBasis.ESTIMATED,
    )


def attempted_usage(model_id: str) -> UsageRecord:
    return UsageRecord(
        model_id=model_id,
        phase="generation",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        calls=1,
        retries=0,
        elapsed_seconds=0.0,
        api_cost=None,
        cost_basis=CostBasis.UNKNOWN,
    )


def failure_result(
    error: _OperatorFailure,
    usage: UsageRecord,
) -> OperatorResult:
    return OperatorResult(
        success=False,
        usage=error.usage or usage,
        error=OperatorError(
            code=error.code,
            message=error.message,
            details=error.details,
        ),
    )


def validate_context(context: OperatorContext) -> None:
    if not isinstance(context.model_id, str) or not context.model_id:
        raise _OperatorFailure("invalid_context", "model_id must be a non-empty string")
    if (
        isinstance(context.generation, bool)
        or not isinstance(context.generation, int)
        or context.generation < 1
    ):
        raise _OperatorFailure("invalid_context", "generation must be an integer of at least 1")
    if context.child_id is not None and (
        not isinstance(context.child_id, str) or not context.child_id
    ):
        raise _OperatorFailure(
            "invalid_context", "child_id must be a non-empty string when provided"
        )
    if not isinstance(context.created_at_utc, datetime) or (
        context.created_at_utc.utcoffset() != timedelta(0)
    ):
        raise _OperatorFailure("invalid_context", "created_at_utc must include the UTC timezone")


def normalize_parents(parents: ParentInput, *, minimum: int, maximum: int | None) -> list[Genome]:
    if isinstance(parents, Genome):
        raw_parents = [parents]
    else:
        try:
            raw_parents = list(parents)
        except TypeError as error:
            raise _OperatorFailure(
                "invalid_parents", "parents must be a Genome or a sequence of Genomes"
            ) from error
    if len(raw_parents) < minimum or (maximum is not None and len(raw_parents) > maximum):
        expected = str(minimum) if maximum == minimum else f"at least {minimum}"
        raise _OperatorFailure(
            "invalid_parents",
            f"operator requires {expected} parents; received {len(raw_parents)}",
        )

    normalized = []
    parent_ids = []
    for parent in raw_parents:
        try:
            cloned = validate_genome(parent)
        except (TypeError, ValueError) as error:
            raise _OperatorFailure(
                "invalid_parents",
                "every parent must be a valid Genome",
                details={"validation_error": str(error)},
            ) from error
        parent_id = cloned.identity.get("genome_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise _OperatorFailure(
                "invalid_parents", "every parent requires a non-empty identity.genome_id"
            )
        normalized.append(cloned)
        parent_ids.append(parent_id)
    if len(set(parent_ids)) != len(parent_ids):
        raise _OperatorFailure("invalid_parents", "parent genome_id values must be unique")
    return normalized


def parent_id(parent: Genome) -> str:
    value = parent.identity["genome_id"]
    if not isinstance(value, str):
        raise TypeError("validated parent genome_id must be a string")
    return value


def content_payload(genome: Genome) -> dict[str, Any]:
    return genome.model_dump(mode="json", include=set(CONTENT_FIELDS))


def call_model(context: OperatorContext, prompt: str, system_prompt: str) -> ModelResponse:
    """Call a generation-phase model without discarding final failure usage."""
    if context.model is None:
        raise _OperatorFailure(
            "model_unavailable",
            "this operator requires a model adapter",
            usage=zero_usage(context.model_id),
        )
    try:
        request = ModelRequest(
            model_id=context.model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            seed=context.seed,
            temperature=context.temperature,
            max_tokens=context.max_tokens,
            timeout_seconds=context.timeout_seconds,
            max_retries=context.max_retries,
            price_id=context.price_id,
        )
    except ValueError as error:
        raise _OperatorFailure(
            "invalid_context",
            "model request configuration is invalid",
            details={"validation_error": str(error)},
            usage=zero_usage(context.model_id),
        ) from error
    try:
        response = context.model.generate(request)
        usage = response.usage.model_copy(update={"phase": "generation"})
        return response.model_copy(update={"usage": usage})
    except ModelCallError as error:
        raise _OperatorFailure(
            "model_error",
            f"model generation failed: {error}",
            details={"exception_type": type(error).__name__},
            usage=error.usage.model_copy(update={"phase": "generation"}),
        ) from error
    except Exception as error:
        raise _OperatorFailure(
            "model_error",
            f"model generation failed: {error}",
            details={"exception_type": type(error).__name__},
            usage=attempted_usage(context.model_id),
        ) from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_strict_json(content: str) -> Any:
    try:
        return json.loads(
            content,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise _OperatorFailure(
            "invalid_json",
            "model output must be one strict JSON value without Markdown fences",
            details={"parse_error": str(error)},
        ) from error


def deterministic_child_id(
    operator_id: str,
    parents: list[Genome],
    spec: MutationSpec,
    generation: int,
    child_hash: str,
) -> str:
    payload = {
        "operator_id": operator_id,
        "generation": generation,
        "parent_ids": [parent_id(parent) for parent in parents],
        "parent_hashes": {parent_id(parent): genome_hash(parent) for parent in parents},
        "spec": spec.model_dump(mode="json"),
        "child_hash": child_hash,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{operator_id.lower()}-{hashlib.sha256(serialized).hexdigest()[:24]}"


def build_success(
    *,
    operator_id: str,
    parents: list[Genome],
    primary_parent: Genome,
    candidate: Genome | dict[str, Any],
    component_sources: dict[str, str],
    modified_paths: list[str],
    spec: MutationSpec,
    context: OperatorContext,
    usage: UsageRecord,
) -> OperatorResult:
    candidate_payload = content_payload(candidate) if isinstance(candidate, Genome) else candidate
    primary_hash = genome_hash(primary_parent)
    child_hash = genome_hash(candidate_payload)
    if child_hash == primary_hash:
        raise _OperatorFailure(
            "no_change", "operator output must change the primary parent's content hash"
        )

    child_id = context.child_id or deterministic_child_id(
        operator_id, parents, spec, context.generation, child_hash
    )
    ids = [parent_id(parent) for parent in parents]
    if child_id in ids:
        raise _OperatorFailure(
            "invalid_child_identity", "child_id must differ from every parent genome_id"
        )
    identity = deepcopy(primary_parent.identity)
    identity.update(
        {
            "genome_id": child_id,
            "origin": "evolved",
            "generation": context.generation,
            "operator_id": operator_id,
        }
    )
    lineage = LineageRecord(
        operator_id=operator_id,
        parent_ids=ids,
        primary_parent_id=parent_id(primary_parent),
        child_id=child_id,
        modified_paths=modified_paths,
        component_sources=component_sources,
        parent_hashes={parent_id(parent): genome_hash(parent) for parent in parents},
        child_hash=child_hash,
        generation=context.generation,
        usage=usage,
        created_at_utc=context.created_at_utc,
    )
    child_data = deepcopy(candidate_payload)
    child_data["identity"] = identity
    child_data["lineage"] = lineage.model_dump(mode="json")
    child = validate_genome(child_data)
    return OperatorResult(
        success=True,
        child=child,
        lineage=lineage,
        usage=usage,
        modified_paths=modified_paths,
    )


__all__ = [
    "EvolutionOperator",
    "MutationSpec",
    "OperatorContext",
    "OperatorError",
    "OperatorResult",
]
