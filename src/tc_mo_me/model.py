from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CostBasis",
    "DeterministicMockAdapter",
    "MockModelAdapter",
    "ModelAdapter",
    "ModelCallError",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleAdapter",
    "OpenAIModelAdapter",
    "UnsupportedSeedError",
    "UsageGroupKey",
    "UsageRecord",
    "aggregate_usage",
    "canonical_prompt_bytes",
    "group_usage",
    "prompt_hash",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class CostBasis(StrEnum):
    UNKNOWN = "unknown"
    ESTIMATED = "estimated"
    ACTUAL = "actual"
    TOKEN_ESTIMATE = "estimated"
    API_RETURNED = "actual"


class ModelRequest(_StrictModel):
    model_id: str = Field(min_length=1)
    prompt: str
    system_prompt: str | None = None
    seed: int | None = None
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0)
    price_id: str | None = Field(default=None, min_length=1)

    @property
    def prompt_hash(self) -> str:
        return prompt_hash(self)


class UsageRecord(_StrictModel):
    """Logical-call usage, with retries as extra attempts and an optional phase."""

    model_id: str = Field(min_length=1)
    model_version: str | None = None
    price_id: str | None = None
    phase: Literal["generation", "evaluation"] | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    calls: int = Field(default=1, ge=0)
    retries: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    api_cost: Decimal | None = Field(default=None, ge=0)
    cost_basis: CostBasis = CostBasis.UNKNOWN

    @model_validator(mode="after")
    def validate_cost_basis(self) -> Self:
        if self.cost_basis is CostBasis.UNKNOWN and self.api_cost is not None:
            raise ValueError("api_cost must be null when cost_basis is unknown")
        if self.cost_basis is not CostBasis.UNKNOWN and self.api_cost is None:
            raise ValueError("api_cost is required for actual or estimated cost_basis")
        return self

    @property
    def input_tokens(self) -> int | None:
        return self.prompt_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.completion_tokens


class UsageGroupKey(NamedTuple):
    model_id: str
    model_version: str | None
    price_id: str | None
    cost_basis: CostBasis
    phase: Literal["generation", "evaluation"] | None = None


class ModelResponse(_StrictModel):
    content: str
    model_id: str = Field(min_length=1)
    model_version: str | None = None
    prompt_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    model_seed_supported: bool
    usage: UsageRecord
    finish_reason: str | None = None

    @property
    def text(self) -> str:
        return self.content

    @property
    def output_text(self) -> str:
        return self.content

    @property
    def price_id(self) -> str | None:
        return self.usage.price_id


class UnsupportedSeedError(ValueError):
    pass


class ModelCallError(Exception):
    """A final provider failure with its attempts, timing, and known usage."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        retries: int,
        elapsed_seconds: float,
        usage: UsageRecord,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retries = retries
        self.elapsed_seconds = elapsed_seconds
        self.usage = usage

    @property
    def elapsed(self) -> float:
        return self.elapsed_seconds


def canonical_prompt_bytes(request: ModelRequest) -> bytes:
    payload = {"prompt": request.prompt, "system_prompt": request.system_prompt}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def prompt_hash(request: ModelRequest) -> str:
    return hashlib.sha256(canonical_prompt_bytes(request)).hexdigest()


def _sum_optional(values: Iterable[int | None]) -> int | None:
    items = list(values)
    if any(value is None for value in items):
        return None
    return sum(value for value in items if value is not None)


def group_usage(records: Iterable[UsageRecord]) -> dict[UsageGroupKey, UsageRecord]:
    """Aggregate records without merging model, price, cost basis, or phase."""
    grouped: dict[UsageGroupKey, list[UsageRecord]] = {}
    for record in records:
        key = UsageGroupKey(
            record.model_id,
            record.model_version,
            record.price_id,
            record.cost_basis,
            record.phase,
        )
        grouped.setdefault(key, []).append(record)

    result: dict[UsageGroupKey, UsageRecord] = {}
    for key, items in grouped.items():
        api_cost = None
        if key.cost_basis is not CostBasis.UNKNOWN:
            api_cost = sum(
                (item.api_cost for item in items if item.api_cost is not None),
                Decimal(0),
            )
        result[key] = UsageRecord(
            model_id=key.model_id,
            model_version=key.model_version,
            price_id=key.price_id,
            phase=key.phase,
            prompt_tokens=_sum_optional(item.prompt_tokens for item in items),
            completion_tokens=_sum_optional(item.completion_tokens for item in items),
            total_tokens=_sum_optional(item.total_tokens for item in items),
            calls=sum(item.calls for item in items),
            retries=sum(item.retries for item in items),
            elapsed_seconds=sum(item.elapsed_seconds for item in items),
            api_cost=api_cost,
            cost_basis=key.cost_basis,
        )
    return result


def aggregate_usage(records: Iterable[UsageRecord]) -> dict[UsageGroupKey, UsageRecord]:
    return group_usage(records)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        try:
            status = int(status_code)
        except (TypeError, ValueError):
            pass
        else:
            return status in {408, 409, 429} or status >= 500
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "InternalServerError",
        "PoolTimeout",
        "RateLimitError",
        "ReadTimeout",
        "TimeoutException",
        "WriteTimeout",
    }


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


class ModelAdapter(ABC):
    def __init__(
        self,
        *,
        timeout_seconds: float | None = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        model_seed_supported: bool = False,
        price_id: str | None = None,
        input_price_per_million: Decimal | None = None,
        output_price_per_million: Decimal | None = None,
        deterministic_timing: bool = False,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if input_price_per_million is not None and input_price_per_million < 0:
            raise ValueError("input_price_per_million must be non-negative")
        if output_price_per_million is not None and output_price_per_million < 0:
            raise ValueError("output_price_per_million must be non-negative")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.model_seed_supported = model_seed_supported
        self.price_id = price_id
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self._deterministic_timing = deterministic_timing

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate once logically, retrying with additional provider attempts."""
        if request.seed is not None and not self.model_seed_supported:
            raise UnsupportedSeedError(
                f"model {request.model_id!r} does not declare support for seed"
            )
        if (
            self.price_id is not None
            and request.price_id is not None
            and request.price_id != self.price_id
        ):
            raise ValueError("request price_id must match the adapter's configured price_id")

        max_retries = self.max_retries if request.max_retries is None else request.max_retries
        timeout = self.timeout_seconds if request.timeout_seconds is None else request.timeout_seconds
        started = time.monotonic()
        retries = 0
        while True:
            try:
                response = self._generate_once(request, timeout)
            except Exception as error:
                if retries >= max_retries or not _is_retryable(error):
                    elapsed = 0.0 if self._deterministic_timing else time.monotonic() - started
                    attempts = retries + 1
                    usage = UsageRecord(
                        model_id=request.model_id,
                        price_id=self.price_id
                        if self.price_id is not None
                        else request.price_id,
                        calls=1,
                        retries=retries,
                        elapsed_seconds=elapsed,
                    )
                    raise ModelCallError(
                        f"model generation failed after {attempts} provider attempts: {error}",
                        attempts=attempts,
                        retries=retries,
                        elapsed_seconds=elapsed,
                        usage=usage,
                    ) from error
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**retries))
                retries += 1
                continue

            elapsed = 0.0 if self._deterministic_timing else time.monotonic() - started
            usage = response.usage.model_copy(
                update={"calls": 1, "retries": retries, "elapsed_seconds": elapsed}
            )
            return response.model_copy(update={"usage": usage})

    @abstractmethod
    def _generate_once(
        self,
        request: ModelRequest,
        timeout_seconds: float | None,
    ) -> ModelResponse:
        raise NotImplementedError

    def _usage(
        self,
        request: ModelRequest,
        *,
        model_version: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        actual_cost: Any = None,
    ) -> UsageRecord:
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        api_cost = _optional_decimal(actual_cost)
        cost_basis = CostBasis.ACTUAL if api_cost is not None else CostBasis.UNKNOWN
        if (
            api_cost is None
            and prompt_tokens is not None
            and completion_tokens is not None
            and self.input_price_per_million is not None
            and self.output_price_per_million is not None
        ):
            api_cost = (
                Decimal(prompt_tokens) * self.input_price_per_million
                + Decimal(completion_tokens) * self.output_price_per_million
            ) / Decimal(1_000_000)
            cost_basis = CostBasis.ESTIMATED

        return UsageRecord(
            model_id=request.model_id,
            model_version=model_version,
            price_id=self.price_id if self.price_id is not None else request.price_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            api_cost=api_cost,
            cost_basis=cost_basis,
        )


class MockModelAdapter(ModelAdapter):
    def __init__(
        self,
        script: Mapping[str, str] | None = None,
        *,
        model_version: str | None = "mock-v1",
        timeout_seconds: float | None = 30.0,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        model_seed_supported: bool = True,
        price_id: str | None = None,
        input_price_per_million: Decimal | None = None,
        output_price_per_million: Decimal | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            model_seed_supported=model_seed_supported,
            price_id=price_id,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            deterministic_timing=True,
        )
        self._script = dict(script or {})
        self._model_version = model_version

    def _generate_once(
        self,
        request: ModelRequest,
        timeout_seconds: float | None,
    ) -> ModelResponse:
        content = self._script.get(request.prompt, f"mock:{request.prompt}")
        usage = self._usage(
            request,
            model_version=self._model_version,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=None,
        )
        return ModelResponse(
            content=content,
            model_id=request.model_id,
            model_version=self._model_version,
            prompt_hash=request.prompt_hash,
            model_seed_supported=self.model_seed_supported,
            usage=usage,
            finish_reason="stop",
        )


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.0,
        model_seed_supported: bool = False,
        price_id: str | None = None,
        input_price_per_million: Decimal | None = None,
        output_price_per_million: Decimal | None = None,
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            model_seed_supported=model_seed_supported,
            price_id=price_id,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
        if client is not None and api_key is not None:
            raise ValueError("pass api_key only when constructing the SDK client")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "OpenAICompatibleAdapter requires the 'api' extra or an injected client"
                ) from error

            client_options: dict[str, Any] = {
                "timeout": timeout_seconds,
                "max_retries": 0,
            }
            resolved_api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
            if resolved_api_key is not None:
                client_options["api_key"] = resolved_api_key
            if base_url is not None:
                client_options["base_url"] = base_url
            client = OpenAI(**client_options)
        self._client = client

    def _generate_once(
        self,
        request: ModelRequest,
        timeout_seconds: float | None,
    ) -> ModelResponse:
        messages = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {"model": request.model_id, "messages": messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed

        options: dict[str, Any] = {"max_retries": 0}
        if timeout_seconds is not None:
            options["timeout"] = timeout_seconds
        raw_response = self._client.with_options(**options).chat.completions.create(**payload)

        choices = _field(raw_response, "choices")
        if not choices:
            raise ValueError("provider response did not contain a choice")
        choice = choices[0]
        content = _field(_field(choice, "message"), "content")
        if not isinstance(content, str):
            raise ValueError("provider response did not contain text content")

        raw_usage = _field(raw_response, "usage")
        prompt_tokens = _field(raw_usage, "prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = _field(raw_usage, "input_tokens")
        completion_tokens = _field(raw_usage, "completion_tokens")
        if completion_tokens is None:
            completion_tokens = _field(raw_usage, "output_tokens")
        total_tokens = _field(raw_usage, "total_tokens")
        actual_cost = _field(raw_usage, "cost")
        if actual_cost is None:
            actual_cost = _field(raw_usage, "total_cost")
        if actual_cost is None:
            actual_cost = _field(raw_response, "cost")
        if actual_cost is None:
            actual_cost = _field(raw_response, "total_cost")

        model_version = _field(raw_response, "model")
        usage = self._usage(
            request,
            model_version=model_version,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            actual_cost=actual_cost,
        )
        return ModelResponse(
            content=content,
            model_id=request.model_id,
            model_version=model_version,
            prompt_hash=request.prompt_hash,
            model_seed_supported=self.model_seed_supported,
            usage=usage,
            finish_reason=_field(choice, "finish_reason"),
        )


DeterministicMockAdapter = MockModelAdapter
OpenAIModelAdapter = OpenAICompatibleAdapter
