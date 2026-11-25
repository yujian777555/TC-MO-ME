import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import tc_mo_me.model as model_module
from tc_mo_me.model import (
    CostBasis,
    MockModelAdapter,
    ModelRequest,
    OpenAICompatibleAdapter,
    UnsupportedSeedError,
    UsageGroupKey,
    UsageRecord,
    aggregate_usage,
    canonical_prompt_bytes,
    group_usage,
    prompt_hash,
)


class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider status {status_code}")
        self.status_code = status_code


class FakeClient:
    def __init__(self, items: list[object]) -> None:
        self.items = list(items)
        self.options: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **options: object) -> "FakeClient":
        self.options.append(options)
        return self

    def create(self, **payload: object) -> object:
        self.calls.append(payload)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def provider_response(
    *,
    content: str = "completed",
    model: str | None = "vendor-model-v2",
    prompt_tokens: int | None = 3,
    completion_tokens: int | None = 2,
    total_tokens: int | None = 5,
    cost: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        ),
    )


def test_prompt_hash_uses_canonical_content_and_excludes_configuration() -> None:
    request = ModelRequest(
        model_id="model-a",
        system_prompt="system",
        prompt="你好",
        seed=1,
        temperature=0.0,
    )
    same_prompt = ModelRequest(
        model_id="model-b",
        system_prompt="system",
        prompt="你好",
        seed=99,
        temperature=1.0,
        max_tokens=10,
        timeout_seconds=2.0,
    )
    expected_bytes = json.dumps(
        {"prompt": "你好", "system_prompt": "system"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert canonical_prompt_bytes(request) == expected_bytes
    assert prompt_hash(request) == hashlib.sha256(expected_bytes).hexdigest()
    assert request.prompt_hash == same_prompt.prompt_hash
    assert prompt_hash(request) != prompt_hash(same_prompt.model_copy(update={"prompt": "other"}))


def test_mock_is_deterministic_and_returns_model_metadata() -> None:
    adapter = MockModelAdapter(script={"hello": "scripted"}, price_id="mock-free")
    request = ModelRequest(model_id="mock", prompt="hello", seed=7)

    first = adapter.generate(request)
    second = adapter.generate(request)

    assert first == second
    assert first.content == "scripted"
    assert first.text == "scripted"
    assert first.model_version == "mock-v1"
    assert first.model_seed_supported is True
    assert first.prompt_hash == request.prompt_hash
    assert first.usage.model_id == "mock"
    assert first.usage.model_version == "mock-v1"
    assert first.usage.price_id == "mock-free"
    assert first.usage.calls == 1
    assert first.usage.retries == 0
    assert first.usage.elapsed_seconds == 0.0
    assert first.usage.api_cost is None
    assert first.usage.cost_basis is CostBasis.UNKNOWN
    assert first.usage.phase is None


def test_mock_estimates_cost_only_when_tokens_and_prices_are_known() -> None:
    priced = MockModelAdapter(
        script={"one two": "three four five"},
        price_id="mock-priced-v1",
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("2"),
    )
    unpriced = MockModelAdapter(script={"one two": "three four five"})
    request = ModelRequest(model_id="mock", prompt="one two")

    priced_usage = priced.generate(request).usage
    unpriced_usage = unpriced.generate(request).usage

    assert priced_usage.prompt_tokens == 2
    assert priced_usage.completion_tokens == 3
    assert priced_usage.total_tokens == 5
    assert priced_usage.api_cost == Decimal("0.000008")
    assert priced_usage.cost_basis is CostBasis.ESTIMATED
    assert unpriced_usage.api_cost is None
    assert unpriced_usage.cost_basis is CostBasis.UNKNOWN
    assert unpriced_usage.api_cost != 0


def usage_record(
    *,
    version: str | None = "v1",
    price_id: str | None = "price-a",
    cost_basis: CostBasis = CostBasis.ACTUAL,
    cost: Decimal | None = Decimal("1"),
    prompt_tokens: int | None = 1,
    phase: str | None = None,
) -> UsageRecord:
    total_tokens = None if prompt_tokens is None else prompt_tokens + 1
    return UsageRecord(
        model_id="model",
        model_version=version,
        price_id=price_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=None if prompt_tokens is None else 1,
        total_tokens=total_tokens,
        calls=1,
        retries=0,
        elapsed_seconds=0.25,
        api_cost=cost,
        cost_basis=cost_basis,
        phase=phase,
    )


def test_usage_grouping_does_not_merge_versions_prices_or_cost_basis() -> None:
    records = [
        usage_record(),
        usage_record(cost=Decimal("2")),
        usage_record(version="v2", cost=Decimal("4")),
        usage_record(price_id="price-b", cost=Decimal("8")),
        usage_record(cost_basis=CostBasis.ESTIMATED, cost=Decimal("0.5")),
        usage_record(cost_basis=CostBasis.UNKNOWN, cost=None, prompt_tokens=None),
        usage_record(cost_basis=CostBasis.UNKNOWN, cost=None),
    ]

    grouped = group_usage(records)
    actual_key = UsageGroupKey("model", "v1", "price-a", CostBasis.ACTUAL)
    unknown_key = UsageGroupKey("model", "v1", "price-a", CostBasis.UNKNOWN)

    assert aggregate_usage(records) == grouped
    assert len(grouped) == 5
    assert grouped[actual_key].calls == 2
    assert grouped[actual_key].prompt_tokens == 2
    assert grouped[actual_key].total_tokens == 4
    assert grouped[actual_key].elapsed_seconds == 0.5
    assert grouped[actual_key].api_cost == Decimal("3")
    assert grouped[unknown_key].prompt_tokens is None
    assert grouped[unknown_key].completion_tokens is None
    assert grouped[unknown_key].total_tokens is None
    assert grouped[unknown_key].api_cost is None
    assert grouped[unknown_key].api_cost != 0
    assert UsageGroupKey("model", "v2", "price-a", CostBasis.ACTUAL) in grouped
    assert UsageGroupKey("model", "v1", "price-b", CostBasis.ACTUAL) in grouped
    assert UsageGroupKey("model", "v1", "price-a", CostBasis.ESTIMATED) in grouped


def test_usage_grouping_separates_generation_evaluation_and_unscoped_phases() -> None:
    records = [
        usage_record(phase="generation"),
        usage_record(phase="generation", cost=Decimal("2")),
        usage_record(phase="evaluation"),
        usage_record(),
    ]

    grouped = group_usage(records)
    generation_key = UsageGroupKey("model", "v1", "price-a", CostBasis.ACTUAL, "generation")
    evaluation_key = UsageGroupKey("model", "v1", "price-a", CostBasis.ACTUAL, "evaluation")
    unscoped_key = UsageGroupKey("model", "v1", "price-a", CostBasis.ACTUAL)

    assert len(grouped) == 3
    assert grouped[generation_key].calls == 2
    assert grouped[generation_key].api_cost == Decimal("3")
    assert grouped[generation_key].phase == "generation"
    assert grouped[evaluation_key].calls == 1
    assert grouped[evaluation_key].phase == "evaluation"
    assert grouped[unscoped_key].calls == 1
    assert grouped[unscoped_key].phase is None


def test_openai_adapter_maps_response_actual_cost_and_timeout() -> None:
    client = FakeClient([provider_response(cost="0.125")])
    adapter = OpenAICompatibleAdapter(
        client=client,
        price_id="provider-price-v1",
        input_price_per_million=Decimal("100"),
        output_price_per_million=Decimal("100"),
        timeout_seconds=30.0,
        model_seed_supported=True,
    )
    request = ModelRequest(
        model_id="vendor/model",
        system_prompt="system",
        prompt="hello",
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=2.5,
    )

    response = adapter.generate(request)

    assert response.content == "completed"
    assert response.model_id == "vendor/model"
    assert response.model_version == "vendor-model-v2"
    assert response.prompt_hash == request.prompt_hash
    assert response.model_seed_supported is True
    assert response.usage.price_id == "provider-price-v1"
    assert response.usage.api_cost == Decimal("0.125")
    assert response.usage.cost_basis is CostBasis.ACTUAL
    assert response.usage.retries == 0
    assert response.usage.calls == 1
    assert response.usage.elapsed_seconds >= 0.0
    assert client.options == [{"timeout": 2.5, "max_retries": 0}]
    assert client.calls == [
        {
            "model": "vendor/model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.0,
            "max_tokens": 32,
        }
    ]


def test_openai_adapter_uses_token_estimate_when_provider_cost_is_absent() -> None:
    client = FakeClient([provider_response(prompt_tokens=2, completion_tokens=3, total_tokens=5)])
    adapter = OpenAICompatibleAdapter(
        client=client,
        price_id="provider-price-v1",
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("2"),
    )

    usage = adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello")).usage

    assert usage.api_cost == Decimal("0.000008")
    assert usage.cost_basis is CostBasis.ESTIMATED


def test_request_price_id_mismatch_is_rejected_before_provider_call() -> None:
    client = FakeClient([provider_response()])
    adapter = OpenAICompatibleAdapter(
        client=client,
        price_id="provider-price-v1",
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("2"),
    )
    request = ModelRequest(
        model_id="vendor/model",
        prompt="hello",
        price_id="unrelated-request-price",
    )

    with pytest.raises(ValueError, match="price_id"):
        adapter.generate(request)

    assert client.calls == []
    assert client.options == []


def test_openai_adapter_preserves_unknown_version_tokens_and_cost() -> None:
    client = FakeClient(
        [
            provider_response(
                model=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(client=client)

    response = adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello"))

    assert response.model_version is None
    assert response.usage.model_version is None
    assert response.usage.price_id is None
    assert response.usage.prompt_tokens is None
    assert response.usage.completion_tokens is None
    assert response.usage.total_tokens is None
    assert response.usage.api_cost is None
    assert response.usage.cost_basis is CostBasis.UNKNOWN


def test_retryable_errors_are_retried_and_recorded() -> None:
    client = FakeClient(
        [ProviderError(500), ProviderError(429), provider_response()]
    )
    adapter = OpenAICompatibleAdapter(client=client, max_retries=2)

    response = adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello"))

    assert response.content == "completed"
    assert response.usage.calls == 1
    assert response.usage.retries == 2
    assert len(client.calls) == 3


def test_timeout_errors_are_retryable() -> None:
    client = FakeClient([TimeoutError("slow"), provider_response()])
    adapter = OpenAICompatibleAdapter(client=client, max_retries=1)

    response = adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello"))

    assert response.usage.retries == 1
    assert len(client.calls) == 2


def test_retry_exhaustion_raises_structured_error_with_complete_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([ProviderError(500), ProviderError(500), ProviderError(500)])
    adapter = OpenAICompatibleAdapter(
        client=client,
        max_retries=2,
        price_id="provider-price-v1",
    )
    monkeypatch.setattr(model_module.time, "monotonic", iter([10.0, 10.375]).__next__)

    with pytest.raises(Exception) as exc_info:
        adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello"))

    error = exc_info.value
    assert type(error).__name__ == "ModelCallError"
    assert isinstance(error, model_module.ModelCallError)
    assert isinstance(error.__cause__, ProviderError)
    assert "provider status 500" in str(error)
    assert error.attempts == 3
    assert error.retries == 2
    assert error.elapsed_seconds == 0.375
    assert error.usage.calls == 1
    assert error.usage.retries == 2
    assert error.usage.elapsed_seconds == 0.375
    assert error.usage.price_id == "provider-price-v1"
    assert error.usage.prompt_tokens is None
    assert error.usage.completion_tokens is None
    assert error.usage.total_tokens is None
    assert error.usage.api_cost is None
    assert error.usage.cost_basis is CostBasis.UNKNOWN
    assert error.usage.phase is None
    assert len(client.calls) == 3


def test_non_retryable_error_is_wrapped_without_additional_attempts() -> None:
    client = FakeClient([ProviderError(400)])
    adapter = OpenAICompatibleAdapter(client=client, max_retries=5)

    with pytest.raises(Exception) as exc_info:
        adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello"))

    error = exc_info.value
    assert type(error).__name__ == "ModelCallError"
    assert isinstance(error, model_module.ModelCallError)
    assert isinstance(error.__cause__, ProviderError)
    assert "provider status 400" in str(error)
    assert error.attempts == 1
    assert error.retries == 0
    assert error.usage.calls == 1
    assert error.usage.retries == 0
    assert len(client.calls) == 1


def test_unsupported_seed_fails_before_provider_call() -> None:
    client = FakeClient([provider_response()])
    adapter = OpenAICompatibleAdapter(client=client, model_seed_supported=False)

    with pytest.raises(UnsupportedSeedError):
        adapter.generate(ModelRequest(model_id="vendor/model", prompt="hello", seed=1))

    assert client.calls == []


def test_retry_and_timeout_values_are_validated() -> None:
    with pytest.raises(ValidationError):
        ModelRequest(model_id="model", prompt="hello", max_retries=-1)
    with pytest.raises(ValidationError):
        ModelRequest(model_id="model", prompt="hello", timeout_seconds=0)
    with pytest.raises(ValueError, match="max_retries"):
        OpenAICompatibleAdapter(client=FakeClient([]), max_retries=-1)
    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAICompatibleAdapter(client=FakeClient([]), timeout_seconds=0)


def test_attempt_progress_callback_reports_attempt_start_and_preserves_final_usage() -> None:
    client = FakeClient([ProviderError(500), provider_response()])
    adapter = OpenAICompatibleAdapter(client=client, max_retries=1)
    events = []

    def record_progress(event) -> None:
        events.append((event.attempt, event.retries, event.elapsed_seconds, len(client.calls)))

    response = adapter.generate(
        ModelRequest(model_id="vendor/model", prompt="hello"),
        progress_callback=record_progress,
    )

    assert [(attempt, retries, calls) for attempt, retries, _elapsed, calls in events] == [
        (1, 0, 0),
        (2, 1, 1),
    ]
    assert all(elapsed >= 0 for _attempt, _retries, elapsed, _calls in events)
    assert response.usage.calls == 1
    assert response.usage.retries == 1
    assert len(client.calls) == 2
