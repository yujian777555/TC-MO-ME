from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from tc_mo_me.agent import AgentRunner, GenerationAttempt, WorkflowMockAdapter
from tc_mo_me.e1_protocol import (
    GENOME_IDS,
    E1Plan,
    EpisodePlan,
    OfficialTestResult,
    ParentEpisodeStateInput,
    ParentStateEvent,
    PreChildState,
    ProtocolEvent,
    build_block_events,
    build_plan_from_file,
    build_pre_child_state,
    validate_event_order,
    validate_pre_child_state,
)
from tc_mo_me.genome import Genome, genome_hash, validate_genome
from tc_mo_me.ledger import EpisodeRecord, RunLedger
from tc_mo_me.model import MockModelAdapter, ModelAdapter, ModelRequest, ModelResponse
from tc_mo_me.operators import (
    MutationSpec,
    O1LocalOperator,
    O2StructuralOperator,
    O3CrossoverOperator,
    OperatorContext,
)
from tc_mo_me.provenance import capture_provenance

Sha256 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", validate_default=True)


class SourceProtocolConfig(_StrictModel):
    config_path: Literal["configs/e1/pilot.yaml"]
    config_sha256: Sha256
    plan_sha256: Sha256


class MockModelConfig(_StrictModel):
    adapter: Literal["mock"]
    model_id: NonEmptyString
    model_version: NonEmptyString
    price_id: NonEmptyString
    input_price_per_million: Decimal = Field(ge=0)
    output_price_per_million: Decimal = Field(ge=0)
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_retries: int = Field(ge=0)

    @field_validator("input_price_per_million", "output_price_per_million", mode="before")
    @classmethod
    def parse_price(cls, value: object) -> object:
        return Decimal(str(value)) if isinstance(value, str | int | float) else value

    @model_validator(mode="after")
    def validate_zero_cost(self) -> Self:
        if self.input_price_per_million != 0 or self.output_price_per_million != 0:
            raise ValueError("E1 mock model prices must both be explicitly zero")
        return self


class MockEnvironmentConfig(_StrictModel):
    kind: Literal["synthetic_mock"]
    instruction: NonEmptyString


class MockRunnerConfig(_StrictModel):
    max_steps: int = Field(gt=0)
    process_isolation: bool
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_timeout(self) -> Self:
        if not self.process_isolation and self.timeout_seconds is not None:
            raise ValueError("non-isolated mock execution cannot enforce timeout_seconds")
        return self


class MockLimits(_StrictModel):
    max_blocks: int = Field(ge=1, le=16)
    max_episodes: int = Field(ge=7, le=112)

    @model_validator(mode="after")
    def validate_episode_limit(self) -> Self:
        if self.max_episodes != self.max_blocks * 7:
            raise ValueError("max_episodes must equal seven times max_blocks")
        return self


class MockProvenanceConfig(_StrictModel):
    require_clean: Literal[True]


class E1MockConfig(_StrictModel):
    schema_version: Literal[1]
    enabled: Literal[True]
    run_id: Literal["e1-mock-dry-run-v2"]
    experiment_id: Literal["e1-mock-dry-run"]
    source_protocol: SourceProtocolConfig
    model: MockModelConfig
    environment: MockEnvironmentConfig
    runner: MockRunnerConfig
    limits: MockLimits
    provenance: MockProvenanceConfig


def load_mock_config(path: str | Path) -> E1MockConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E1 mock config must be a YAML mapping")
    return E1MockConfig.model_validate(payload)


def _mock_model(model_config: MockModelConfig) -> E1OperatorMockAdapter:
    return E1OperatorMockAdapter(
        model_version=model_config.model_version,
        timeout_seconds=model_config.timeout_seconds,
        max_retries=model_config.max_retries,
        price_id=model_config.price_id,
        input_price_per_million=model_config.input_price_per_million,
        output_price_per_million=model_config.output_price_per_million,
    )


def _o1_mock_value(component: str, current: Any, parent_hash: str) -> Any:
    if component == "system_prompt":
        return f"{current} Synthetic O1 mock revision {parent_hash[:12]}."
    value = deepcopy(current)
    if not isinstance(value, dict):
        raise ValueError(f"unsupported O1 mock component: {component}")
    value["synthetic_o1_mock"] = {"parent_sha256": parent_hash, "revision": 1}
    return value


def _o2_mock_patch(payload: Mapping[str, Any]) -> dict[str, Any]:
    operation = payload["operation"]
    graph = payload["workflow_graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    parent_hash = payload["parent_hash"]
    by_id = {node["id"]: node for node in nodes}
    if operation == "insert":
        edge = next(edge for edge in edges if by_id[edge["source"]]["type"] == "generator")
        return {
            "operation": "insert",
            "node": {
                "id": f"synthetic_o2_{parent_hash[:12]}",
                "type": "executor",
                "config": {"synthetic_mock": True},
            },
            "source": edge["source"],
            "target": edge["target"],
        }
    if operation == "delete":
        candidates = [
            node
            for node in nodes
            if node["id"] not in {graph["entry_id"], *graph["exit_ids"]}
            and sum(edge["target"] == node["id"] for edge in edges) == 1
            and sum(edge["source"] == node["id"] for edge in edges) == 1
        ]
        node = next(
            (node for node in candidates if node["type"] == "executor"),
            next((node for node in candidates if node["type"] == "generator"), None),
        )
        if node is None:
            raise ValueError("O2 mock found no safely reconnectable node")
        return {"operation": "delete", "node_id": node["id"]}
    if operation == "replace":
        original = next(node for node in nodes if node["type"] == "generator")
        node = deepcopy(original)
        node["config"] = {
            **node["config"],
            "synthetic_o2_mock": {"parent_sha256": parent_hash, "revision": 1},
        }
        return {"operation": "replace", "node_id": node["id"], "node": node}
    if operation == "reorder":
        return {
            "operation": "reorder",
            "node_ids": [node["id"] for node in reversed(nodes)],
        }
    raise ValueError(f"unsupported O2 mock operation: {operation}")


class E1OperatorMockAdapter(MockModelAdapter):
    def _generate_once(
        self,
        request: ModelRequest,
        timeout_seconds: float | None,
    ) -> ModelResponse:
        payload = json.loads(request.prompt)
        if "component" in payload:
            content = {
                "component": payload["component"],
                "value": _o1_mock_value(
                    payload["component"], payload["current_value"], payload["parent_hash"]
                ),
            }
        else:
            content = _o2_mock_patch(payload)
        self._script[request.prompt] = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return super()._generate_once(request, timeout_seconds)


def generate_child(
    episode: EpisodePlan,
    parent_genomes: Mapping[str, Genome],
    *,
    model_config: MockModelConfig,
    model: ModelAdapter | None = None,
) -> GenerationAttempt:
    if episode.role != "child" or episode.operator_id is None:
        raise ValueError("generate_child requires an E1 child episode")
    if episode.primary_parent_index is None:
        raise ValueError("child episode requires a primary parent")
    primary_id = GENOME_IDS[episode.primary_parent_index]
    parents = [parent_genomes[GENOME_IDS[index]] for index in episode.parent_indices]
    context = OperatorContext(
        model=_mock_model(model_config) if model is None else model,
        model_id=model_config.model_id,
        child_id=f"{episode.episode_id}-child",
        seed=episode.environment_seed,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout_seconds=model_config.timeout_seconds,
        max_retries=model_config.max_retries,
        price_id=model_config.price_id,
    )
    if episode.operator_id == "O1":
        result = O1LocalOperator().generate(
            parents[0], MutationSpec(component=episode.target_component), context
        )
    elif episode.operator_id == "O2":
        result = O2StructuralOperator().generate(
            parents[0], MutationSpec(operation=episode.operation), context
        )
    elif episode.operator_id == "O3":
        result = O3CrossoverOperator().generate(
            parents, MutationSpec(component_sources=episode.component_sources), context
        )
    else:
        raise ValueError(f"unsupported E1 operator: {episode.operator_id}")
    return GenerationAttempt(
        result=result,
        operator_id=episode.operator_id,
        parent_ids=tuple(parent.identity["genome_id"] for parent in parents),
        primary_parent_id=primary_id,
    )


class SyntheticE1Environment:
    def __init__(self, task_id: str, instruction: str, seed: int) -> None:
        self.task_id = task_id
        self.instruction = instruction
        self.seed = seed
        self.reset_calls: list[str] = []
        self.executed: list[str] = []
        self.evaluation_result: dict[str, Any] | None = None
        self.closed = False

    def reset(self, task_id: str) -> dict[str, str]:
        if task_id != self.task_id:
            raise ValueError("synthetic environment received an unexpected task_id")
        self.reset_calls.append(task_id)
        return {"task_id": task_id, "instruction": self.instruction}

    def execute(self, code: str) -> str:
        self.executed.append(code)
        return "synthetic E1 mock execution output"

    def evaluate(self) -> dict[str, Any]:
        tests = [
            {
                "test_id": "synthetic-mock:task-completion",
                "passed": True,
                "no_op": False,
            },
            {"test_id": "synthetic-mock:no-op", "passed": False, "no_op": True},
        ]
        self.evaluation_result = {
            "success": True,
            "report": "synthetic E1 mock evaluation only",
            "details": {
                "synthetic_mock": True,
                "task_id": self.task_id,
                "seed": self.seed,
                "official_tests": tests,
            },
        }
        return self.evaluation_result

    def trajectory(self) -> list[dict[str, str]]:
        return []

    def close(self) -> None:
        self.closed = True


@dataclass
class E1MockRunResult:
    config: E1MockConfig
    plan: E1Plan
    ledger: RunLedger
    episodes: list[EpisodeRecord]
    parent_inputs: dict[str, ParentEpisodeStateInput]
    pre_child_states: dict[int, PreChildState]
    events: list[ProtocolEvent]


def _parent_state_input(
    episode: EpisodePlan,
    record: EpisodeRecord,
    environment: SyntheticE1Environment,
) -> ParentEpisodeStateInput:
    if episode.initial_genome_hash is None:
        raise ValueError("parent control requires an initial Genome hash")
    evaluation = environment.evaluation_result
    tests = (
        [OfficialTestResult.model_validate(test) for test in evaluation["details"]["official_tests"]]
        if evaluation is not None
        else None
    )
    run_status = {
        "succeeded": "completed",
        "task_failed": "completed",
        "timeout": "timeout",
        "max_steps_exceeded": "budget_exhausted",
    }.get(record.run_status, "failed")
    events: list[ParentStateEvent] = []
    if environment.reset_calls:
        events.append(
            ParentStateEvent(
                step=0,
                dimension="state_dependency",
                outcome="observed",
                source="trajectory",
            )
        )
        events.append(
            ParentStateEvent(
                step=1,
                dimension="planning",
                outcome="observed",
                source="trajectory",
            )
        )
    if environment.executed:
        events.extend(
            [
                ParentStateEvent(
                    step=2,
                    dimension="tool",
                    outcome="observed",
                    source="trajectory",
                ),
                ParentStateEvent(
                    step=3,
                    dimension="code_execution",
                    outcome="observed",
                    source="trajectory",
                ),
            ]
        )
    return ParentEpisodeStateInput(
        block_index=episode.block_index,
        episode_id=episode.episode_id,
        genome_hash=episode.initial_genome_hash,
        events=events,
        run_status=run_status,
        official_success=None if evaluation is None else evaluation["success"],
        official_tests=tests,
        progress_unavailable_reason=(
            None
            if evaluation is not None
            else f"synthetic_evaluation_not_reached:{record.run_status}"
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _serialize_state(
    ledger: RunLedger,
    config: E1MockConfig,
    state: PreChildState,
) -> PreChildState:
    path = ledger.run_dir / "states" / f"block-{state.block_index:02d}-pre-child.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": 1,
        "evaluation_provenance": "synthetic_mock_only",
        "source_protocol_config": config.source_protocol.config_path,
        "source_protocol_config_sha256": config.source_protocol.config_sha256,
        "source_protocol_plan_sha256": config.source_protocol.plan_sha256,
        "state": state.model_dump(mode="json"),
    }
    path.write_text(_canonical_json(envelope) + "\n", encoding="utf-8")
    restored_envelope = json.loads(path.read_text(encoding="utf-8"))
    restored = PreChildState.model_validate(restored_envelope["state"])
    return validate_pre_child_state(restored)


def _append_protocol_event(
    ledger: RunLedger,
    events: list[ProtocolEvent],
    event: ProtocolEvent,
) -> None:
    validate_event_order([*events, event])
    events.append(event)
    with (ledger.run_dir / "protocol_events.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as destination:
        destination.write(_canonical_json(event.model_dump(mode="json")) + "\n")


def _initial_genomes(plan: E1Plan, repo_root: Path) -> dict[str, Genome]:
    genomes = {
        genome_id: validate_genome(
            (repo_root / plan.config.genomes_directory / f"{genome_id}.json").read_bytes()
        )
        for genome_id in GENOME_IDS
    }
    actual_hashes = {genome_id: genome_hash(genome) for genome_id, genome in genomes.items()}
    if actual_hashes != plan.config.initial_genome_hashes:
        raise ValueError("initial Genome hashes do not match the source protocol")
    return genomes


def run_e1_mock(
    config_path: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    allow_dirty: bool = False,
) -> E1MockRunResult:
    repository = Path(repo_root).resolve()
    mock_config_path = Path(config_path).resolve()
    config = load_mock_config(mock_config_path)
    source_path = repository / config.source_protocol.config_path
    plan = build_plan_from_file(source_path, repo_root=repository)
    if plan.config_sha256 != config.source_protocol.config_sha256:
        raise ValueError("source protocol config hash does not match the mock config")
    if plan.plan_sha256 != config.source_protocol.plan_sha256:
        raise ValueError("source protocol plan hash does not match the mock config")
    provenance = capture_provenance(
        mock_config_path,
        repo_root=repository,
        allow_dirty=allow_dirty,
    )
    ledger = RunLedger(results_root, config.run_id, provenance=provenance)
    genomes = _initial_genomes(plan, repository)
    model = WorkflowMockAdapter(
        model_version=config.model.model_version,
        timeout_seconds=config.model.timeout_seconds,
        max_retries=config.model.max_retries,
        price_id=config.model.price_id,
        input_price_per_million=config.model.input_price_per_million,
        output_price_per_million=config.model.output_price_per_million,
    )
    runner = AgentRunner(
        model=model,
        model_id=config.model.model_id,
        ledger=ledger,
        provenance=provenance,
        max_steps=config.runner.max_steps,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        model_timeout_seconds=config.model.timeout_seconds,
        max_retries=config.model.max_retries,
        price_id=config.model.price_id,
        timeout_seconds=config.runner.timeout_seconds,
        process_isolation=config.runner.process_isolation,
    )
    episodes: list[EpisodeRecord] = []
    parent_inputs: dict[str, ParentEpisodeStateInput] = {}
    states: dict[int, PreChildState] = {}
    events: list[ProtocolEvent] = []

    for block in plan.blocks[: config.limits.max_blocks]:
        block_event_start = len(events)
        current_parent_inputs: dict[str, ParentEpisodeStateInput] = {}
        for episode_id, evaluation_event_id in zip(
            block.parent_episode_ids,
            block.parent_evaluation_event_ids,
            strict=True,
        ):
            episode = next(item for item in plan.episodes if item.episode_id == episode_id)
            environment = SyntheticE1Environment(
                episode.task_id,
                config.environment.instruction,
                episode.environment_seed,
            )
            record = runner.run_episode(
                genome=genomes[episode.initial_genome_id or ""],
                environment=environment,
                task_id=episode.task_id,
                task_family=episode.task_family,
                seed=episode.environment_seed,
                experiment_id=config.experiment_id,
                episode_id=episode.episode_id,
            )
            episodes.append(record)
            parent_input = _parent_state_input(episode, record, environment)
            current_parent_inputs[episode_id] = parent_input
            parent_inputs[episode_id] = parent_input
            _append_protocol_event(
                ledger,
                events,
                ProtocolEvent(
                    event_id=evaluation_event_id,
                    block_index=block.block_index,
                    kind="parent_evaluated",
                    episode_id=episode_id,
                ),
            )

        state = build_pre_child_state(block, current_parent_inputs)
        state = _serialize_state(ledger, config, state)
        states[block.block_index] = state
        expected_events = build_block_events(plan, block.block_index, pre_child_state=state)
        if events[block_event_start:] != expected_events[:4]:
            raise ValueError("parent evaluation events do not match the source plan")
        _append_protocol_event(ledger, events, expected_events[4])

        for child_position, child_id in enumerate(block.child_episode_ids):
            child = next(item for item in plan.episodes if item.episode_id == child_id)
            attempt = generate_child(child, genomes, model_config=config.model)
            generated_event = expected_events[5 + child_position * 2]
            _append_protocol_event(ledger, events, generated_event)
            primary_id = GENOME_IDS[child.primary_parent_index or 0]
            environment = SyntheticE1Environment(
                child.task_id,
                config.environment.instruction,
                child.environment_seed,
            )
            record = runner.run_episode(
                genome=attempt.result.child or genomes[primary_id],
                environment=environment,
                task_id=child.task_id,
                task_family=child.task_family,
                seed=child.environment_seed,
                experiment_id=config.experiment_id,
                episode_id=child.episode_id,
                generation=attempt,
            )
            episodes.append(record)
            _append_protocol_event(
                ledger,
                events,
                expected_events[6 + child_position * 2],
            )

        if events[block_event_start:] != expected_events:
            raise ValueError("block protocol events do not match the source plan")
        validate_event_order(events)

    if len(episodes) != config.limits.max_episodes:
        raise ValueError("executed episode count does not match the configured limit")
    return E1MockRunResult(
        config=config,
        plan=plan,
        ledger=ledger,
        episodes=episodes,
        parent_inputs=parent_inputs,
        pre_child_states=states,
        events=events,
    )


__all__ = [
    "E1MockConfig",
    "E1MockRunResult",
    "E1OperatorMockAdapter",
    "MockModelConfig",
    "SyntheticE1Environment",
    "generate_child",
    "load_mock_config",
    "run_e1_mock",
]
