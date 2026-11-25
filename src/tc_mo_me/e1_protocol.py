from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, pvariance
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from tc_mo_me.genome import CONTENT_COMPONENTS, genome_hash, validate_genome

ENVIRONMENT_SEEDS = (100, 200)
GENOME_IDS = ("g0_baseline", "g1_planner", "g2_verifier", "g3_workflow")
O1_COMPONENTS = (
    "system_prompt",
    "planning_policy",
    "code_generation_policy",
    "verification_policy",
)
O2_OPERATIONS = ("insert", "delete", "replace", "reorder")

Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )


class TaskSpec(StrictModel):
    task_id: str = Field(pattern=r"^[0-9a-f]{7}_[123]$")
    task_family: str = Field(pattern=r"^[0-9a-f]{7}$")
    metadata_sha256: Sha256

    @model_validator(mode="after")
    def validate_family(self) -> Self:
        if self.task_id.rsplit("_", 1)[0] != self.task_family:
            raise ValueError("task_family must be the task_id prefix before the final underscore")
        return self


class SelectionSpec(StrictModel):
    dataset_split: Literal["train"]
    selection_seed: int
    task_family_rule: Literal["task_id_prefix_before_final_underscore"]
    task_type_rule: Literal["required_apps_cardinality_single_1_multi_greater_than_1"]
    stratum_rule: Literal["exact_sorted_required_apps_set"]
    strata_ranking_rule: Literal["descending_task_count_then_app_set_lexicographic"]
    strata_per_task_type: Literal[4]
    within_stratum_rule: Literal["minimum_sha256_of_selection_seed_colon_task_id_then_task_id"]
    fold_rule: Literal["within_task_type_sort_task_id_then_alternate_A_B"]
    metadata_hash_rule: Literal["sha256_canonical_json_of_specs_and_required_apps"]
    train_list_sha256: Sha256
    selection_manifest_sha256: Sha256


class ModelSpec(StrictModel):
    model_id: NonEmptyString
    model_version: NonEmptyString | None
    seed_policy: Literal["same_as_environment_seed"]
    temperature: float = Field(ge=0, allow_inf_nan=False)


class EpisodeBudget(StrictModel):
    max_model_calls_per_episode: int = Field(gt=0)
    max_input_tokens_per_episode: int = Field(gt=0)
    max_output_tokens_per_episode: int = Field(gt=0)
    timeout_seconds_per_episode: int = Field(gt=0)
    max_retries_per_call: int = Field(ge=0)


class E1Config(StrictModel):
    experiment_id: Literal["e1_operator_heterogeneity_pilot"]
    protocol_version: Literal["1.1"]
    analysis_status: Literal["not_run"]
    train_role: Literal["pilot"]
    dev_role: Literal["reserved_independent_validation"]
    environment_seeds: list[int]
    genomes_directory: Literal["configs/e1/genomes"]
    model: ModelSpec
    budget: EpisodeBudget
    selection: SelectionSpec
    initial_genome_hashes: dict[str, Sha256]
    tasks: list[TaskSpec]

    @model_validator(mode="after")
    def validate_frozen_design(self) -> Self:
        if self.environment_seeds != list(ENVIRONMENT_SEEDS):
            raise ValueError("environment_seeds must be [100, 200] in this order")
        if len(self.tasks) != 8:
            raise ValueError("E1 pilot requires exactly eight train tasks")
        task_ids = [task.task_id for task in self.tasks]
        families = [task.task_family for task in self.tasks]
        if len(set(task_ids)) != len(task_ids) or len(set(families)) != len(families):
            raise ValueError("task_id and task_family values must be unique")
        if set(self.initial_genome_hashes) != set(GENOME_IDS):
            raise ValueError("initial_genome_hashes must cover exactly the four initial Genomes")
        return self


class EpisodePlan(StrictModel):
    episode_id: NonEmptyString
    block_index: int = Field(ge=0, le=15)
    task_id: NonEmptyString
    task_family: NonEmptyString
    environment_seed: Literal[100, 200]
    model_id: NonEmptyString
    budget: EpisodeBudget
    role: Literal["parent_control", "child"]
    initial_genome_id: NonEmptyString | None = None
    initial_genome_hash: Sha256 | None = None
    operator_id: Literal["O1", "O2", "O3"] | None = None
    primary_parent_index: int | None = Field(default=None, ge=0, le=3)
    parent_indices: list[int] = Field(default_factory=list)
    donor_indices: list[int] = Field(default_factory=list)
    target_component: (
        Literal[
            "system_prompt",
            "planning_policy",
            "code_generation_policy",
            "verification_policy",
        ]
        | None
    ) = None
    operation: Literal["insert", "delete", "replace", "reorder"] | None = None
    component_sources: dict[str, NonEmptyString] = Field(default_factory=dict)
    depends_on_episode_ids: list[NonEmptyString] = Field(default_factory=list)
    depends_on_event_ids: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if any(index < 0 or index > 3 for index in self.parent_indices + self.donor_indices):
            raise ValueError("parent and donor indices must be in 0..3")
        if self.role == "parent_control":
            if self.initial_genome_id is None or self.initial_genome_hash is None:
                raise ValueError("parent controls require an initial Genome and hash")
            if (
                self.operator_id is not None
                or self.primary_parent_index is not None
                or self.parent_indices
                or self.donor_indices
                or self.component_sources
                or self.depends_on_episode_ids
                or self.depends_on_event_ids
            ):
                raise ValueError("parent controls cannot have child lineage or dependencies")
        else:
            if self.initial_genome_id is not None or self.initial_genome_hash is not None:
                raise ValueError("children cannot claim an unmodified initial Genome")
            if self.operator_id is None or self.primary_parent_index is None:
                raise ValueError("children require an operator and primary parent")
            if len(self.depends_on_episode_ids) != 4 or len(self.depends_on_event_ids) != 5:
                raise ValueError("children depend on four parent evaluations and one state event")
        return self


class BlockPlan(StrictModel):
    block_index: int = Field(ge=0, le=15)
    task_id: NonEmptyString
    task_family: NonEmptyString
    environment_seed: Literal[100, 200]
    model_id: NonEmptyString
    budget: EpisodeBudget
    episode_ids: list[NonEmptyString]
    parent_episode_ids: list[NonEmptyString]
    child_episode_ids: list[NonEmptyString]
    parent_evaluation_event_ids: list[NonEmptyString]
    pre_child_state_event_id: NonEmptyString


class E1Plan(StrictModel):
    config: E1Config
    config_sha256: Sha256
    blocks: list[BlockPlan]
    episodes: list[EpisodePlan]
    plan_sha256: Sha256


class ProtocolEvent(StrictModel):
    event_id: NonEmptyString
    block_index: int = Field(ge=0, le=15)
    kind: Literal[
        "parent_evaluated",
        "pre_child_state_serialized",
        "child_generated",
        "child_evaluated",
        "child_finalized",
    ]
    episode_id: NonEmptyString | None = None
    depends_on_event_ids: list[NonEmptyString] = Field(default_factory=list)
    payload_sha256: Sha256 | None = None


class ParentStateEvent(StrictModel):
    step: int = Field(ge=0)
    dimension: Literal["planning", "tool", "state_dependency", "code_execution"]
    outcome: Literal["observed", "failed"]
    source: Literal["trajectory", "appworld_error"]
    error_type: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        if self.outcome == "failed" and self.error_type is None:
            raise ValueError("failed parent events require error_type")
        if self.outcome == "observed" and self.error_type is not None:
            raise ValueError("observed parent events cannot contain error_type")
        return self


class OfficialTestResult(StrictModel):
    test_id: NonEmptyString
    passed: bool
    no_op: bool = False


class ParentEpisodeStateInput(StrictModel):
    block_index: int = Field(ge=0, le=15)
    episode_id: NonEmptyString
    genome_hash: Sha256
    events: list[ParentStateEvent]
    run_status: Literal["completed", "failed", "timeout", "budget_exhausted"]
    official_success: bool | None
    official_tests: list[OfficialTestResult] | None
    progress_unavailable_reason: NonEmptyString | None

    @model_validator(mode="after")
    def validate_parent_input(self) -> Self:
        steps = [event.step for event in self.events]
        if any(later <= earlier for earlier, later in zip(steps, steps[1:], strict=False)):
            raise ValueError("parent event steps must be strictly increasing")
        if self.official_tests is None:
            if self.progress_unavailable_reason is None:
                raise ValueError("missing official tests require progress_unavailable_reason")
        elif self.progress_unavailable_reason is not None:
            raise ValueError("official tests and progress_unavailable_reason are exclusive")
        test_ids = [test.test_id for test in self.official_tests or []]
        if len(set(test_ids)) != len(test_ids):
            raise ValueError("official test_id values must be unique")
        return self


class TaskProgress(StrictModel):
    value: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    null_reason: NonEmptyString | None

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if (self.value is None) == (self.null_reason is None):
            raise ValueError("task progress requires exactly one of value or null_reason")
        return self


class TaskState(StrictModel):
    planning: Literal["not_observed", "observed", "failed"]
    tool: Literal["not_observed", "observed", "failed"]
    state_dependency: Literal["not_observed", "observed", "failed"]
    code_execution: Literal["not_observed", "observed", "failed"]
    task_incomplete: Literal["complete", "incomplete", "unknown"]
    task_progress: TaskProgress


class PopulationState(StrictModel):
    mean: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    best: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    variance: float | None = Field(ge=0, allow_inf_nan=False)
    genome_diversity: float = Field(ge=0, le=1, allow_inf_nan=False)
    stagnation: Literal["insufficient_history"]


class PreChildState(StrictModel):
    block_index: int = Field(ge=0, le=15)
    parent_episode_ids: list[NonEmptyString]
    parent_genome_hashes: dict[NonEmptyString, Sha256]
    task_states: dict[NonEmptyString, TaskState]
    population: PopulationState
    population_null_reason: NonEmptyString | None
    source_inputs_sha256: Sha256
    state_sha256: Sha256

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if len(self.parent_episode_ids) != 4 or len(set(self.parent_episode_ids)) != 4:
            raise ValueError("pre-child state requires exactly four unique parent episodes")
        if set(self.task_states) != set(self.parent_episode_ids):
            raise ValueError("task_states must cover exactly the four parent episodes")
        if set(self.parent_genome_hashes) != set(self.parent_episode_ids):
            raise ValueError("parent_genome_hashes must cover exactly the four parents")
        aggregate_values = (
            self.population.mean,
            self.population.best,
            self.population.variance,
        )
        if any(value is None for value in aggregate_values):
            if not all(value is None for value in aggregate_values):
                raise ValueError("population aggregates must be jointly available or null")
            if self.population_null_reason is None:
                raise ValueError("null population aggregates require a reason")
        elif self.population_null_reason is not None:
            raise ValueError("available population aggregates cannot have a null reason")
        return self


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def load_config(path: str | Path) -> E1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E1 config must be a YAML mapping")
    return E1Config.model_validate(payload)


def _load_and_check_genomes(config: E1Config, repo_root: Path) -> dict[str, str]:
    genome_directory = repo_root / config.genomes_directory
    paths = {path.stem: path for path in genome_directory.glob("*.json")}
    if set(paths) != set(GENOME_IDS):
        raise ValueError(
            "genomes_directory must contain exactly the four initial Genome JSON files"
        )
    actual_hashes = {
        genome_id: genome_hash(validate_genome(path.read_bytes()))
        for genome_id, path in paths.items()
    }
    if actual_hashes != config.initial_genome_hashes:
        raise ValueError("initial_genome_hashes do not match the Genome content hashes")
    return actual_hashes


def _parent_episode(
    config: E1Config,
    task: TaskSpec,
    seed: int,
    block_index: int,
    parent_index: int,
) -> EpisodePlan:
    genome_id = GENOME_IDS[parent_index]
    return EpisodePlan(
        episode_id=f"e1-b{block_index:02d}-p{parent_index}",
        block_index=block_index,
        task_id=task.task_id,
        task_family=task.task_family,
        environment_seed=seed,
        model_id=config.model.model_id,
        budget=config.budget,
        role="parent_control",
        initial_genome_id=genome_id,
        initial_genome_hash=config.initial_genome_hashes[genome_id],
    )


def _child_episode(
    config: E1Config,
    task: TaskSpec,
    seed: int,
    block_index: int,
    operator_id: Literal["O1", "O2", "O3"],
    parent_episode_ids: list[str],
    parent_evaluation_event_ids: list[str],
    state_event_id: str,
) -> EpisodePlan:
    primary = block_index % 4
    primary_id = GENOME_IDS[primary]
    parent_indices = [primary]
    donor_indices: list[int] = []
    target_component = None
    operation = None
    sources = {component: primary_id for component in CONTENT_COMPONENTS}
    if operator_id == "O1":
        target_component = O1_COMPONENTS[block_index % 4]
    elif operator_id == "O2":
        operation = O2_OPERATIONS[block_index % 4]
    else:
        parent_indices = [(block_index + offset) % 4 for offset in range(4)]
        donor_indices = parent_indices[1:]
        if primary in (0, 3):
            sources["planning_policy"] = "g1_planner"
            sources["verification_policy"] = "g2_verifier"
        else:
            sources["workflow_graph"] = "g3_workflow"
    return EpisodePlan(
        episode_id=f"e1-b{block_index:02d}-{operator_id.lower()}",
        block_index=block_index,
        task_id=task.task_id,
        task_family=task.task_family,
        environment_seed=seed,
        model_id=config.model.model_id,
        budget=config.budget,
        role="child",
        operator_id=operator_id,
        primary_parent_index=primary,
        parent_indices=parent_indices,
        donor_indices=donor_indices,
        target_component=target_component,
        operation=operation,
        component_sources=sources,
        depends_on_episode_ids=parent_episode_ids.copy(),
        depends_on_event_ids=[*parent_evaluation_event_ids, state_event_id],
    )


def build_plan(config: E1Config, *, repo_root: str | Path) -> E1Plan:
    config = E1Config.model_validate(config.model_dump())
    _load_and_check_genomes(config, Path(repo_root))
    config_sha256 = _sha256(config.model_dump(mode="json"))
    blocks: list[BlockPlan] = []
    episodes: list[EpisodePlan] = []
    task_seed_pairs = [
        (task, seed)
        for task in sorted(config.tasks, key=lambda item: item.task_id)
        for seed in config.environment_seeds
    ]
    for block_index, (task, seed) in enumerate(task_seed_pairs):
        parents = [
            _parent_episode(config, task, seed, block_index, parent_index)
            for parent_index in range(4)
        ]
        parent_ids = [episode.episode_id for episode in parents]
        parent_event_ids = [f"{episode_id}:evaluated" for episode_id in parent_ids]
        state_event_id = f"e1-b{block_index:02d}:pre-child-state-serialized"
        children = [
            _child_episode(
                config,
                task,
                seed,
                block_index,
                operator_id,
                parent_ids,
                parent_event_ids,
                state_event_id,
            )
            for operator_id in ("O1", "O2", "O3")
        ]
        block_episodes = [*parents, *children]
        episodes.extend(block_episodes)
        blocks.append(
            BlockPlan(
                block_index=block_index,
                task_id=task.task_id,
                task_family=task.task_family,
                environment_seed=seed,
                model_id=config.model.model_id,
                budget=config.budget,
                episode_ids=[episode.episode_id for episode in block_episodes],
                parent_episode_ids=parent_ids,
                child_episode_ids=[episode.episode_id for episode in children],
                parent_evaluation_event_ids=parent_event_ids,
                pre_child_state_event_id=state_event_id,
            )
        )
    payload = {
        "config": config.model_dump(mode="json"),
        "config_sha256": config_sha256,
        "blocks": [block.model_dump(mode="json") for block in blocks],
        "episodes": [episode.model_dump(mode="json") for episode in episodes],
    }
    return E1Plan(**payload, plan_sha256=_sha256(payload))


def build_plan_from_file(path: str | Path, *, repo_root: str | Path) -> E1Plan:
    return build_plan(load_config(path), repo_root=repo_root)


def validate_plan(plan: E1Plan, *, repo_root: str | Path) -> E1Plan:
    validated = E1Plan.model_validate(plan.model_dump())
    expected = build_plan(validated.config, repo_root=repo_root)
    if validated != expected:
        raise ValueError("E1 plan does not match the frozen config and generation rules")
    return expected


def episode_by_id(plan: E1Plan, episode_id: str) -> EpisodePlan:
    matches = [episode for episode in plan.episodes if episode.episode_id == episode_id]
    if len(matches) != 1:
        raise KeyError(episode_id)
    return matches[0]


def derive_task_state(parent_input: ParentEpisodeStateInput) -> TaskState:
    parent_input = ParentEpisodeStateInput.model_validate(parent_input.model_dump())
    dimensions = {
        "planning": "not_observed",
        "tool": "not_observed",
        "state_dependency": "not_observed",
        "code_execution": "not_observed",
    }
    for event in parent_input.events:
        dimensions[event.dimension] = event.outcome

    if parent_input.official_tests is None:
        progress = TaskProgress(
            value=None,
            null_reason=parent_input.progress_unavailable_reason,
        )
    else:
        target_tests = [test for test in parent_input.official_tests if not test.no_op]
        if target_tests:
            progress = TaskProgress(
                value=sum(test.passed for test in target_tests) / len(target_tests),
                null_reason=None,
            )
        else:
            progress = TaskProgress(value=None, null_reason="no_non_noop_target_tests")

    if parent_input.official_success is not None:
        incomplete = "complete" if parent_input.official_success else "incomplete"
    elif parent_input.run_status == "completed":
        incomplete = "unknown"
    else:
        incomplete = "incomplete"
    return TaskState(**dimensions, task_incomplete=incomplete, task_progress=progress)


def _summarize_population(
    parent_episode_ids: list[str],
    parent_genome_hashes: dict[str, str],
    task_states: dict[str, TaskState],
) -> tuple[PopulationState, str | None]:
    missing = [
        episode_id
        for episode_id in parent_episode_ids
        if task_states[episode_id].task_progress.value is None
    ]
    null_reason = None
    if missing:
        mean = best = variance = None
        null_reason = "parent_task_progress_unavailable:" + ",".join(missing)
    else:
        values = [task_states[episode_id].task_progress.value for episode_id in parent_episode_ids]
        if any(value is None for value in values):
            raise AssertionError("missing progress was not identified")
        numeric_values = [value for value in values if value is not None]
        mean = fmean(numeric_values)
        best = max(numeric_values)
        variance = pvariance(numeric_values)
    population = PopulationState(
        mean=mean,
        best=best,
        variance=variance,
        genome_diversity=len(set(parent_genome_hashes.values())) / len(parent_episode_ids),
        stagnation="insufficient_history",
    )
    return population, null_reason


def build_pre_child_state(
    block: BlockPlan,
    parent_inputs: Mapping[str, ParentEpisodeStateInput],
) -> PreChildState:
    selected = []
    for episode_id in block.parent_episode_ids:
        if episode_id not in parent_inputs:
            raise ValueError(f"missing parent state input: {episode_id}")
        parent_input = ParentEpisodeStateInput.model_validate(
            parent_inputs[episode_id].model_dump()
        )
        if parent_input.episode_id != episode_id or parent_input.block_index != block.block_index:
            raise ValueError("parent state input does not match the requested block episode")
        selected.append(parent_input)

    parent_ids = block.parent_episode_ids.copy()
    task_states = {
        parent_input.episode_id: derive_task_state(parent_input) for parent_input in selected
    }
    genome_hashes = {parent_input.episode_id: parent_input.genome_hash for parent_input in selected}
    population, null_reason = _summarize_population(parent_ids, genome_hashes, task_states)
    source_inputs_sha256 = _sha256(
        [parent_input.model_dump(mode="json") for parent_input in selected]
    )
    payload = {
        "block_index": block.block_index,
        "parent_episode_ids": parent_ids,
        "parent_genome_hashes": genome_hashes,
        "task_states": {
            episode_id: task_state.model_dump(mode="json")
            for episode_id, task_state in task_states.items()
        },
        "population": population.model_dump(mode="json"),
        "population_null_reason": null_reason,
        "source_inputs_sha256": source_inputs_sha256,
    }
    return PreChildState(**payload, state_sha256=_sha256(payload))


def canonical_pre_child_state_bytes(state: PreChildState) -> bytes:
    return _canonical_bytes(state.model_dump(mode="json", exclude={"state_sha256"}))


def pre_child_state_hash(state: PreChildState) -> str:
    return hashlib.sha256(canonical_pre_child_state_bytes(state)).hexdigest()


def validate_pre_child_state(state: PreChildState) -> PreChildState:
    validated = PreChildState.model_validate(state.model_dump())
    if pre_child_state_hash(validated) != validated.state_sha256:
        raise ValueError("pre-child state hash does not match its canonical serialization")
    population, null_reason = _summarize_population(
        validated.parent_episode_ids,
        validated.parent_genome_hashes,
        validated.task_states,
    )
    if validated.population != population or validated.population_null_reason != null_reason:
        raise ValueError("pre-child population state does not match parent task states")
    return validated


def build_block_events(
    plan: E1Plan,
    block_index: int,
    *,
    pre_child_state: PreChildState,
) -> list[ProtocolEvent]:
    block = plan.blocks[block_index]
    if block.block_index != block_index:
        raise ValueError("blocks must be stored in block_index order")
    validated_state = validate_pre_child_state(pre_child_state)
    if (
        validated_state.block_index != block_index
        or validated_state.parent_episode_ids != block.parent_episode_ids
    ):
        raise ValueError("pre-child state does not match the requested block")
    state_payload_hash = validated_state.state_sha256
    events = [
        ProtocolEvent(
            event_id=event_id,
            block_index=block_index,
            kind="parent_evaluated",
            episode_id=episode_id,
        )
        for event_id, episode_id in zip(
            block.parent_evaluation_event_ids,
            block.parent_episode_ids,
            strict=True,
        )
    ]
    events.append(
        ProtocolEvent(
            event_id=block.pre_child_state_event_id,
            block_index=block_index,
            kind="pre_child_state_serialized",
            depends_on_event_ids=block.parent_evaluation_event_ids.copy(),
            payload_sha256=state_payload_hash,
        )
    )
    for child_id in block.child_episode_ids:
        child = episode_by_id(plan, child_id)
        generated_event_id = f"{child_id}:generated"
        events.append(
            ProtocolEvent(
                event_id=generated_event_id,
                block_index=block_index,
                kind="child_generated",
                episode_id=child_id,
                depends_on_event_ids=child.depends_on_event_ids.copy(),
            )
        )
        events.append(
            ProtocolEvent(
                event_id=f"{child_id}:evaluated",
                block_index=block_index,
                kind="child_evaluated",
                episode_id=child_id,
                depends_on_event_ids=[generated_event_id],
            )
        )
    validate_event_order(events)
    return events


def _validate_event_identity(event: ProtocolEvent) -> None:
    block_id = f"e1-b{event.block_index:02d}"
    if event.kind == "parent_evaluated":
        expected_episodes = {f"{block_id}-p{index}" for index in range(4)}
        expected_event_id = f"{event.episode_id}:evaluated"
    elif event.kind == "pre_child_state_serialized":
        if event.episode_id is not None:
            raise ValueError("pre-child state events cannot reference an episode")
        expected_event_id = f"{block_id}:pre-child-state-serialized"
    elif event.kind in {"child_generated", "child_evaluated", "child_finalized"}:
        expected_episodes = {f"{block_id}-o{index}" for index in range(1, 4)}
        suffix = {
            "child_generated": "generated",
            "child_evaluated": "evaluated",
            "child_finalized": "finalized",
        }[event.kind]
        expected_event_id = f"{event.episode_id}:{suffix}"
    else:
        raise ValueError(f"unsupported protocol event kind: {event.kind}")

    if event.kind != "pre_child_state_serialized" and event.episode_id not in expected_episodes:
        raise ValueError(
            f"event {event.event_id!r} has an invalid episode for block {event.block_index}"
        )
    if event.event_id != expected_event_id:
        raise ValueError(
            f"event identity mismatch: expected {expected_event_id!r}, got {event.event_id!r}"
        )


def _validate_event_dependencies(
    event: ProtocolEvent,
    dependencies: list[ProtocolEvent],
) -> None:
    block_id = f"e1-b{event.block_index:02d}"
    parent_event_ids = [f"{block_id}-p{index}:evaluated" for index in range(4)]
    state_event_id = f"{block_id}:pre-child-state-serialized"
    if event.kind == "parent_evaluated":
        if dependencies:
            raise ValueError("parent_evaluated events cannot have dependencies")
        return
    if event.kind == "pre_child_state_serialized":
        if event.depends_on_event_ids != parent_event_ids:
            raise ValueError("pre-child state requires this block's four parent evaluations")
        if any(dependency.kind != "parent_evaluated" for dependency in dependencies):
            raise ValueError("pre-child state dependencies must be parent_evaluated events")
        if event.payload_sha256 is None:
            raise ValueError("serialized pre-child state requires a payload hash")
    elif event.kind == "child_generated":
        if event.depends_on_event_ids != [*parent_event_ids, state_event_id]:
            raise ValueError(
                "child generation requires this block's four parent evaluations and state"
            )
        expected_kinds = [
            "parent_evaluated",
            "parent_evaluated",
            "parent_evaluated",
            "parent_evaluated",
            "pre_child_state_serialized",
        ]
        if [dependency.kind for dependency in dependencies] != expected_kinds:
            raise ValueError("child generation has invalid dependency event types")
    else:
        if len(dependencies) != 1:
            raise ValueError("child terminal events require exactly one child_generated event")
        generated = dependencies[0]
        if (
            generated.kind != "child_generated"
            or generated.block_index != event.block_index
            or generated.episode_id != event.episode_id
        ):
            raise ValueError("child terminal events must match their generated episode and block")
    if any(dependency.block_index != event.block_index for dependency in dependencies):
        raise ValueError("protocol event dependencies cannot cross blocks")


def validate_event_order(events: list[ProtocolEvent]) -> None:
    event_ids = [event.event_id for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event_id values must be unique")
    seen: dict[str, ProtocolEvent] = {}
    terminal_episodes: set[tuple[int, str]] = set()
    for event in events:
        _validate_event_identity(event)
        missing = set(event.depends_on_event_ids) - set(seen)
        if missing:
            raise ValueError(
                f"event {event.event_id!r} depends on events that do not precede it: "
                f"{sorted(missing)}"
            )
        dependencies = [seen[event_id] for event_id in event.depends_on_event_ids]
        _validate_event_dependencies(event, dependencies)
        if event.kind in {"child_evaluated", "child_finalized"}:
            if event.episode_id is None:
                raise ValueError("child terminal events require an episode")
            terminal_key = (event.block_index, event.episode_id)
            if terminal_key in terminal_episodes:
                raise ValueError("each child must have exactly one terminal event")
            terminal_episodes.add(terminal_key)
        seen[event.event_id] = event


__all__ = [
    "BlockPlan",
    "E1Config",
    "E1Plan",
    "EpisodeBudget",
    "EpisodePlan",
    "OfficialTestResult",
    "ParentEpisodeStateInput",
    "ParentStateEvent",
    "PopulationState",
    "PreChildState",
    "ProtocolEvent",
    "TaskProgress",
    "TaskSpec",
    "TaskState",
    "build_block_events",
    "build_plan",
    "build_plan_from_file",
    "build_pre_child_state",
    "canonical_pre_child_state_bytes",
    "derive_task_state",
    "episode_by_id",
    "genome_hash",
    "load_config",
    "pre_child_state_hash",
    "validate_event_order",
    "validate_genome",
    "validate_plan",
    "validate_pre_child_state",
]
