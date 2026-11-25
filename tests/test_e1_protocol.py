import importlib
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "e1" / "pilot.yaml"
GENOME_IDS = ("g0_baseline", "g1_planner", "g2_verifier", "g3_workflow")
O1_COMPONENTS = (
    "system_prompt",
    "planning_policy",
    "code_generation_policy",
    "verification_policy",
)
O2_OPERATIONS = ("insert", "delete", "replace", "reorder")
O3_COMPOSITIONS = (
    {
        "system_prompt": 0,
        "planning_policy": 1,
        "code_generation_policy": 0,
        "verification_policy": 2,
        "workflow_graph": 0,
    },
    {
        "system_prompt": 1,
        "planning_policy": 1,
        "code_generation_policy": 1,
        "verification_policy": 1,
        "workflow_graph": 3,
    },
    {
        "system_prompt": 2,
        "planning_policy": 2,
        "code_generation_policy": 2,
        "verification_policy": 2,
        "workflow_graph": 3,
    },
    {
        "system_prompt": 3,
        "planning_policy": 1,
        "code_generation_policy": 3,
        "verification_policy": 2,
        "workflow_graph": 3,
    },
)


def load_api():
    return importlib.import_module("tc_mo_me.e1_protocol")


def make_plan():
    api = load_api()
    return api, api.build_plan_from_file(CONFIG_PATH, repo_root=ROOT)


def test_plan_has_112_unique_episodes_and_16_seven_episode_blocks():
    _, plan = make_plan()

    assert len(plan.episodes) == 112
    assert len({episode.episode_id for episode in plan.episodes}) == 112
    assert Counter(episode.role for episode in plan.episodes) == {
        "parent_control": 64,
        "child": 48,
    }
    assert len(plan.blocks) == 16
    assert [block.block_index for block in plan.blocks] == list(range(16))
    assert all(len(block.episode_ids) == 7 for block in plan.blocks)
    assert all(len(block.parent_episode_ids) == 4 for block in plan.blocks)
    assert all(len(block.child_episode_ids) == 3 for block in plan.blocks)


def test_blocks_follow_sorted_task_then_seed_and_share_conditions():
    api, plan = make_plan()
    expected_tasks = sorted(task.task_id for task in plan.config.tasks)
    expected_pairs = [(task_id, seed) for task_id in expected_tasks for seed in (100, 200)]

    assert [(block.task_id, block.environment_seed) for block in plan.blocks] == expected_pairs
    for block in plan.blocks:
        episodes = [api.episode_by_id(plan, episode_id) for episode_id in block.episode_ids]
        assert {episode.task_id for episode in episodes} == {block.task_id}
        assert {episode.task_family for episode in episodes} == {block.task_family}
        assert {episode.environment_seed for episode in episodes} == {block.environment_seed}
        assert {episode.model_id for episode in episodes} == {block.model_id}
        assert {episode.budget for episode in episodes} == {block.budget}


def test_parent_and_donor_indices_are_balanced_across_blocks():
    _, plan = make_plan()
    by_block = {block.block_index: block for block in plan.blocks}

    for block_index, block in by_block.items():
        children = [
            next(episode for episode in plan.episodes if episode.episode_id == episode_id)
            for episode_id in block.child_episode_ids
        ]
        assert [episode.primary_parent_index for episode in children] == [block_index % 4] * 3
        o3 = children[2]
        assert o3.donor_indices == [
            (block_index + 1) % 4,
            (block_index + 2) % 4,
            (block_index + 3) % 4,
        ]

    children = [episode for episode in plan.episodes if episode.role == "child"]
    assert Counter(episode.primary_parent_index for episode in children) == {
        0: 12,
        1: 12,
        2: 12,
        3: 12,
    }
    o3_episodes = [episode for episode in children if episode.operator_id == "O3"]
    assert Counter(index for episode in o3_episodes for index in episode.donor_indices) == {
        0: 12,
        1: 12,
        2: 12,
        3: 12,
    }


def test_operator_specs_and_component_sources_are_frozen_per_block():
    _, plan = make_plan()

    for block in plan.blocks:
        b = block.block_index
        primary = b % 4
        donor_one = (b + 1) % 4
        donor_two = (b + 2) % 4
        donor_three = (b + 3) % 4
        children = [
            next(episode for episode in plan.episodes if episode.episode_id == episode_id)
            for episode_id in block.child_episode_ids
        ]
        o1, o2, o3 = children
        assert [episode.operator_id for episode in children] == ["O1", "O2", "O3"]
        assert o1.target_component == O1_COMPONENTS[b % 4]
        assert o1.operation is None
        assert o1.parent_indices == [primary]
        assert o2.operation == O2_OPERATIONS[b % 4]
        assert o2.target_component is None
        assert o2.parent_indices == [primary]
        assert o3.target_component is None
        assert o3.operation is None
        assert o3.parent_indices == [primary, donor_one, donor_two, donor_three]
        primary_id = GENOME_IDS[primary]
        assert o1.component_sources == {
            component: primary_id
            for component in (*O1_COMPONENTS[:3], "workflow_graph", "verification_policy")
        }
        assert o2.component_sources == o1.component_sources
        assert o3.component_sources == {
            component: GENOME_IDS[source_index]
            for component, source_index in O3_COMPOSITIONS[primary].items()
        }


def test_every_child_depends_on_four_parent_evaluations_and_serialized_state():
    api, plan = make_plan()

    for block in plan.blocks:
        for parent_id in block.parent_episode_ids:
            parent = api.episode_by_id(plan, parent_id)
            assert parent.depends_on_episode_ids == []
            assert parent.depends_on_event_ids == []
        for child_id in block.child_episode_ids:
            child = api.episode_by_id(plan, child_id)
            assert child.depends_on_episode_ids == block.parent_episode_ids
            assert child.depends_on_event_ids == [
                *block.parent_evaluation_event_ids,
                block.pre_child_state_event_id,
            ]

        state = api.build_pre_child_state(
            block,
            {
                episode_id: make_state_input(api, plan, episode_id)
                for episode_id in block.parent_episode_ids
            },
        )
        events = api.build_block_events(plan, block.block_index, pre_child_state=state)
        assert len(events) == 11
        api.validate_event_order(events)
        positions = {event.event_id: index for index, event in enumerate(events)}
        for event in events:
            if event.kind == "child_generated":
                assert all(
                    positions[dependency] < positions[event.event_id]
                    for dependency in event.depends_on_event_ids
                )
                assert set(event.depends_on_event_ids) == {
                    *block.parent_evaluation_event_ids,
                    block.pre_child_state_event_id,
                }

    with pytest.raises(ValueError, match="depend|order|before"):
        api.validate_event_order(list(reversed(events)))


def test_initial_genome_hashes_match_files_and_plan_is_reproducible():
    api, plan = make_plan()
    second = api.build_plan_from_file(CONFIG_PATH, repo_root=ROOT)

    assert plan.plan_sha256 == second.plan_sha256
    assert api.validate_plan(plan, repo_root=ROOT) == plan
    assert set(plan.config.initial_genome_hashes) == set(GENOME_IDS)
    for genome_id, expected_hash in plan.config.initial_genome_hashes.items():
        genome = api.validate_genome(
            (ROOT / "configs" / "e1" / "genomes" / f"{genome_id}.json").read_bytes()
        )
        assert api.genome_hash(genome) == expected_hash
        parent = next(
            episode for episode in plan.episodes[:4] if episode.initial_genome_id == genome_id
        )
        assert parent.initial_genome_hash == expected_hash


def test_config_models_are_strict_and_reject_unknown_task_fields():
    api = load_api()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["tasks"][0]["score"] = 1.0

    with pytest.raises(ValueError, match="extra|forbid"):
        api.E1Config.model_validate(payload)


def make_state_input(api, plan, episode_id, *, passed=2, total=2, events=None):
    episode = api.episode_by_id(plan, episode_id)
    return api.ParentEpisodeStateInput(
        block_index=episode.block_index,
        episode_id=episode_id,
        genome_hash=episode.initial_genome_hash or "0" * 64,
        events=events
        if events is not None
        else [
            api.ParentStateEvent(
                step=0,
                dimension="planning",
                outcome="observed",
                source="trajectory",
            )
        ],
        run_status="completed",
        official_success=passed == total,
        official_tests=[
            *[
                api.OfficialTestResult(
                    test_id=f"target-{index}",
                    passed=index < passed,
                    no_op=False,
                )
                for index in range(total)
            ],
            api.OfficialTestResult(test_id="no-op", passed=False, no_op=True),
        ],
        progress_unavailable_reason=None,
    )


def test_task_state_uses_latest_parent_events_and_non_noop_official_progress():
    api, plan = make_plan()
    parent_input = make_state_input(
        api,
        plan,
        plan.blocks[0].parent_episode_ids[0],
        passed=2,
        total=3,
        events=[
            api.ParentStateEvent(
                step=0,
                dimension="planning",
                outcome="observed",
                source="trajectory",
            ),
            api.ParentStateEvent(
                step=1,
                dimension="tool",
                outcome="failed",
                source="appworld_error",
                error_type="ConnectionError",
            ),
            api.ParentStateEvent(
                step=2,
                dimension="planning",
                outcome="failed",
                source="trajectory",
                error_type="PlanningError",
            ),
            api.ParentStateEvent(
                step=3,
                dimension="tool",
                outcome="observed",
                source="trajectory",
            ),
            api.ParentStateEvent(
                step=4,
                dimension="code_execution",
                outcome="failed",
                source="appworld_error",
                error_type="SyntaxError",
            ),
        ],
    )

    state = api.derive_task_state(parent_input)

    assert set(api.TaskState.model_fields) == {
        "planning",
        "tool",
        "state_dependency",
        "code_execution",
        "task_incomplete",
        "task_progress",
    }
    assert state.planning == "failed"
    assert state.tool == "observed"
    assert state.state_dependency == "not_observed"
    assert state.code_execution == "failed"
    assert state.task_incomplete == "incomplete"
    assert state.task_progress.value == pytest.approx(2 / 3)
    assert state.task_progress.null_reason is None


def test_unknown_official_progress_is_null_with_reason_and_not_imputed():
    api, plan = make_plan()
    parent_id = plan.blocks[0].parent_episode_ids[0]
    parent_input = make_state_input(api, plan, parent_id).model_copy(
        update={
            "official_success": None,
            "official_tests": None,
            "progress_unavailable_reason": "official_test_counts_unavailable",
        }
    )

    task_state = api.derive_task_state(parent_input)
    state = api.build_pre_child_state(
        plan.blocks[0],
        {
            **{
                episode_id: make_state_input(api, plan, episode_id)
                for episode_id in plan.blocks[0].parent_episode_ids[1:]
            },
            parent_id: parent_input,
        },
    )

    assert task_state.task_incomplete == "unknown"
    assert task_state.task_progress.value is None
    assert task_state.task_progress.null_reason == "official_test_counts_unavailable"
    assert state.population.mean is None
    assert state.population.best is None
    assert state.population.variance is None
    assert parent_id in state.population_null_reason


def test_population_state_has_frozen_metrics_and_verifiable_serialization_hash():
    api, plan = make_plan()
    block = plan.blocks[0]
    inputs = {
        episode_id: make_state_input(
            api,
            plan,
            episode_id,
            passed=index,
            total=2,
        )
        for index, episode_id in enumerate(block.parent_episode_ids)
    }
    inputs[block.parent_episode_ids[0]] = make_state_input(
        api, plan, block.parent_episode_ids[0], passed=0, total=4
    )
    inputs[block.parent_episode_ids[1]] = make_state_input(
        api, plan, block.parent_episode_ids[1], passed=1, total=2
    )
    inputs[block.parent_episode_ids[2]] = make_state_input(
        api, plan, block.parent_episode_ids[2], passed=1, total=1
    )
    inputs[block.parent_episode_ids[3]] = make_state_input(
        api, plan, block.parent_episode_ids[3], passed=1, total=1
    )

    state = api.build_pre_child_state(block, inputs)

    assert set(api.PopulationState.model_fields) == {
        "mean",
        "best",
        "variance",
        "genome_diversity",
        "stagnation",
    }
    assert state.population.mean == pytest.approx(0.625)
    assert state.population.best == 1.0
    assert state.population.variance == pytest.approx(0.171875)
    assert state.population.genome_diversity == 1.0
    assert state.population.stagnation == "insufficient_history"
    assert state.parent_episode_ids == block.parent_episode_ids
    assert set(state.task_states) == set(block.parent_episode_ids)
    assert api.validate_pre_child_state(state) == state
    assert api.pre_child_state_hash(state) == state.state_sha256
    assert api.canonical_pre_child_state_bytes(state) == api.canonical_pre_child_state_bytes(
        api.PreChildState.model_validate_json(state.model_dump_json())
    )

    events = api.build_block_events(plan, 0, pre_child_state=state)
    state_event = next(event for event in events if event.kind == "pre_child_state_serialized")
    assert state_event.payload_sha256 == state.state_sha256
    api.validate_event_order(events)

    with pytest.raises(ValueError, match="hash"):
        api.validate_pre_child_state(state.model_copy(update={"state_sha256": "0" * 64}))


def test_current_state_hash_ignores_every_child_and_future_block_perturbation():
    api, plan = make_plan()
    block = plan.blocks[0]
    inputs = {
        episode.episode_id: make_state_input(api, plan, episode.episode_id)
        for episode in plan.episodes
    }
    baseline = api.build_pre_child_state(block, inputs)

    for child_id in block.child_episode_ids:
        perturbed = inputs.copy()
        perturbed[child_id] = make_state_input(
            api,
            plan,
            child_id,
            passed=0,
            total=3,
            events=[
                api.ParentStateEvent(
                    step=99,
                    dimension="tool",
                    outcome="failed",
                    source="appworld_error",
                    error_type="ChildOnlyError",
                )
            ],
        )
        assert api.build_pre_child_state(block, perturbed).state_sha256 == baseline.state_sha256

    for future_block in plan.blocks[1:]:
        perturbed = inputs.copy()
        for episode_id in future_block.episode_ids:
            perturbed[episode_id] = make_state_input(
                api, plan, episode_id, passed=0, total=5, events=[]
            )
        assert api.build_pre_child_state(block, perturbed).state_sha256 == baseline.state_sha256

    current_parent = block.parent_episode_ids[0]
    parent_perturbed = inputs.copy()
    parent_perturbed[current_parent] = make_state_input(
        api, plan, current_parent, passed=0, total=2
    )
    assert api.build_pre_child_state(block, parent_perturbed).state_sha256 != baseline.state_sha256


def test_stagnation_is_disabled_and_state_input_schema_is_strict():
    api, plan = make_plan()

    for block in plan.blocks[:2]:
        state = api.build_pre_child_state(
            block,
            {
                episode_id: make_state_input(api, plan, episode_id)
                for episode_id in block.parent_episode_ids
            },
        )
        assert state.population.stagnation == "insufficient_history"

    with pytest.raises(ValueError, match="stagnation"):
        api.PopulationState(
            mean=0.0,
            best=0.0,
            variance=0.0,
            genome_diversity=1.0,
            stagnation="stagnant",
        )
    with pytest.raises(ValueError, match="error_type"):
        api.ParentStateEvent(
            step=0,
            dimension="tool",
            outcome="failed",
            source="appworld_error",
        )
    with pytest.raises(ValueError, match="increasing"):
        api.ParentEpisodeStateInput(
            block_index=0,
            episode_id="parent",
            genome_hash="0" * 64,
            events=[
                api.ParentStateEvent(
                    step=1,
                    dimension="tool",
                    outcome="observed",
                    source="trajectory",
                ),
                api.ParentStateEvent(
                    step=1,
                    dimension="planning",
                    outcome="observed",
                    source="trajectory",
                ),
            ],
            run_status="completed",
            official_success=True,
            official_tests=[],
            progress_unavailable_reason=None,
        )


def test_child_execution_events_require_a_serialized_state_payload():
    api, plan = make_plan()

    with pytest.raises(TypeError):
        api.build_block_events(plan, 0)

    block = plan.blocks[0]
    state = api.build_pre_child_state(
        block,
        {
            episode_id: make_state_input(api, plan, episode_id)
            for episode_id in block.parent_episode_ids
        },
    )
    events = api.build_block_events(plan, 0, pre_child_state=state)
    without_payload = [
        event.model_copy(update={"payload_sha256": None})
        if event.kind == "pre_child_state_serialized"
        else event
        for event in events
    ]

    with pytest.raises(ValueError, match="payload|serializ"):
        api.validate_event_order(without_payload)


def valid_success_events(api):
    plan = api.build_plan_from_file(CONFIG_PATH, repo_root=ROOT)
    block = plan.blocks[0]
    state = api.build_pre_child_state(
        block,
        {
            episode_id: make_state_input(api, plan, episode_id)
            for episode_id in block.parent_episode_ids
        },
    )
    return api.build_block_events(plan, 0, pre_child_state=state)


@pytest.mark.parametrize(
    ("event_index", "updates"),
    [
        (0, {"episode_id": None}),
        (0, {"episode_id": "e1-b01-p0"}),
        (0, {"depends_on_event_ids": ["e1-b00-p1:evaluated"]}),
        (4, {"block_index": 1}),
        (4, {"episode_id": "e1-b00-p0"}),
        (4, {"depends_on_event_ids": []}),
        (
            4,
            {
                "depends_on_event_ids": [
                    "e1-b00-p0:evaluated",
                    "e1-b00-p1:evaluated",
                    "e1-b00-p2:evaluated",
                    "e1-b00-p0:evaluated",
                ]
            },
        ),
        (
            4,
            {
                "depends_on_event_ids": [
                    "e1-b00-p0:evaluated",
                    "e1-b00-p1:evaluated",
                    "e1-b00-p2:evaluated",
                    "e1-b01-p0:evaluated",
                ]
            },
        ),
        (5, {"event_id": "e1-b00-o1:other"}),
        (5, {"block_index": 1}),
        (5, {"episode_id": "e1-b00-o2"}),
        (5, {"depends_on_event_ids": []}),
        (
            5,
            {
                "depends_on_event_ids": [
                    "e1-b00-p0:evaluated",
                    "e1-b00-p1:evaluated",
                    "e1-b00-p2:evaluated",
                    "e1-b00-p3:evaluated",
                ]
            },
        ),
        (
            5,
            {
                "depends_on_event_ids": [
                    "e1-b00-p0:evaluated",
                    "e1-b00-p1:evaluated",
                    "e1-b00-p2:evaluated",
                    "e1-b00-p3:evaluated",
                    "e1-b00-p0:evaluated",
                ]
            },
        ),
        (6, {"event_id": "e1-b00-o1:finalized"}),
        (6, {"block_index": 1}),
        (6, {"episode_id": "e1-b00-o2"}),
        (6, {"depends_on_event_ids": []}),
        (6, {"depends_on_event_ids": ["e1-b00:pre-child-state-serialized"]}),
        (
            6,
            {
                "depends_on_event_ids": [
                    "e1-b00-o1:generated",
                    "e1-b00-o1:generated",
                ]
            },
        ),
    ],
)
def test_event_order_rejects_invalid_episode_block_and_dependencies(
    event_index, updates
):
    api = load_api()
    events = valid_success_events(api)
    events[event_index] = events[event_index].model_copy(update=updates)

    with pytest.raises(ValueError):
        api.validate_event_order(events)


def test_child_finalized_is_valid_and_terminal_events_are_unique():
    api = load_api()
    events = valid_success_events(api)
    generated = events[5]
    evaluated = events[6]
    finalized = api.ProtocolEvent(
        event_id="e1-b00-o1:finalized",
        block_index=0,
        kind="child_finalized",
        episode_id="e1-b00-o1",
        depends_on_event_ids=[generated.event_id],
    )
    events[6] = finalized

    api.validate_event_order(events)

    with pytest.raises(ValueError, match="terminal|multiple|unique"):
        api.validate_event_order([*events[:6], finalized, evaluated, *events[7:]])
