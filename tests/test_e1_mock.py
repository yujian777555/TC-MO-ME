from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import tc_mo_me.e1_mock as e1_mock
from tc_mo_me.e1_mock import generate_child, load_mock_config, run_e1_mock
from tc_mo_me.e1_protocol import (
    GENOME_IDS,
    OfficialTestResult,
    ParentEpisodeStateInput,
    build_plan_from_file,
    build_pre_child_state,
    validate_pre_child_state,
)
from tc_mo_me.genome import CONTENT_COMPONENTS, genome_hash, validate_genome
from tc_mo_me.ledger import RunLedger
from tc_mo_me.model import MockModelAdapter

ROOT = Path(__file__).parents[1]
MOCK_CONFIG = ROOT / "configs" / "e1" / "mock_dry_run.yaml"
PILOT_CONFIG = ROOT / "configs" / "e1" / "pilot.yaml"
V1_O3_NO_CHANGE_BLOCKS = frozenset({2, 3, 6, 7, 10, 11, 14, 15})


def test_mock_config_freezes_full_pilot_and_explicit_zero_cost() -> None:
    config = load_mock_config(MOCK_CONFIG)
    plan = build_plan_from_file(PILOT_CONFIG, repo_root=ROOT)

    assert config.run_id == "e1-mock-dry-run-v2"
    assert config.experiment_id == "e1-mock-dry-run"
    assert plan.config.protocol_version == "1.1"
    assert config.source_protocol.config_path == "configs/e1/pilot.yaml"
    assert config.source_protocol.config_sha256 == plan.config_sha256
    assert config.source_protocol.plan_sha256 == plan.plan_sha256
    assert config.model.adapter == "mock"
    assert config.model.model_id == "e1-deterministic-mock"
    assert config.model.model_version == "e1-mock-v1"
    assert config.model.input_price_per_million == 0
    assert config.model.output_price_per_million == 0
    assert config.limits.max_blocks == 16
    assert config.limits.max_episodes == 112
    assert len(plan.blocks) == config.limits.max_blocks
    assert len(plan.episodes) == config.limits.max_episodes
    assert config.provenance.require_clean is True


def initial_genomes():
    return {
        genome_id: validate_genome(
            (ROOT / "configs" / "e1" / "genomes" / f"{genome_id}.json").read_bytes()
        )
        for genome_id in ("g0_baseline", "g1_planner", "g2_verifier", "g3_workflow")
    }


def child_for(block_index: int, operator_id: str):
    plan = build_plan_from_file(PILOT_CONFIG, repo_root=ROOT)
    block = plan.blocks[block_index]
    episode_id = next(
        episode_id
        for episode_id in block.child_episode_ids
        if episode_id.endswith(operator_id.lower())
    )
    return next(episode for episode in plan.episodes if episode.episode_id == episode_id)


@pytest.mark.parametrize(
    ("block_index", "component"),
    [
        (0, "system_prompt"),
        (1, "planning_policy"),
        (2, "code_generation_policy"),
        (3, "verification_policy"),
    ],
)
def test_o1_mock_changes_only_the_plan_target(block_index: int, component: str) -> None:
    config = load_mock_config(MOCK_CONFIG)
    episode = child_for(block_index, "O1")
    parents = initial_genomes()
    primary_id = f"g{block_index}_" + ("baseline", "planner", "verifier", "workflow")[block_index]
    before = parents[primary_id]

    attempt = generate_child(episode, parents, model_config=config.model)

    assert attempt.result.success is True
    assert attempt.result.modified_paths == [component]
    child = attempt.result.child
    assert child is not None
    assert getattr(child, component) != getattr(before, component)
    for other in CONTENT_COMPONENTS:
        if other != component:
            assert getattr(child, other) == getattr(before, other)
    assert child.identity["genome_id"] == f"{episode.episode_id}-child"
    assert attempt.result.usage.model_version == config.model.model_version
    assert attempt.result.usage.phase == "generation"
    assert attempt.result.usage.api_cost == 0


@pytest.mark.parametrize("block_index", range(4))
@pytest.mark.parametrize("parent_index", range(4))
def test_o2_mock_applies_each_operation_to_every_initial_genome(
    block_index: int, parent_index: int
) -> None:
    config = load_mock_config(MOCK_CONFIG)
    base_episode = child_for(block_index, "O2")
    genome_ids = ("g0_baseline", "g1_planner", "g2_verifier", "g3_workflow")
    primary_id = genome_ids[parent_index]
    episode = base_episode.model_copy(
        update={
            "primary_parent_index": parent_index,
            "parent_indices": [parent_index],
            "component_sources": {component: primary_id for component in CONTENT_COMPONENTS},
        }
    )
    parents = initial_genomes()
    before = parents[primary_id]

    attempt = generate_child(episode, parents, model_config=config.model)

    assert attempt.result.success is True, attempt.result.error
    assert attempt.result.modified_paths == ["workflow_graph"]
    child = attempt.result.child
    assert child is not None
    for component in CONTENT_COMPONENTS:
        if component != "workflow_graph":
            assert getattr(child, component) == getattr(before, component)
    if episode.operation == "insert":
        assert len(child.workflow_graph.nodes) == len(before.workflow_graph.nodes) + 1
    elif episode.operation == "delete":
        assert len(child.workflow_graph.nodes) == len(before.workflow_graph.nodes) - 1
    elif episode.operation == "replace":
        assert child.workflow_graph.nodes != before.workflow_graph.nodes
        assert {node.id for node in child.workflow_graph.nodes} == {
            node.id for node in before.workflow_graph.nodes
        }
    else:
        assert [node.id for node in child.workflow_graph.nodes] == [
            node.id for node in reversed(before.workflow_graph.nodes)
        ]


def test_o3_mock_recombines_plan_sources_deterministically() -> None:
    config = load_mock_config(MOCK_CONFIG)
    episode = child_for(0, "O3")
    parents = initial_genomes()
    before = {genome_id: genome.model_dump(mode="json") for genome_id, genome in parents.items()}

    first = generate_child(episode, parents, model_config=config.model)
    second = generate_child(episode, parents, model_config=config.model)

    assert first == second
    assert first.result.success is True
    child = first.result.child
    assert child is not None
    for component, source_id in episode.component_sources.items():
        assert getattr(child, component) == getattr(parents[source_id], component)
    assert first.result.lineage is not None
    assert first.result.lineage.component_sources == episode.component_sources
    assert first.result.usage.calls == 0
    assert first.result.usage.api_cost == 0
    assert {genome_id: genome.model_dump(mode="json") for genome_id, genome in parents.items()} == before


def o3_block_label(block_index: int) -> str:
    suffix = "-v1-structural-no-change" if block_index in V1_O3_NO_CHANGE_BLOCKS else ""
    return f"b{block_index:02d}{suffix}"


@pytest.mark.parametrize("block_index", range(16), ids=o3_block_label)
def test_o3_v2_child_hash_is_novel_for_all_four_parents(block_index: int) -> None:
    config = load_mock_config(MOCK_CONFIG)
    episode = child_for(block_index, "O3")
    parents = initial_genomes()
    expected_parent_ids = tuple(GENOME_IDS[index] for index in episode.parent_indices)

    attempt = generate_child(episode, parents, model_config=config.model)

    assert attempt.result.success is True, attempt.result.error
    assert episode.parent_indices == [
        block_index % 4,
        (block_index + 1) % 4,
        (block_index + 2) % 4,
        (block_index + 3) % 4,
    ]
    assert set(episode.parent_indices) == {0, 1, 2, 3}
    assert attempt.parent_ids == expected_parent_ids
    child = attempt.result.child
    assert child is not None
    parent_hashes = {genome_hash(parents[parent_id]) for parent_id in expected_parent_ids}
    assert genome_hash(child) not in parent_hashes
    assert attempt.result.lineage is not None
    assert attempt.result.lineage.parent_ids == list(expected_parent_ids)
    assert attempt.result.lineage.primary_parent_id == GENOME_IDS[block_index % 4]


def small_config(tmp_path: Path, *, blocks: int = 1) -> Path:
    payload = yaml.safe_load(MOCK_CONFIG.read_text(encoding="utf-8"))
    payload["limits"] = {"max_blocks": blocks, "max_episodes": blocks * 7}
    path = tmp_path / f"mock-{blocks}-blocks.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_single_block_commits_parents_then_state_then_each_child(tmp_path: Path) -> None:
    repo_index = ROOT / "results" / "index.jsonl"
    index_before = repo_index.read_bytes() if repo_index.exists() else None
    results_root = tmp_path / "external-results"

    result = run_e1_mock(
        small_config(tmp_path), ROOT, results_root, allow_dirty=True
    )

    plan = build_plan_from_file(PILOT_CONFIG, repo_root=ROOT)
    block = plan.blocks[0]
    assert [episode.episode_id for episode in result.episodes] == block.episode_ids
    assert [episode.operator_id for episode in result.episodes] == [
        "control",
        "control",
        "control",
        "control",
        "O1",
        "O2",
        "O3",
    ]
    assert all(episode.run_status == "succeeded" for episode in result.episodes)
    assert all(episode.task_score == 1.0 for episode in result.episodes)
    assert all(episode.model_id == "e1-deterministic-mock" for episode in result.episodes)
    assert all(episode.api_cost == 0 for episode in result.episodes)
    assert all(
        usage.model_version == "e1-mock-v1"
        for episode in result.episodes
        for usage in episode.usage_records
        if usage.model_id == "e1-deterministic-mock"
    )
    assert [event.kind for event in result.events] == [
        "parent_evaluated",
        "parent_evaluated",
        "parent_evaluated",
        "parent_evaluated",
        "pre_child_state_serialized",
        "child_generated",
        "child_evaluated",
        "child_generated",
        "child_evaluated",
        "child_generated",
        "child_evaluated",
    ]
    state = result.pre_child_states[0]
    assert validate_pre_child_state(state) == state
    assert state.parent_episode_ids == block.parent_episode_ids
    assert set(result.parent_inputs) == set(block.parent_episode_ids)
    for parent_input in result.parent_inputs.values():
        assert parent_input.official_tests is not None
        assert all(
            test.test_id.startswith("synthetic-mock:")
            for test in parent_input.official_tests
        )
    state_path = result.ledger.run_dir / "states" / "block-00-pre-child.json"
    state_envelope = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_envelope["evaluation_provenance"] == "synthetic_mock_only"
    assert "appworld" not in state_path.read_text(encoding="utf-8").lower()
    assert state_envelope["state"]["state_sha256"] == state.state_sha256
    event_lines = (result.ledger.run_dir / "protocol_events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(event_lines) == 11
    assert RunLedger(results_root, config_run_id(), allow_existing=True).read_episodes() == result.episodes
    index_entries = [
        json.loads(line)
        for line in (results_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(index_entries) == 7
    assert all(
        not {"code", "output", "request", "response", "raw_steps", "trajectory"} & set(entry)
        for entry in index_entries
    )
    assert (repo_index.read_bytes() if repo_index.exists() else None) == index_before


def config_run_id() -> str:
    return "e1-mock-dry-run-v2"


def test_generation_failure_is_committed_and_does_not_skip_later_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_generate_child = e1_mock.generate_child

    def fail_o1(episode, parents, *, model_config, model=None):
        if episode.operator_id == "O1":
            model = MockModelAdapter(
                model_version=model_config.model_version,
                timeout_seconds=model_config.timeout_seconds,
                max_retries=0,
                price_id=model_config.price_id,
                input_price_per_million=model_config.input_price_per_million,
                output_price_per_million=model_config.output_price_per_million,
            )
        return original_generate_child(
            episode,
            parents,
            model_config=model_config,
            model=model,
        )

    monkeypatch.setattr(e1_mock, "generate_child", fail_o1)
    result = run_e1_mock(
        small_config(tmp_path), ROOT, tmp_path / "failure-results", allow_dirty=True
    )

    assert len(result.episodes) == 7
    failed = result.episodes[4]
    assert failed.episode_id == "e1-b00-o1"
    assert failed.run_status == "generation_failed"
    assert failed.task_score is None
    assert failed.api_calls == 1
    assert failed.api_cost == 0
    assert [episode.run_status for episode in result.episodes[5:]] == [
        "succeeded",
        "succeeded",
    ]
    assert result.events[5].event_id == "e1-b00-o1:generated"
    assert result.events[6].event_id == "e1-b00-o1:evaluated"
    assert len(result.ledger.read_episodes()) == 7
    private_steps = (
        result.ledger.run_dir
        / "episodes"
        / failed.episode_id
        / "private_steps.jsonl"
    ).read_text(encoding="utf-8")
    generation = json.loads(private_steps.splitlines()[0])["generation"]
    assert generation["success"] is False
    assert generation["error"]["code"] == "invalid_json"


def perturbed_input(episode) -> ParentEpisodeStateInput:
    return ParentEpisodeStateInput(
        block_index=episode.block_index,
        episode_id=episode.episode_id,
        genome_hash="0" * 64,
        events=[],
        run_status="completed",
        official_success=False,
        official_tests=[
            OfficialTestResult(
                test_id="synthetic-mock:perturbed",
                passed=False,
                no_op=False,
            )
        ],
        progress_unavailable_reason=None,
    )


def test_current_block_state_hash_ignores_child_and_future_plan_perturbations(
    tmp_path: Path,
) -> None:
    result = run_e1_mock(
        small_config(tmp_path), ROOT, tmp_path / "state-results", allow_dirty=True
    )
    plan = build_plan_from_file(PILOT_CONFIG, repo_root=ROOT)
    block = plan.blocks[0]
    baseline = result.pre_child_states[0]
    inputs = dict(result.parent_inputs)
    for episode in plan.episodes:
        if episode.episode_id not in inputs:
            inputs[episode.episode_id] = perturbed_input(episode)

    rebuilt = build_pre_child_state(block, inputs)

    assert rebuilt.state_sha256 == baseline.state_sha256
    assert rebuilt.source_inputs_sha256 == baseline.source_inputs_sha256
    changed_parent_id = block.parent_episode_ids[0]
    inputs[changed_parent_id] = perturbed_input(plan.episodes[0])
    assert build_pre_child_state(block, inputs).state_sha256 != baseline.state_sha256


def test_source_plan_hash_mismatch_fails_before_creating_a_run(tmp_path: Path) -> None:
    config_path = small_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["source_protocol"]["plan_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    results_root = tmp_path / "rejected-results"

    with pytest.raises(ValueError, match="plan hash"):
        run_e1_mock(config_path, ROOT, results_root, allow_dirty=True)

    assert not (results_root / "raw" / config_run_id()).exists()
