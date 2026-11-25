from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tc_mo_me.agent import AgentRunner, GenerationAttempt, ResultIndexError, TaskSpec
from tc_mo_me.genome import CONTENT_COMPONENTS, Genome
from tc_mo_me.ledger import RunLedger
from tc_mo_me.model import (
    CostBasis,
    MockModelAdapter,
    ModelAdapter,
    ModelCallError,
    UsageRecord,
)
from tc_mo_me.operators import MutationSpec, OperatorContext
from tc_mo_me.operators.o1_local import O1LocalOperator
from tc_mo_me.operators.o2_structural import O2StructuralOperator
from tc_mo_me.operators.o3_crossover import O3CrossoverOperator
from tc_mo_me.provenance import RunProvenance

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
CONFIG_HASH = "a" * 64


class QueuedModel(MockModelAdapter):
    def __init__(self, *responses: str) -> None:
        super().__init__(
            price_id="mock-price-v1",
            input_price_per_million=Decimal("1"),
            output_price_per_million=Decimal("2"),
        )
        self.responses = list(responses)
        self.requests = []

    def _generate_once(self, request, timeout_seconds):
        self.requests.append(request)
        self._script[request.prompt] = self.responses.pop(0)
        return super()._generate_once(request, timeout_seconds)


class FailingModel(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(max_retries=0, deterministic_timing=True, model_seed_supported=True)

    def _generate_once(self, request, timeout_seconds):
        raise RuntimeError("deterministic provider failure")


class FakeEnvironment:
    def __init__(self, *, success: bool = True, execute_error: bool = False) -> None:
        self.success = success
        self.execute_error = execute_error
        self.reset_calls: list[str] = []
        self.executed: list[str] = []
        self.closed = False

    def reset(self, task_id: str) -> dict[str, str]:
        self.reset_calls.append(task_id)
        return {"task_id": task_id, "instruction": "Complete the deterministic mock task."}

    def execute(self, code: str) -> str:
        self.executed.append(code)
        if self.execute_error:
            raise RuntimeError("deterministic execution failure")
        return "mock execution output"

    def evaluate(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "report": "official mock evaluation",
            "details": {"tests": 1, "passed": int(self.success)},
        }

    def trajectory(self) -> list[dict[str, str]]:
        return []

    def close(self) -> None:
        self.closed = True


def make_genome(*, loop: bool = False, genome_id: str = "control") -> Genome:
    edges = [
        {"source": "plan", "target": "generate"},
        {"source": "generate", "target": "execute"},
        {"source": "execute", "target": "verify"},
    ]
    if loop:
        edges.append({"source": "verify", "target": "generate"})
    return Genome.model_validate_json(
        json.dumps(
            {
                "schema_version": "0.1",
                "system_prompt": "Solve the task and follow the node response format.",
                "planning_policy": {"strategy": "decompose", "max_steps": 8},
                "code_generation_policy": {"language": "python", "tests_required": True},
                "verification_policy": {"require_checks": True},
                "workflow_graph": {
                    "entry_id": "plan",
                    "exit_ids": ["verify"],
                    "max_nodes": 8,
                    "nodes": [
                        {"id": "plan", "type": "planner", "config": {}},
                        {"id": "generate", "type": "generator", "config": {}},
                        {"id": "execute", "type": "executor", "config": {}},
                        {"id": "verify", "type": "verifier", "config": {}},
                    ],
                    "edges": edges,
                },
                "identity": {"genome_id": genome_id},
                "lineage": {},
            }
        )
    )


def provenance() -> RunProvenance:
    return RunProvenance(
        git_sha=GIT_SHA,
        config_hash=CONFIG_HASH,
        git_dirty=False,
        captured_at_utc=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )


def make_runner(
    tmp_path: Path,
    model: ModelAdapter,
    *,
    max_steps: int = 8,
) -> tuple[AgentRunner, RunLedger]:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=provenance())
    runner = AgentRunner(
        model=model,
        model_id="mock-model",
        ledger=ledger,
        provenance=provenance(),
        max_steps=max_steps,
        process_isolation=False,
    )
    return runner, ledger


def run_episode(runner: AgentRunner, genome: Genome, environment: FakeEnvironment):
    return runner.run_episode(
        genome=genome,
        environment=environment,
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="episode-1",
    )


def successful_responses() -> tuple[str, str, str]:
    return (
        json.dumps({"plan": "inspect then solve"}),
        json.dumps({"code": "print('deterministic mock')"}),
        json.dumps({"action": "finish", "reason": "ready for official evaluation"}),
    )


def artifact(ledger: RunLedger, episode_id: str, name: str) -> Path:
    return ledger.run_dir / "episodes" / episode_id / name


def test_linear_episode_executes_graph_and_persists_raw_trajectory_and_usage(
    tmp_path: Path,
) -> None:
    model = QueuedModel(*successful_responses())
    environment = FakeEnvironment()
    runner, ledger = make_runner(tmp_path, model)

    episode = run_episode(runner, make_genome(), environment)

    assert episode.run_status == "succeeded"
    assert episode.task_score == 1.0
    assert episode.task_id == "task-1"
    assert episode.run_id == "run-1"
    assert episode.experiment_id == "experiment-1"
    assert episode.environment_seed == 7
    assert episode.git_sha == GIT_SHA
    assert episode.config_hash == CONFIG_HASH
    assert episode.operator_id == "control"
    assert episode.parent_id is None
    assert episode.parent_ids == []
    assert episode.child_id is None
    assert environment.executed == ["print('deterministic mock')"]
    assert environment.closed is True
    assert len(model.requests) == 3
    assert [json.loads(request.prompt)["node_type"] for request in model.requests] == [
        "planner",
        "generator",
        "verifier",
    ]
    for request in model.requests:
        prompt = json.loads(request.prompt)
        assert request.system_prompt == make_genome().system_prompt
        assert prompt["task_instruction"] == "Complete the deterministic mock task."
        assert set(prompt["policies"]) == {"planning", "code_generation", "verification"}
        assert request.seed == 7
    assert episode.prompt_hash == model.requests[0].prompt_hash
    assert episode.api_calls == 3
    assert episode.retry_count == 0
    assert episode.api_cost is not None
    assert episode.generation_cost is not None
    assert episode.evaluation_cost is not None
    assert episode.total_cost == episode.generation_cost + episode.evaluation_cost
    assert episode.cost_complete is True
    assert len(episode.usage_records) == 1
    assert episode.usage_records[0].calls == 3
    assert episode.usage_records[0].price_id == "mock-price-v1"
    assert episode.usage_records[0].model_version == "mock-v1"
    assert {usage.phase for usage in episode.usage_records} == {"evaluation"}
    assert episode.generation_cost == 0
    assert episode.evaluation_cost == episode.api_cost

    private_path = artifact(ledger, episode.episode_id, "private_steps.jsonl")
    trajectory_path = artifact(ledger, episode.episode_id, "trajectory.jsonl")
    private_steps = [json.loads(line) for line in private_path.read_text().splitlines()]
    trajectory = trajectory_path.read_text(encoding="utf-8")
    assert len(private_steps) == 4
    assert private_steps[0]["request"]["prompt"] == model.requests[0].prompt
    assert private_steps[0]["response"]["content"] == successful_responses()[0]
    assert private_steps[1]["code"] == "print('deterministic mock')"
    assert private_steps[2]["output"] == "mock execution output"
    assert private_steps[3]["evaluation"]["success"] is True
    assert "model_request" in trajectory
    assert "model_response" in trajectory
    assert "official_evaluation" in trajectory
    assert set(episode.artifact_hashes) == {"private_steps.jsonl", "trajectory.jsonl"}
    assert ledger.read_episodes() == [episode]
    assert json.loads(ledger.manifest_path.read_text())["episode_count"] == 1


def test_official_zero_score_is_not_unknown(tmp_path: Path) -> None:
    environment = FakeEnvironment(success=False)
    runner, ledger = make_runner(tmp_path, QueuedModel(*successful_responses()))

    episode = run_episode(runner, make_genome(), environment)

    assert episode.run_status == "task_failed"
    assert episode.task_score == 0.0
    assert episode.task_score is not None
    assert environment.closed is True
    assert ledger.read_episodes()[0].task_score == 0.0


def test_model_failure_preserves_attempted_usage_and_writes_episode(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    runner, ledger = make_runner(tmp_path, FailingModel())

    episode = run_episode(runner, make_genome(), environment)

    assert episode.run_status == "model_failed"
    assert episode.task_score is None
    assert episode.api_calls == 1
    assert episode.api_cost is None
    assert episode.cost_complete is False
    assert episode.total_cost is None
    assert len(episode.usage_records) == 1
    assert episode.usage_records[0].calls == 1
    assert episode.usage_records[0].cost_basis is CostBasis.UNKNOWN
    assert environment.closed is True
    assert ledger.read_episodes() == [episode]
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    assert "deterministic provider failure" in private


def test_generator_requires_exact_strict_json_code_object(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    runner, ledger = make_runner(
        tmp_path,
        QueuedModel(
            json.dumps({"plan": "solve"}),
            "```json\n{\"code\": \"print('bad')\"}\n```",
        ),
    )

    episode = run_episode(runner, make_genome(), environment)

    assert episode.run_status == "invalid_model_output"
    assert episode.task_score is None
    assert episode.api_calls == 2
    assert episode.api_cost is not None
    assert environment.executed == []
    assert environment.closed is True
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    assert "```json" in private
    assert "JSONDecodeError" in artifact(
        ledger, episode.episode_id, "trajectory.jsonl"
    ).read_text()
    assert ledger.read_episodes() == [episode]


def test_execution_exception_is_recorded_and_environment_is_closed(tmp_path: Path) -> None:
    environment = FakeEnvironment(execute_error=True)
    runner, ledger = make_runner(
        tmp_path,
        QueuedModel(
            json.dumps({"plan": "solve"}),
            json.dumps({"code": "raise RuntimeError('from generated code')"}),
        ),
    )

    episode = run_episode(runner, make_genome(), environment)

    assert episode.run_status == "execution_failed"
    assert episode.task_score is None
    assert episode.api_calls == 2
    assert episode.execution_time >= 0
    assert environment.closed is True
    trajectory = artifact(ledger, episode.episode_id, "trajectory.jsonl").read_text()
    assert "deterministic execution failure" in trajectory
    assert ledger.read_episodes() == [episode]


def test_verifier_continue_loop_stops_at_max_steps(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    runner, ledger = make_runner(
        tmp_path,
        QueuedModel(
            json.dumps({"plan": "solve"}),
            json.dumps({"code": "first"}),
            json.dumps({"action": "continue", "reason": "needs another attempt"}),
            json.dumps({"code": "second"}),
        ),
        max_steps=5,
    )

    episode = run_episode(runner, make_genome(loop=True), environment)

    assert episode.run_status == "max_steps_exceeded"
    assert episode.task_score is None
    assert episode.api_calls == 4
    assert environment.executed == ["first"]
    assert environment.closed is True
    trajectory = artifact(ledger, episode.episode_id, "trajectory.jsonl").read_text()
    assert "max_steps_exceeded" in trajectory
    assert ledger.read_episodes() == [episode]


@pytest.mark.parametrize(
    "response",
    [
        json.dumps({"code": "ok", "extra": True}),
        json.dumps({"code": 1}),
        '{"code":"first","code":"second"}',
    ],
)
def test_generator_rejects_nonconforming_structures(tmp_path: Path, response: str) -> None:
    runner, _ = make_runner(
        tmp_path,
        QueuedModel(json.dumps({"plan": "solve"}), response),
    )

    episode = run_episode(runner, make_genome(), FakeEnvironment())

    assert episode.run_status == "invalid_model_output"
    assert episode.task_score is None


class BlockingEnvironment(FakeEnvironment):
    def execute(self, code: str) -> str:
        import time

        time.sleep(30)
        return "unreachable"


def test_process_isolated_episode_returns_and_commits(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "results", "process-run", provenance=provenance())
    runner = AgentRunner(
        model=QueuedModel(*successful_responses()),
        model_id="mock-model",
        ledger=ledger,
        provenance=provenance(),
        process_isolation=True,
        timeout_seconds=10,
    )

    episode = runner.run_episode(
        genome=make_genome(),
        environment=FakeEnvironment(),
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="process-success",
    )

    assert episode.run_status == "succeeded"
    assert episode.task_score == 1.0
    assert ledger.read_episodes() == [episode]


def test_blocked_execution_is_terminated_and_timeout_is_committed(tmp_path: Path) -> None:
    import time

    ledger = RunLedger(tmp_path / "results", "timeout-run", provenance=provenance())
    runner = AgentRunner(
        model=QueuedModel(
            json.dumps({"plan": "solve"}),
            json.dumps({"code": "while True: pass"}),
        ),
        model_id="mock-model",
        ledger=ledger,
        provenance=provenance(),
        process_isolation=True,
        timeout_seconds=0.5,
    )
    started = time.monotonic()

    episode = runner.run_episode(
        genome=make_genome(),
        environment=BlockingEnvironment(),
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="process-timeout",
    )

    assert time.monotonic() - started < 8
    assert episode.run_status == "timeout"
    assert episode.task_score is None
    assert episode.api_calls == 2
    assert episode.api_cost is not None
    assert episode.cost_complete is True
    trajectory = artifact(ledger, episode.episode_id, "trajectory.jsonl").read_text()
    assert "while True: pass" in trajectory
    assert "process_terminated" in trajectory
    assert ledger.read_episodes() == [episode]


def make_generation_results():
    parent = make_genome()
    o1 = O1LocalOperator().generate(
        parent,
        MutationSpec(component="system_prompt"),
        OperatorContext(
            model=QueuedModel(
                json.dumps({"component": "system_prompt", "value": "Child O1 prompt"})
            ),
            model_id="mock-model",
            child_id="child-o1",
        ),
    )
    planner_node = parent.workflow_graph.nodes[0].model_dump(mode="json")
    planner_node["config"] = {"mode": "child-o2"}
    o2 = O2StructuralOperator().generate(
        parent,
        MutationSpec(operation="replace"),
        OperatorContext(
            model=QueuedModel(
                json.dumps(
                    {
                        "operation": "replace",
                        "node_id": "plan",
                        "node": planner_node,
                    }
                )
            ),
            model_id="mock-model",
            child_id="child-o2",
        ),
    )
    parent_b = make_genome(genome_id="parent-b").model_copy(
        update={"system_prompt": "Prompt from parent B"}
    )
    parent_c = make_genome(genome_id="parent-c").model_copy(
        update={"code_generation_policy": {"language": "python", "style": "parent-c"}}
    )
    sources = {component: "control" for component in CONTENT_COMPONENTS}
    sources["system_prompt"] = "parent-b"
    sources["code_generation_policy"] = "parent-c"
    o3 = O3CrossoverOperator().generate(
        [parent, parent_b, parent_c],
        MutationSpec(component_sources=sources),
        OperatorContext(child_id="child-o3"),
    )
    assert o1.success and o2.success and o3.success
    return [GenerationAttempt(result=o1), GenerationAttempt(result=o2), GenerationAttempt(result=o3)]


def test_run_experiment_executes_control_and_o1_o2_o3_children_without_leakage(
    tmp_path: Path,
) -> None:
    task_model = QueuedModel(*(successful_responses() * 4))
    runner, ledger = make_runner(tmp_path, task_model)
    environments: list[FakeEnvironment] = []

    def environment_factory(task_id: str, seed: int) -> FakeEnvironment:
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    episodes = runner.run_experiment(
        control=make_genome(),
        tasks=[TaskSpec(task_id="task-1", task_family="mock", seed=7)],
        environment_factory=environment_factory,
        generations=make_generation_results(),
        experiment_id="experiment-1",
    )

    assert [episode.operator_id for episode in episodes] == ["control", "O1", "O2", "O3"]
    assert [episode.child_id for episode in episodes] == [
        None,
        "child-o1",
        "child-o2",
        "child-o3",
    ]
    assert episodes[0].parent_ids == []
    assert episodes[1].parent_id == "control"
    assert episodes[1].parent_ids == ["control"]
    assert episodes[2].parent_ids == ["control"]
    assert episodes[3].parent_id == "control"
    assert episodes[3].parent_ids == ["control", "parent-b", "parent-c"]
    assert [episode.api_calls for episode in episodes] == [3, 4, 4, 3]
    assert {usage.phase for usage in episodes[0].usage_records} == {"evaluation"}
    for child_episode in episodes[1:]:
        assert {usage.phase for usage in child_episode.usage_records} == {
            "generation",
            "evaluation",
        }
        assert sum(
            usage.calls for usage in child_episode.usage_records if usage.phase == "generation"
        ) == (0 if child_episode.operator_id == "O3" else 1)
    assert all(episode.run_status == "succeeded" for episode in episodes)
    assert all(environment.closed for environment in environments)
    control_private = artifact(ledger, episodes[0].episode_id, "private_steps.jsonl").read_text()
    assert "child-o1" not in control_private
    assert "child-o2" not in control_private
    assert "child-o3" not in control_private
    assert "generations" not in control_private
    assert ledger.read_episodes() == episodes


def test_failed_operator_generation_is_committed_with_usage_and_parent_lineage(
    tmp_path: Path,
) -> None:
    failed = O1LocalOperator().generate(
        make_genome(),
        MutationSpec(component="system_prompt"),
        OperatorContext(
            model=QueuedModel("not-json"),
            model_id="mock-model",
            child_id="uncreated-child",
        ),
    )
    assert not failed.success
    runner, ledger = make_runner(tmp_path, QueuedModel(*successful_responses()))
    environments: list[FakeEnvironment] = []

    def environment_factory(task_id: str, seed: int) -> FakeEnvironment:
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    episodes = runner.run_experiment(
        control=make_genome(),
        tasks=[TaskSpec(task_id="task-1", task_family="mock", seed=7)],
        environment_factory=environment_factory,
        generations=[
            GenerationAttempt(
                result=failed,
                operator_id="O1",
                parent_ids=("control",),
                primary_parent_id="control",
            )
        ],
        experiment_id="experiment-1",
    )

    episode = episodes[1]
    assert episode.run_status == "generation_failed"
    assert episode.task_score is None
    assert episode.api_calls == 1
    assert episode.api_cost == failed.usage.api_cost
    assert episode.total_cost == failed.usage.api_cost
    assert episode.model_version == failed.usage.model_version
    assert [usage.phase for usage in episode.usage_records] == ["generation"]
    assert episode.generation_cost == failed.usage.api_cost
    assert episode.evaluation_cost == 0
    assert episode.parent_id == "control"
    assert episode.parent_ids == ["control"]
    assert episode.child_id is None
    assert environments[1].closed is True
    assert environments[1].reset_calls == []
    assert ledger.read_episodes() == episodes


class VersionedQueuedModel(QueuedModel):
    def __init__(self, *responses: str) -> None:
        super().__init__(*responses)
        self.versions = ["mock-v1", "mock-v2", "mock-v2"]
        self.prices = ["price-a", "price-a", "price-b"]

    def _generate_once(self, request, timeout_seconds):
        index = len(self.requests)
        self._model_version = self.versions[index]
        self.price_id = self.prices[index]
        return super()._generate_once(request, timeout_seconds)


def test_usage_aggregation_keeps_model_version_and_price_groups(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, VersionedQueuedModel(*successful_responses()))

    episode = run_episode(runner, make_genome(), FakeEnvironment())

    groups = {
        (usage.model_version, usage.price_id): usage for usage in episode.usage_records
    }
    assert set(groups) == {
        ("mock-v1", "price-a"),
        ("mock-v2", "price-a"),
        ("mock-v2", "price-b"),
    }
    assert sum(usage.calls for usage in groups.values()) == 3
    assert episode.model_version is None
    assert episode.cost_complete is True


def test_non_isolated_runner_rejects_an_unenforced_timeout(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=provenance())

    with pytest.raises(ValueError, match="process_isolation"):
        AgentRunner(
            model=QueuedModel(*successful_responses()),
            model_id="mock-model",
            ledger=ledger,
            provenance=provenance(),
            process_isolation=False,
            timeout_seconds=1,
        )


def test_verifier_continue_cannot_finish_at_an_exit_without_an_edge(tmp_path: Path) -> None:
    runner, ledger = make_runner(
        tmp_path,
        QueuedModel(
            json.dumps({"plan": "solve"}),
            json.dumps({"code": "print('mock')"}),
            json.dumps({"action": "continue", "reason": "another pass is required"}),
        ),
    )

    episode = run_episode(runner, make_genome(), FakeEnvironment())

    assert episode.run_status == "invalid_workflow"
    assert episode.task_score is None
    assert ledger.read_episodes() == [episode]


class RichFailureModel(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(model_seed_supported=True)

    def generate(self, request):
        usage = UsageRecord(
            model_id=request.model_id,
            model_version="failed-model-v9",
            price_id="failed-price-v3",
            prompt_tokens=11,
            completion_tokens=0,
            total_tokens=11,
            calls=1,
            retries=3,
            elapsed_seconds=1.25,
            api_cost=Decimal("0.125"),
            cost_basis=CostBasis.ACTUAL,
        )
        raise ModelCallError(
            "rich provider failure",
            attempts=4,
            retries=3,
            elapsed_seconds=1.25,
            usage=usage,
        )

    def _generate_once(self, request, timeout_seconds):
        raise AssertionError("not used")


class RetryFailureModel(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(
            max_retries=2,
            retry_backoff_seconds=0.01,
            model_seed_supported=True,
        )
        self.attempts = 0

    def _generate_once(self, request, timeout_seconds):
        self.attempts += 1
        raise TimeoutError("retryable mock timeout")


class UnseededQueuedModel(QueuedModel):
    def __init__(self, *responses: str) -> None:
        super().__init__(*responses)
        self.model_seed_supported = False


def test_model_call_error_preserves_rich_usage_as_evaluation(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    runner, ledger = make_runner(tmp_path, RichFailureModel())

    episode = run_episode(runner, make_genome(), environment)

    assert episode.run_status == "model_failed"
    assert episode.retry_count == 3
    assert episode.input_tokens == 11
    assert episode.output_tokens == 0
    assert episode.api_cost == Decimal("0.125")
    assert episode.generation_cost == 0
    assert episode.evaluation_cost == Decimal("0.125")
    assert episode.total_cost == Decimal("0.125")
    assert episode.model_version == "failed-model-v9"
    assert len(episode.usage_records) == 1
    usage = episode.usage_records[0]
    assert usage.phase == "evaluation"
    assert usage.retries == 3
    assert usage.elapsed_seconds == 1.25
    assert usage.price_id == "failed-price-v3"
    assert environment.closed is True
    assert ledger.read_episodes() == [episode]


def test_retry_exhaustion_preserves_retry_count_elapsed_and_trajectory(
    tmp_path: Path,
) -> None:
    model = RetryFailureModel()
    runner, ledger = make_runner(tmp_path, model)

    episode = run_episode(runner, make_genome(), FakeEnvironment())

    assert model.attempts == 3
    assert episode.run_status == "model_failed"
    assert episode.api_calls == 1
    assert episode.retry_count == 2
    assert episode.usage_records[0].phase == "evaluation"
    assert episode.usage_records[0].retries == 2
    assert episode.usage_records[0].elapsed_seconds > 0
    events = [
        json.loads(line)
        for line in artifact(ledger, episode.episode_id, "trajectory.jsonl")
        .read_text()
        .splitlines()
    ]
    exceptions = [event for event in events if event["kind"] == "exception"]
    assert exceptions[-1]["retry_count"] == 2
    assert exceptions[-1]["will_retry"] is False


def test_unseeded_adapter_receives_no_model_seed_but_environment_seed_is_retained(
    tmp_path: Path,
) -> None:
    model = UnseededQueuedModel(*successful_responses())
    runner, _ = make_runner(tmp_path, model)

    episode = run_episode(runner, make_genome(), FakeEnvironment())

    assert episode.run_status == "succeeded"
    assert episode.environment_seed == 7
    assert episode.model_seed_supported is False
    assert len(model.requests) == 3
    assert all(request.seed is None for request in model.requests)
    assert all(request.prompt for request in model.requests)


def test_commit_appends_metadata_index_without_code_trajectory_or_secrets(
    tmp_path: Path,
) -> None:
    responses = (
        json.dumps({"plan": "keep fake-secret-sentinel only in private raw"}),
        json.dumps({"code": "api_key = 'fake-secret-sentinel'"}),
        json.dumps({"action": "finish", "reason": "done"}),
    )
    runner, ledger = make_runner(tmp_path, QueuedModel(*responses))

    episode = run_episode(runner, make_genome(), FakeEnvironment())

    index_path = tmp_path / "results" / "index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["episode_id"] == episode.episode_id
    assert entries[0]["run_status"] == "succeeded"
    assert entries[0]["artifact_hashes"] == episode.artifact_hashes
    serialized = index_path.read_text()
    assert "fake-secret-sentinel" not in serialized
    assert "api_key = " not in serialized
    assert "model_request" not in serialized
    assert "official mock evaluation" not in serialized
    assert "raw_steps" not in entries[0]
    assert "response" not in entries[0]
    assert "code" not in entries[0]
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    assert "fake-secret-sentinel" in private


def test_index_failure_is_explicit_after_nontransactional_ledger_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, ledger = make_runner(tmp_path, QueuedModel(*successful_responses()))

    def fail_index(self, episode):
        raise OSError("deterministic index disk failure")

    monkeypatch.setattr("tc_mo_me.agent.ResultIndexer.append", fail_index)

    with pytest.raises(ResultIndexError, match="index_failed") as caught:
        run_episode(runner, make_genome(), FakeEnvironment())

    error = caught.value
    assert error.run_status == "index_failed"
    assert error.episode.run_status == "index_failed"
    assert error.committed_episode.run_status == "succeeded"
    assert error.ledger_committed is True
    assert ledger.read_episodes() == [error.committed_episode]
    assert not (tmp_path / "results" / "index.jsonl").exists()


class OsExitEnvironment(FakeEnvironment):
    def reset(self, task_id: str):
        import os

        os._exit(23)


def test_worker_os_exit_is_reaped_and_committed_exactly_once(tmp_path: Path) -> None:
    import multiprocessing

    ledger = RunLedger(tmp_path / "results", "os-exit-run", provenance=provenance())
    runner = AgentRunner(
        model=QueuedModel(*successful_responses()),
        model_id="mock-model",
        ledger=ledger,
        provenance=provenance(),
        process_isolation=True,
        timeout_seconds=10,
    )

    episode = runner.run_episode(
        genome=make_genome(),
        environment=OsExitEnvironment(),
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="os-exit-episode",
    )

    assert episode.run_status == "runner_failed"
    assert episode.task_score is None
    assert ledger.read_episodes() == [episode]
    index_entries = [
        json.loads(line)
        for line in (tmp_path / "results" / "index.jsonl").read_text().splitlines()
    ]
    assert len(index_entries) == 1
    assert index_entries[0]["episode_id"] == "os-exit-episode"
    assert index_entries[0]["run_status"] == "runner_failed"
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    trajectory = artifact(ledger, episode.episode_id, "trajectory.jsonl").read_text()
    assert "process_exited" in private
    assert "process_exited" in trajectory
    assert not multiprocessing.active_children()


class HangingModel(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(
            max_retries=0,
            model_seed_supported=True,
            price_id="hanging-price-v1",
        )

    def _generate_once(self, request, timeout_seconds):
        import time

        time.sleep(30)
        raise AssertionError("unreachable after hard timeout")


def test_hanging_model_timeout_preserves_pending_unknown_evaluation_usage(
    tmp_path: Path,
) -> None:
    import time

    ledger = RunLedger(tmp_path / "results", "model-hang-run", provenance=provenance())
    runner = AgentRunner(
        model=HangingModel(),
        model_id="mock-model",
        ledger=ledger,
        provenance=provenance(),
        process_isolation=True,
        timeout_seconds=0.5,
    )
    started = time.monotonic()

    episode = runner.run_episode(
        genome=make_genome(),
        environment=FakeEnvironment(),
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="model-hang-episode",
    )

    assert time.monotonic() - started < 8
    assert episode.run_status == "timeout"
    assert episode.api_calls == 1
    assert episode.retry_count == 0
    assert episode.input_tokens is None
    assert episode.output_tokens is None
    assert episode.api_cost is None
    assert episode.total_cost is None
    assert episode.evaluation_cost is None
    assert episode.generation_cost == 0
    assert episode.cost_complete is False
    assert episode.cost_basis is CostBasis.UNKNOWN
    assert len(episode.usage_records) == 1
    usage = episode.usage_records[0]
    assert usage.phase == "evaluation"
    assert usage.calls == 1
    assert usage.retries == 0
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.api_cost is None
    assert usage.price_id == "hanging-price-v1"
    assert ledger.read_episodes() == [episode]
    assert len((tmp_path / "results" / "index.jsonl").read_text().splitlines()) == 1
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    assert "pending_usage" in private
    assert "process_terminated" in private


class CloseFailingEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        raise RuntimeError("deterministic close failure")


def test_generation_failure_with_close_failure_still_commits_once(tmp_path: Path) -> None:
    failed = O1LocalOperator().generate(
        make_genome(),
        MutationSpec(component="system_prompt"),
        OperatorContext(
            model=QueuedModel("not-json"),
            model_id="mock-model",
            child_id="uncreated-child",
        ),
    )
    assert not failed.success
    runner, ledger = make_runner(tmp_path, QueuedModel(*successful_responses()))
    environment = CloseFailingEnvironment()

    episode = runner.run_episode(
        genome=make_genome(),
        environment=environment,
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="generation-close-failure",
        generation=GenerationAttempt(
            result=failed,
            operator_id="O1",
            parent_ids=("control",),
            primary_parent_id="control",
        ),
    )

    assert episode.run_status == "generation_failed"
    assert episode.task_score is None
    assert episode.api_calls == 1
    assert episode.api_cost == failed.usage.api_cost
    assert episode.generation_cost == failed.usage.api_cost
    assert episode.evaluation_cost == 0
    assert episode.total_cost == failed.usage.api_cost
    assert [usage.phase for usage in episode.usage_records] == ["generation"]
    assert environment.close_calls == 1
    assert environment.reset_calls == []
    assert ledger.read_episodes() == [episode]
    index_entries = [
        json.loads(line)
        for line in (tmp_path / "results" / "index.jsonl").read_text().splitlines()
    ]
    assert len(index_entries) == 1
    assert index_entries[0]["run_status"] == "generation_failed"
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    trajectory = artifact(ledger, episode.episode_id, "trajectory.jsonl").read_text()
    assert "close_exception" in private
    assert "deterministic close failure" in private
    assert "deterministic close failure" in trajectory
    assert "generation_failed" in private


class RetryThenHangModel(ModelAdapter):
    def __init__(self) -> None:
        super().__init__(
            max_retries=2,
            model_seed_supported=True,
            price_id="retry-hang-price-v1",
        )
        self.attempts = 0

    def _generate_once(self, request, timeout_seconds):
        import time

        self.attempts += 1
        if self.attempts == 1:
            raise TimeoutError("first attempt is retryable")
        time.sleep(30)
        raise AssertionError("unreachable after second-attempt hard timeout")


def test_second_attempt_hang_preserves_observed_retry_lower_bound(tmp_path: Path) -> None:
    import time

    ledger = RunLedger(tmp_path / "results", "retry-hang-run", provenance=provenance())
    runner = AgentRunner(
        model=RetryThenHangModel(),
        model_id="mock-model",
        ledger=ledger,
        provenance=provenance(),
        process_isolation=True,
        timeout_seconds=0.75,
    )
    started = time.monotonic()

    episode = runner.run_episode(
        genome=make_genome(),
        environment=FakeEnvironment(),
        task_id="task-1",
        task_family="mock",
        seed=7,
        experiment_id="experiment-1",
        episode_id="retry-hang-episode",
    )

    assert time.monotonic() - started < 8
    assert episode.run_status == "timeout"
    assert episode.api_calls == 1
    assert episode.retry_count >= 1
    assert episode.input_tokens is None
    assert episode.output_tokens is None
    assert episode.api_cost is None
    assert episode.total_cost is None
    assert episode.cost_complete is False
    assert len(episode.usage_records) == 1
    usage = episode.usage_records[0]
    assert usage.calls == 1
    assert usage.retries >= 1
    assert usage.phase == "evaluation"
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.api_cost is None
    assert ledger.read_episodes() == [episode]
    assert len((tmp_path / "results" / "index.jsonl").read_text().splitlines()) == 1
    private = artifact(ledger, episode.episode_id, "private_steps.jsonl").read_text()
    assert '"attempt": 2' in private
    assert "observed_lower_bound" in private
    assert "process_terminated" in private
