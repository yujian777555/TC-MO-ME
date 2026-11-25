from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from tc_mo_me.environments.base import Environment, Evaluation, Observation
from tc_mo_me.genome import Genome, WorkflowNode
from tc_mo_me.ledger import EpisodeRecord, ResultIndexer, RunLedger
from tc_mo_me.model import (
    CostBasis,
    MockModelAdapter,
    ModelAdapter,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    UsageRecord,
    group_usage,
)
from tc_mo_me.operators.base import OperatorResult
from tc_mo_me.provenance import RunProvenance
from tc_mo_me.trajectory import Trajectory

TaskEnvironment = Environment

_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ResultIndexError(RuntimeError):
    """Ledger append succeeded, but the separate non-transactional index append failed."""

    def __init__(self, episode: EpisodeRecord, cause: Exception) -> None:
        super().__init__(
            f"ledger committed but result index append failed; "
            f"run_status=index_failed; episode_id={episode.episode_id}: {cause}"
        )
        self.run_status = "index_failed"
        self.episode = episode.model_copy(update={"run_status": self.run_status})
        self.committed_episode = episode
        self.ledger_committed = True


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_family: str
    seed: int


@dataclass(frozen=True)
class GenerationAttempt:
    result: OperatorResult
    operator_id: str | None = None
    parent_ids: tuple[str, ...] = ()
    primary_parent_id: str | None = None


@dataclass
class _UsageEntry:
    usage: UsageRecord
    phase: str


@dataclass
class _EpisodeState:
    run_id: str
    experiment_id: str
    episode_id: str
    task_id: str
    task_family: str
    seed: int
    model_id: str
    model_seed_supported: bool
    provenance: RunProvenance
    operator_id: str
    parent_id: str | None
    parent_ids: list[str]
    child_id: str | None
    started: float
    trajectory: Trajectory = field(default_factory=Trajectory)
    raw_steps: list[dict[str, Any]] = field(default_factory=list)
    usage_entries: list[_UsageEntry] = field(default_factory=list)
    prompt_hashes: list[str] = field(default_factory=list)
    task_score: float | None = None
    run_status: str = "running"
    execution_time: float = 0.0
    event_step: int = 0
    cleanup: str = "pending"
    checkpoint: Any = field(default=None, repr=False)

    def next_step(self) -> int:
        value = self.event_step
        self.event_step += 1
        return value


class AgentRunner:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        model_id: str,
        ledger: RunLedger,
        provenance: RunProvenance,
        max_steps: int = 8,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        model_timeout_seconds: float | None = 30.0,
        max_retries: int | None = None,
        price_id: str | None = None,
        timeout_seconds: float | None = None,
        process_isolation: bool = True,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if timeout_seconds is not None and not process_isolation:
            raise ValueError("timeout_seconds requires process_isolation=True")
        self.model = model
        self.model_id = model_id
        self.ledger = ledger
        self.provenance = provenance
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_timeout_seconds = model_timeout_seconds
        self.max_retries = max_retries
        self.price_id = price_id
        self.timeout_seconds = timeout_seconds
        self.process_isolation = process_isolation

    def run_episode(
        self,
        *,
        genome: Genome,
        environment: TaskEnvironment,
        task_id: str,
        task_family: str,
        seed: int,
        experiment_id: str,
        episode_id: str,
        generation: GenerationAttempt | None = None,
    ) -> EpisodeRecord:
        if _EPISODE_ID.fullmatch(episode_id) is None:
            raise ValueError("episode_id must be a portable single path component")
        lineage = generation.result.lineage if generation is not None else None
        if generation is not None and generation.result.success and lineage is None:
            raise ValueError("successful generation requires lineage")
        if generation is not None and not generation.result.success:
            if (
                generation.operator_id is None
                or generation.primary_parent_id is None
                or not generation.parent_ids
            ):
                raise ValueError("failed generation requires operator and parent metadata")
        operator_id = "control"
        parent_id = None
        parent_ids: list[str] = []
        child_id = None
        if lineage is not None:
            operator_id = lineage.operator_id
            parent_id = lineage.primary_parent_id
            parent_ids = list(lineage.parent_ids)
            child_id = lineage.child_id
            if genome.identity.get("genome_id") != child_id:
                raise ValueError("executed genome must be the generated child")
        elif generation is not None:
            operator_id = str(generation.operator_id)
            parent_id = generation.primary_parent_id
            parent_ids = list(generation.parent_ids)
        state = _EpisodeState(
            run_id=self.ledger.run_id,
            experiment_id=experiment_id,
            episode_id=episode_id,
            task_id=task_id,
            task_family=task_family,
            seed=seed,
            model_id=self.model_id,
            model_seed_supported=self.model.model_seed_supported,
            provenance=self.provenance,
            operator_id=operator_id,
            parent_id=parent_id,
            parent_ids=parent_ids,
            child_id=child_id,
            started=time.monotonic(),
        )
        if generation is not None:
            generation_usage = generation.result.usage.model_copy(
                update={"phase": "generation"}
            )
            state.usage_entries.append(_UsageEntry(generation_usage, "generation"))
            state.raw_steps.append({"generation": generation.result.model_dump(mode="json")})
            state.trajectory.record_score(
                state.next_step(),
                None,
                details={
                    "event": "child_generation",
                    "success": generation.result.success,
                    "operator_id": operator_id,
                },
            )
            if not generation.result.success:
                state.run_status = "generation_failed"
                state.raw_steps.append({"termination": "generation_failed"})
                self._close_environment(environment, state, preserve_status=True)
                episode = self._build_record(state)
                return self._commit(state, episode)
        if self.process_isolation:
            return self._run_isolated(genome, environment, state)
        self._execute_local(genome, environment, state)
        episode = self._build_record(state)
        return self._commit(state, episode)

    def run_experiment(
        self,
        *,
        control: Genome,
        tasks: Sequence[TaskSpec],
        environment_factory: Callable[[str, int], TaskEnvironment] | None = None,
        generations: Sequence[GenerationAttempt] = (),
        experiment_id: str,
        episode_environment_factory: (
            Callable[[str, int, str, str], TaskEnvironment] | None
        ) = None,
    ) -> list[EpisodeRecord]:
        if (environment_factory is None) == (episode_environment_factory is None):
            raise ValueError("provide exactly one environment factory")
        episodes: list[EpisodeRecord] = []

        def execute(genome: Genome, task: TaskSpec, generation: GenerationAttempt | None) -> None:
            index = len(episodes)
            episode_id = f"episode-{index:04d}"
            if episode_environment_factory is not None:
                environment = episode_environment_factory(
                    task.task_id,
                    task.seed,
                    self.ledger.run_id,
                    episode_id,
                )
            elif environment_factory is not None:
                environment = environment_factory(task.task_id, task.seed)
            else:
                raise AssertionError("environment factory validation already ran")
            episodes.append(
                self.run_episode(
                    genome=genome,
                    environment=environment,
                    task_id=task.task_id,
                    task_family=task.task_family,
                    seed=task.seed,
                    experiment_id=experiment_id,
                    episode_id=episode_id,
                    generation=generation,
                )
            )

        for task in tasks:
            execute(control, task, None)
        for generation in generations:
            genome = generation.result.child if generation.result.child is not None else control
            for task in tasks:
                execute(genome, task, generation)
        return episodes

    def _run_isolated(
        self,
        genome: Genome,
        environment: TaskEnvironment,
        state: _EpisodeState,
    ) -> EpisodeRecord:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        options = {
            "model": self.model,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_timeout_seconds": self.model_timeout_seconds,
            "max_retries": self.max_retries,
            "price_id": self.price_id,
        }
        process = context.Process(
            target=_isolated_entry,
            args=(
                options,
                self.ledger.run_dir.parent.parent,
                self.ledger.run_id,
                self.provenance,
                genome,
                environment,
                state,
                sender,
            ),
        )
        latest = state
        deadline: float | None = None
        startup_deadline = time.monotonic() + 30
        process.start()
        sender.close()
        final_episode: EpisodeRecord | None = None

        def fail_worker(error: BaseException) -> EpisodeRecord:
            latest.run_status = "runner_failed"
            latest.cleanup = "process_exited"
            latest.trajectory.record_score(
                latest.next_step(),
                None,
                details={"event": "termination", "reason": "process_exited"},
            )
            latest.raw_steps.append(
                {
                    "worker_exception": {
                        **_exception_payload(error),
                        "process_exitcode": process.exitcode,
                    },
                    "environment_cleanup": "process_exited",
                }
            )
            return self._build_record(latest)

        try:
            while final_episode is None:
                try:
                    has_message = receiver.poll(0.05)
                    message = receiver.recv() if has_message else None
                except (EOFError, BrokenPipeError, OSError) as error:
                    final_episode = fail_worker(error)
                    break
                if message is not None:
                    if message["kind"] == "ready":
                        if self.timeout_seconds is not None:
                            deadline = time.monotonic() + self.timeout_seconds
                    elif message["kind"] == "checkpoint":
                        latest = message["state"]
                    elif message["kind"] == "final":
                        latest = message["state"]
                        final_episode = message["episode"]
                    else:
                        latest = message["state"]
                        latest.run_status = "runner_failed"
                        latest.raw_steps.append({"worker_exception": message["error"]})
                        final_episode = self._build_record(latest)
                    continue
                active_deadline = deadline if deadline is not None else startup_deadline
                if time.monotonic() >= active_deadline:
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=2)
                    latest.run_status = "timeout"
                    latest.cleanup = "process_terminated"
                    latest.trajectory.record_score(
                        latest.next_step(),
                        None,
                        details={
                            "event": "termination",
                            "reason": "timeout",
                            "environment_cleanup": "process_terminated",
                        },
                    )
                    latest.raw_steps.append(
                        {"termination": "timeout", "environment_cleanup": "process_terminated"}
                    )
                    final_episode = self._build_record(latest)
                    break
                if not process.is_alive():
                    process.join(timeout=0)
                    try:
                        if receiver.poll(0.1):
                            continue
                    except (EOFError, BrokenPipeError, OSError) as error:
                        final_episode = fail_worker(error)
                    else:
                        final_episode = fail_worker(RuntimeError("worker exited without a final record"))
                    break
        finally:
            receiver.close()
            _reap_worker(process)
        return self._commit(latest, final_episode)

    def _execute_local(
        self,
        genome: Genome,
        environment: TaskEnvironment,
        state: _EpisodeState,
    ) -> None:
        try:
            observation = environment.reset(state.task_id)
            self._execute_graph(genome, environment, observation, state)
        except Exception as error:
            if state.run_status == "running":
                state.run_status = "environment_failed"
            self._record_exception(state, error)
            if state.raw_steps:
                state.raw_steps[-1]["runner_exception"] = _exception_payload(error)
            else:
                state.raw_steps.append({"runner_exception": _exception_payload(error)})
        finally:
            self._close_environment(environment, state)
        self._write_artifacts(state)

    def _close_environment(
        self,
        environment: TaskEnvironment,
        state: _EpisodeState,
        *,
        preserve_status: bool = False,
    ) -> None:
        try:
            environment.close()
        except Exception as error:
            state.cleanup = "close_failed"
            if not preserve_status:
                state.run_status = "environment_close_failed"
            self._record_exception(state, error)
            state.raw_steps.append({"close_exception": _exception_payload(error)})
        else:
            state.cleanup = "closed"

    def _execute_graph(
        self,
        genome: Genome,
        environment: TaskEnvironment,
        observation: Observation,
        state: _EpisodeState,
    ) -> None:
        nodes = {node.id: node for node in genome.workflow_graph.nodes}
        current_id = genome.workflow_graph.entry_id
        plan: str | None = None
        code: str | None = None
        output: str | None = None
        last_exception: str | None = None
        visits = 0

        while state.run_status == "running":
            if visits >= self.max_steps:
                state.run_status = "max_steps_exceeded"
                state.trajectory.record_score(
                    state.next_step(),
                    None,
                    details={"event": "termination", "reason": "max_steps_exceeded"},
                )
                state.raw_steps.append({"termination": "max_steps_exceeded"})
                self._write_artifacts(state)
                return
            node = nodes[current_id]
            raw: dict[str, Any] = {
                "visit": visits,
                "node_id": node.id,
                "node_type": node.type.value,
            }
            state.raw_steps.append(raw)
            visits += 1

            if node.type.value == "executor":
                if code is None:
                    state.run_status = "invalid_workflow"
                    error = RuntimeError("executor reached before generator produced code")
                    raw["exception"] = _exception_payload(error)
                    self._record_exception(state, error)
                    return
                state.trajectory.record_code(state.next_step(), code)
                raw["code"] = code
                self._write_artifacts(state)
                started = time.monotonic()
                try:
                    output = environment.execute(code)
                except Exception as error:
                    state.execution_time += time.monotonic() - started
                    state.run_status = "execution_failed"
                    last_exception = str(error)
                    raw["exception"] = _exception_payload(error)
                    self._record_exception(state, error)
                    self._write_artifacts(state)
                    return
                state.execution_time += time.monotonic() - started
                raw["output"] = output
                state.trajectory.record_output(state.next_step(), output)
                self._write_artifacts(state)
            else:
                prompt = _node_prompt(
                    genome=genome,
                    node=node,
                    observation=observation,
                    plan=plan,
                    code=code,
                    output=output,
                    last_exception=last_exception,
                )
                response = self._call_model(genome, node, prompt, raw, state)
                if response is None:
                    return
                if node.type.value == "planner":
                    plan = response.content
                elif node.type.value == "generator":
                    try:
                        code = _parse_generator(response.content)
                    except (ValueError, json.JSONDecodeError) as error:
                        state.run_status = "invalid_model_output"
                        raw["exception"] = _exception_payload(error)
                        self._record_exception(state, error)
                        self._write_artifacts(state)
                        return
                    raw["code"] = code
                    state.trajectory.record_code(state.next_step(), code)
                    self._write_artifacts(state)
                else:
                    try:
                        action, _reason = _parse_verifier(response.content)
                    except (ValueError, json.JSONDecodeError) as error:
                        state.run_status = "invalid_model_output"
                        raw["exception"] = _exception_payload(error)
                        self._record_exception(state, error)
                        self._write_artifacts(state)
                        return
                    raw["verifier_action"] = action
                    state.trajectory.record_score(
                        state.next_step(),
                        None,
                        details={"event": "verifier_decision", "action": action},
                    )
                    if action == "finish":
                        evaluation = self._evaluate(environment, raw, state)
                        if evaluation is None:
                            return
                        state.task_score = 1.0 if evaluation["success"] else 0.0
                        state.run_status = "succeeded" if evaluation["success"] else "task_failed"
                        self._write_artifacts(state)
                        return

            outgoing = [
                edge.target
                for edge in genome.workflow_graph.edges
                if edge.source == current_id
            ]
            if not outgoing:
                if current_id not in genome.workflow_graph.exit_ids:
                    state.run_status = "invalid_workflow"
                    error = RuntimeError(f"node {current_id!r} has no outgoing edge and is not an exit")
                    raw["exception"] = _exception_payload(error)
                    self._record_exception(state, error)
                    return
                if node.type.value == "verifier":
                    state.run_status = "invalid_workflow"
                    error = RuntimeError("verifier requested continue but has no outgoing edge")
                    raw["exception"] = _exception_payload(error)
                    self._record_exception(state, error)
                    return
                evaluation = self._evaluate(environment, raw, state)
                if evaluation is None:
                    return
                state.task_score = 1.0 if evaluation["success"] else 0.0
                state.run_status = "succeeded" if evaluation["success"] else "task_failed"
                self._write_artifacts(state)
                return
            target_key = "continue_target" if node.type.value == "verifier" else "next_target"
            configured_target = node.config.get(target_key)
            if configured_target is None:
                if len(outgoing) != 1:
                    state.run_status = "invalid_workflow"
                    error = RuntimeError(
                        f"node {current_id!r} requires {target_key} for its multiple outgoing edges"
                    )
                    raw["exception"] = _exception_payload(error)
                    self._record_exception(state, error)
                    return
                current_id = outgoing[0]
            elif configured_target not in outgoing:
                state.run_status = "invalid_workflow"
                error = RuntimeError(f"configured target {configured_target!r} is not an outgoing edge")
                raw["exception"] = _exception_payload(error)
                self._record_exception(state, error)
                return
            else:
                current_id = str(configured_target)

    def _call_model(
        self,
        genome: Genome,
        node: WorkflowNode,
        prompt: str,
        raw: dict[str, Any],
        state: _EpisodeState,
    ) -> ModelResponse | None:
        request = ModelRequest(
            model_id=self.model_id,
            prompt=prompt,
            system_prompt=genome.system_prompt,
            seed=state.seed if self.model.model_seed_supported else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.model_timeout_seconds,
            max_retries=self.max_retries,
            price_id=self.price_id,
        )
        raw["request"] = request.model_dump(mode="json")
        pending_entry = _UsageEntry(
            UsageRecord(
                model_id=self.model_id,
                model_version=None,
                price_id=(
                    self.price_id if self.price_id is not None else self.model.price_id
                ),
                phase="evaluation",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                calls=1,
                retries=0,
                elapsed_seconds=0.0,
                api_cost=None,
                cost_basis=CostBasis.UNKNOWN,
            ),
            "evaluation",
        )
        state.usage_entries.append(pending_entry)
        raw["pending_usage"] = pending_entry.usage.model_dump(mode="json")
        state.prompt_hashes.append(request.prompt_hash)
        state.trajectory.record_score(
            state.next_step(),
            None,
            details={
                "event": "model_request",
                "node_id": node.id,
                "node_type": node.type.value,
                "prompt_hash": request.prompt_hash,
            },
        )
        self._write_artifacts(state)
        try:
            response = self.model.generate(request)
        except Exception as error:
            if isinstance(error, ModelCallError):
                usage = error.usage.model_copy(update={"phase": "evaluation"})
                retries = error.retries
                raw["model_call_error"] = {
                    "attempts": error.attempts,
                    "retries": error.retries,
                    "elapsed_seconds": error.elapsed_seconds,
                    "usage": usage.model_dump(mode="json"),
                }
            else:
                usage = UsageRecord(
                    model_id=self.model_id,
                    model_version=None,
                    price_id=(
                        self.price_id if self.price_id is not None else self.model.price_id
                    ),
                    phase="evaluation",
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    calls=1,
                    retries=0,
                    elapsed_seconds=0.0,
                    api_cost=None,
                    cost_basis=CostBasis.UNKNOWN,
                )
                retries = 0
            pending_entry.usage = usage
            pending_entry.phase = "evaluation"
            state.run_status = "model_failed"
            raw["exception"] = _exception_payload(error)
            self._record_exception(state, error, retry_count=retries)
            self._write_artifacts(state)
            return None

        usage = response.usage.model_copy(update={"phase": "evaluation"})
        response = response.model_copy(update={"usage": usage})
        raw["response"] = response.model_dump(mode="json")
        pending_entry.usage = usage
        pending_entry.phase = "evaluation"
        state.model_seed_supported = response.model_seed_supported
        state.trajectory.record_score(
            state.next_step(),
            None,
            details={
                "event": "model_response",
                "node_id": node.id,
                "node_type": node.type.value,
                "prompt_hash": response.prompt_hash,
                "response_sha256": hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
                "response_summary": response.content[:160],
                "model_id": response.model_id,
                "model_version": response.model_version,
                "price_id": response.usage.price_id,
                "finish_reason": response.finish_reason,
            },
        )
        self._write_artifacts(state)
        return response

    def _evaluate(
        self,
        environment: TaskEnvironment,
        raw: dict[str, Any],
        state: _EpisodeState,
    ) -> Evaluation | None:
        try:
            evaluation = environment.evaluate()
        except Exception as error:
            state.run_status = "evaluation_failed"
            raw["exception"] = _exception_payload(error)
            self._record_exception(state, error)
            self._write_artifacts(state)
            return None
        raw["evaluation"] = evaluation
        score = 1.0 if evaluation["success"] else 0.0
        state.trajectory.record_score(
            state.next_step(),
            score,
            details={
                "event": "official_evaluation",
                "success": evaluation["success"],
                "report": evaluation["report"],
                "details": evaluation["details"],
            },
        )
        self._write_artifacts(state)
        return evaluation

    def _record_exception(
        self,
        state: _EpisodeState,
        error: BaseException,
        *,
        retry_count: int = 0,
    ) -> None:
        state.trajectory.record_exception(
            state.next_step(),
            error,
            retry_count=retry_count,
            will_retry=False,
        )
        self._write_artifacts(state)

    def _build_record(self, state: _EpisodeState) -> EpisodeRecord:
        usages = [entry.usage for entry in state.usage_entries]
        grouped = list(group_usage(usages).values())
        input_tokens = _sum_optional(usage.prompt_tokens for usage in usages)
        output_tokens = _sum_optional(usage.completion_tokens for usage in usages)
        costs_known = all(usage.api_cost is not None for usage in usages)
        api_cost = (
            sum((usage.api_cost for usage in usages if usage.api_cost is not None), Decimal(0))
            if costs_known
            else None
        )

        def phase_cost(phase: str) -> Decimal | None:
            entries = [entry.usage for entry in state.usage_entries if entry.phase == phase]
            if any(usage.api_cost is None for usage in entries):
                return None
            return sum(
                (usage.api_cost for usage in entries if usage.api_cost is not None),
                Decimal(0),
            )

        versions = {usage.model_version for usage in usages}
        model_version = next(iter(versions)) if len(versions) == 1 else None
        if state.prompt_hashes:
            prompt_digest = state.prompt_hashes[0]
        else:
            prompt_digest = ModelRequest(
                model_id=state.model_id,
                prompt="",
                system_prompt=None,
            ).prompt_hash
        return EpisodeRecord(
            run_id=state.run_id,
            experiment_id=state.experiment_id,
            episode_id=state.episode_id,
            task_id=state.task_id,
            task_family=state.task_family,
            parent_id=state.parent_id,
            child_id=state.child_id,
            operator_id=state.operator_id,
            parent_ids=state.parent_ids,
            model_id=state.model_id,
            model_version=model_version,
            environment_seed=state.seed,
            model_seed_supported=state.model_seed_supported,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_calls=sum(usage.calls for usage in usages),
            retry_count=sum(usage.retries for usage in usages),
            api_cost=api_cost,
            wall_time=max(0.0, time.monotonic() - state.started),
            task_score=state.task_score,
            run_status=state.run_status,
            git_sha=state.provenance.git_sha,
            config_hash=state.provenance.config_hash,
            prompt_hash=prompt_digest,
            generation_cost=phase_cost("generation"),
            evaluation_cost=phase_cost("evaluation"),
            execution_time=state.execution_time,
            timestamp_utc=datetime.now(UTC),
            usage_records=grouped,
        )

    def _write_artifacts(self, state: _EpisodeState) -> None:
        directory = self.ledger.run_dir / "episodes" / state.episode_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "trajectory.jsonl").write_text(state.trajectory.to_jsonl(), encoding="utf-8")
        serialized = "".join(
            json.dumps(step, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for step in state.raw_steps
        )
        (directory / "private_steps.jsonl").write_text(serialized, encoding="utf-8")
        if state.checkpoint is not None:
            state.checkpoint.send({"kind": "checkpoint", "state": _state_snapshot(state)})

    def _commit(self, state: _EpisodeState, episode: EpisodeRecord) -> EpisodeRecord:
        self._write_artifacts(state)
        directory = self.ledger.run_dir / "episodes" / state.episode_id
        hashes = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("private_steps.jsonl", "trajectory.jsonl")
        }
        episode = episode.model_copy(update={"artifact_hashes": hashes})
        self.ledger.append(episode)
        index_path = self.ledger.run_dir.parent.parent / "index.jsonl"
        try:
            ResultIndexer(index_path).append(episode)
        except Exception as error:
            raise ResultIndexError(episode, error) from error
        return episode


def _reap_worker(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _state_snapshot(state: _EpisodeState) -> _EpisodeState:
    checkpoint = state.checkpoint
    state.checkpoint = None
    try:
        return deepcopy(state)
    finally:
        state.checkpoint = checkpoint


def _isolated_entry(
    options: dict[str, Any],
    results_root: Path,
    run_id: str,
    provenance: RunProvenance,
    genome: Genome,
    environment: TaskEnvironment,
    state: _EpisodeState,
    sender: Any,
) -> None:
    try:
        state.checkpoint = sender
        sender.send({"kind": "ready"})
        ledger = RunLedger(results_root, run_id, provenance=provenance, allow_existing=True)
        runner = AgentRunner(
            **options,
            ledger=ledger,
            provenance=provenance,
            process_isolation=False,
        )
        runner._execute_local(genome, environment, state)
        episode = runner._build_record(state)
        sender.send({"kind": "final", "state": _state_snapshot(state), "episode": episode})
    except Exception as error:
        state.run_status = "runner_failed"
        sender.send(
            {
                "kind": "worker_error",
                "state": _state_snapshot(state),
                "error": _exception_payload(error),
            }
        )
    finally:
        sender.close()


def _sum_optional(values: Any) -> int | None:
    items = list(values)
    if any(value is None for value in items):
        return None
    return sum(value for value in items if value is not None)


def _node_prompt(
    *,
    genome: Genome,
    node: WorkflowNode,
    observation: Observation,
    plan: str | None,
    code: str | None,
    output: str | None,
    last_exception: str | None,
) -> str:
    formats = {
        "planner": "Return a concise plan.",
        "generator": 'Return only strict JSON: {"code":"Python source code"}.',
        "verifier": (
            'Return only strict JSON: {"action":"finish","reason":"..."} or '
            '{"action":"continue","reason":"..."}.'
        ),
    }
    payload = {
        "task_id": observation["task_id"],
        "task_instruction": observation["instruction"],
        "node_id": node.id,
        "node_type": node.type.value,
        "node_config": node.config,
        "policies": {
            "planning": genome.planning_policy,
            "code_generation": genome.code_generation_policy,
            "verification": genome.verification_policy,
        },
        "response_format": formats[node.type.value],
        "state": {
            "plan": plan,
            "code": code,
            "output": output,
            "exception": last_exception,
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json(content: str) -> Any:
    return json.loads(
        content,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _parse_generator(content: str) -> str:
    payload = _strict_json(content)
    if not isinstance(payload, dict) or set(payload) != {"code"}:
        raise ValueError("generator output must be a JSON object containing only code")
    code = payload["code"]
    if not isinstance(code, str) or not code.strip():
        raise ValueError("generator code must be a non-empty string")
    return code


def _parse_verifier(content: str) -> tuple[str, str]:
    payload = _strict_json(content)
    if not isinstance(payload, dict) or set(payload) != {"action", "reason"}:
        raise ValueError("verifier output must contain exactly action and reason")
    action = payload["action"]
    reason = payload["reason"]
    if action not in {"finish", "continue"} or not isinstance(reason, str):
        raise ValueError("verifier action must be finish or continue and reason must be a string")
    return action, reason


def _exception_payload(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


class WorkflowMockAdapter(MockModelAdapter):
    def _generate_once(
        self,
        request: ModelRequest,
        timeout_seconds: float | None,
    ) -> ModelResponse:
        node_type = json.loads(request.prompt)["node_type"]
        if node_type == "planner":
            content = json.dumps({"plan": "Generate code, execute it, and verify the result."})
        elif node_type == "generator":
            content = json.dumps({"code": "pass"})
        elif node_type == "verifier":
            content = json.dumps({"action": "finish", "reason": "mock checks are complete"})
        else:
            raise ValueError(f"unsupported mock node type: {node_type}")
        self._script[request.prompt] = content
        return super()._generate_once(request, timeout_seconds)


class MockTaskEnvironment:
    def __init__(self, task_id: str, instruction: str, seed: int = 0) -> None:
        self.task_id = task_id
        self.instruction = instruction
        self.seed = seed
        self.code: str | None = None
        self.closed = False

    def reset(self, task_id: str) -> Observation:
        if task_id != self.task_id:
            raise ValueError("mock environment received an unexpected task_id")
        return {"task_id": task_id, "instruction": self.instruction}

    def execute(self, code: str) -> str:
        self.code = code
        return "deterministic mock execution output"

    def evaluate(self) -> Evaluation:
        return {
            "success": self.code is not None,
            "report": "mock check only; not an AppWorld evaluation",
            "details": {"mock": True, "code_received": self.code is not None},
        }

    def trajectory(self) -> list[dict[str, str]]:
        return []

    def close(self) -> None:
        self.closed = True


__all__ = [
    "AgentRunner",
    "GenerationAttempt",
    "MockTaskEnvironment",
    "ResultIndexError",
    "TaskEnvironment",
    "TaskSpec",
    "WorkflowMockAdapter",
]
