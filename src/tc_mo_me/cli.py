from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from tc_mo_me.agent import (
    AgentRunner,
    MockTaskEnvironment,
    TaskEnvironment,
    TaskSpec,
    WorkflowMockAdapter,
)
from tc_mo_me.environments.appworld_env import AppWorldEnvironment
from tc_mo_me.genome import validate_genome
from tc_mo_me.ledger import EpisodeRecord, RunLedger
from tc_mo_me.provenance import capture_provenance


def _decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def run_config(
    config_path: str | Path,
    *,
    results_root: str | Path | None = None,
    allow_dirty: bool = False,
) -> tuple[RunLedger, list[EpisodeRecord]]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("configuration must contain a model mapping")
    if config.get("enabled") is not True:
        raise ValueError("configuration is disabled; no model or environment was started")
    model_config = config["model"]
    if model_config.get("adapter") != "mock":
        raise ValueError("this entry point executes only the deterministic mock adapter")
    if model_config.get("enabled") is not True:
        raise ValueError("model execution is disabled")

    runner_config = config.get("runner", {})
    tasks_config = config.get("tasks", [])
    limits = config.get("limits", {})
    if len(tasks_config) > limits.get("max_episodes", len(tasks_config)):
        raise ValueError("configured tasks exceed max_episodes")
    provenance = capture_provenance(path, allow_dirty=allow_dirty)
    output_root = Path(results_root or runner_config.get("results_root", "results"))
    ledger = RunLedger(output_root, config["run_id"], provenance=provenance)
    model = WorkflowMockAdapter(
        model_version=model_config.get("model_version", "mock-v1"),
        timeout_seconds=model_config.get("timeout_seconds", 30.0),
        max_retries=model_config.get("max_retries", 0),
        price_id=model_config.get("price_id"),
        input_price_per_million=_decimal(model_config.get("input_price_per_million")),
        output_price_per_million=_decimal(model_config.get("output_price_per_million")),
    )
    runner = AgentRunner(
        model=model,
        model_id=model_config["model_id"],
        ledger=ledger,
        provenance=provenance,
        max_steps=runner_config.get("max_steps", 8),
        temperature=model_config.get("temperature", 0.0),
        max_tokens=model_config.get("max_tokens"),
        model_timeout_seconds=model_config.get("timeout_seconds", 30.0),
        max_retries=model_config.get("max_retries"),
        price_id=model_config.get("price_id"),
        timeout_seconds=runner_config.get("timeout_seconds"),
        process_isolation=runner_config.get("process_isolation", True),
    )
    tasks = [
        TaskSpec(task_id=item["task_id"], task_family=item["task_family"], seed=item["seed"])
        for item in tasks_config
    ]
    environment_config = config.get("environment", {})
    environment_kind = environment_config.get("kind")

    def environment_factory(task_id: str, seed: int) -> TaskEnvironment:
        if environment_kind != "mock":
            raise ValueError("unsupported two-argument environment factory")
        return MockTaskEnvironment(
            task_id=task_id,
            instruction=environment_config["instruction"],
            seed=seed,
        )

    def episode_environment_factory(
        task_id: str,
        seed: int,
        run_id: str,
        episode_id: str,
    ) -> TaskEnvironment:
        if environment_kind != "appworld":
            raise ValueError("unsupported episode environment factory")
        return AppWorldEnvironment(
            root=environment_config["root"],
            experiment_name=f"{run_id}-{episode_id}",
            seed=seed,
            timeout_seconds=environment_config.get("execution_timeout_seconds", 30),
        )

    factory_options: dict[str, Any]
    if environment_kind == "mock":
        factory_options = {"environment_factory": environment_factory}
    elif environment_kind == "appworld":
        factory_options = {"episode_environment_factory": episode_environment_factory}
    else:
        raise ValueError("environment kind must be mock or appworld")

    episodes = runner.run_experiment(
        control=validate_genome(config["genome"]),
        tasks=tasks,
        experiment_id=config["experiment_id"],
        **factory_options,
    )
    if sum(episode.api_calls for episode in episodes) > limits.get(
        "max_model_calls", sys.maxsize
    ):
        raise ValueError("configured run exceeded max_model_calls")
    return ledger, episodes


def _summary(ledger: RunLedger, episodes: list[EpisodeRecord]) -> dict[str, Any]:
    return {
        "run_id": ledger.run_id,
        "episode_count": len(episodes),
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "run_status": episode.run_status,
                "task_score": episode.task_score,
                "api_calls": episode.api_calls,
                "api_cost": None if episode.api_cost is None else str(episode.api_cost),
            }
            for episode in episodes
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tc-mo-me")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a mock E0 configuration")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--results-root", type=Path)
    run_parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        ledger, episodes = run_config(
            args.config,
            results_root=args.results_root,
            allow_dirty=args.allow_dirty,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(ledger, episodes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
