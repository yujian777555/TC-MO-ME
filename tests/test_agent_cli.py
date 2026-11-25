from __future__ import annotations

import json
from pathlib import Path

import yaml

from tc_mo_me.cli import main
from tc_mo_me.ledger import RunLedger


def test_mock_config_runs_one_e0_episode_through_cli(
    tmp_path: Path, capsys
) -> None:
    config = Path(__file__).parents[1] / "configs" / "e0" / "mock.yaml"
    results_root = tmp_path / "results"

    exit_code = main(
        ["run", str(config), "--results-root", str(results_root), "--allow-dirty"]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_id"] == "e0-mock-smoke"
    assert summary["episode_count"] == 1
    assert summary["episodes"][0]["run_status"] == "succeeded"
    assert summary["episodes"][0]["task_score"] == 1.0
    assert summary["episodes"][0]["api_calls"] == 3
    ledger = RunLedger(results_root, "e0-mock-smoke", allow_existing=True)
    episodes = ledger.read_episodes()
    assert len(episodes) == 1
    assert episodes[0].model_id == "e0-deterministic-mock"
    assert episodes[0].api_cost == 0
    assert episodes[0].cost_complete is True
    assert episodes[0].task_family == "mock"


def test_real_api_smoke_is_disabled_and_has_hard_limits_without_prices(
    tmp_path: Path, capsys
) -> None:
    config = Path(__file__).parents[1] / "configs" / "e0" / "real_api_smoke.yaml"
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))

    assert payload["enabled"] is False
    assert payload["limits"] == {
        "max_episodes": 6,
        "max_model_calls": 40,
        "max_api_cost_usd": "2.00",
    }
    assert payload["model"]["model_id"] is None
    assert payload["model"]["price_id"] is None
    assert payload["model"]["input_price_per_million"] is None
    assert payload["model"]["output_price_per_million"] is None
    assert {
        "model.model_id",
        "model.price_id",
        "model.input_price_per_million",
        "model.output_price_per_million",
        "model.api_key_env",
    } <= set(payload["required_user_provided_fields"])
    assert "auto_execute" in payload
    assert payload["auto_execute"] is False

    exit_code = main(
        ["run", str(config), "--results-root", str(tmp_path), "--allow-dirty"]
    )

    assert exit_code == 2
    assert "disabled" in capsys.readouterr().err
    assert not (tmp_path / "raw").exists()


class FakeAppWorldEnvironment:
    constructions: list[dict[str, object]] = []

    def __init__(
        self,
        *,
        root,
        experiment_name: str,
        seed: int,
        timeout_seconds: int,
    ) -> None:
        self.task_id = ""
        self.code = None
        self.__class__.constructions.append(
            {
                "root": root,
                "experiment_name": experiment_name,
                "seed": seed,
                "timeout_seconds": timeout_seconds,
            }
        )

    def reset(self, task_id: str):
        self.task_id = task_id
        return {"task_id": task_id, "instruction": "Fake AppWorld instruction"}

    def execute(self, code: str):
        self.code = code
        return "fake AppWorld no-op output"

    def evaluate(self):
        return {
            "success": False,
            "report": "fake official zero",
            "details": {"fake_appworld": True},
        }

    def trajectory(self):
        return []

    def close(self):
        pass


def test_appworld_mock_config_uses_unique_experiment_names_without_real_appworld(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    config = Path(__file__).parents[1] / "configs" / "e0" / "appworld_mock.yaml"
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert payload["run_id"] == "e0-appworld-mock-smoke-v2"
    assert [task["task_id"] for task in payload["tasks"]] == [
        "82e2fac_1",
        "82e2fac_2",
        "82e2fac_3",
    ]
    assert payload["model"]["adapter"] == "mock"
    assert payload["environment"]["kind"] == "appworld"
    assert payload["environment"]["root"] == ".local/appworld"
    assert payload["runner"]["process_isolation"] is True

    FakeAppWorldEnvironment.constructions = []
    monkeypatch.setattr(
        "tc_mo_me.cli.AppWorldEnvironment",
        FakeAppWorldEnvironment,
        raising=False,
    )
    results_root = tmp_path / "results"

    exit_code = main(
        ["run", str(config), "--results-root", str(results_root), "--allow-dirty"]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["episode_count"] == 3
    assert [episode["task_score"] for episode in summary["episodes"]] == [0.0, 0.0, 0.0]
    assert [episode["run_status"] for episode in summary["episodes"]] == [
        "task_failed",
        "task_failed",
        "task_failed",
    ]
    assert FakeAppWorldEnvironment.constructions == [
        {
            "root": ".local/appworld",
            "experiment_name": f"e0-appworld-mock-smoke-v2-episode-{index:04d}",
            "seed": 0,
            "timeout_seconds": 30,
        }
        for index in range(3)
    ]
    ledger = RunLedger(results_root, payload["run_id"], allow_existing=True)
    episodes = ledger.read_episodes()
    assert len(episodes) == 3
    for episode in episodes:
        assert {usage.phase for usage in episode.usage_records} == {"evaluation"}
        private = (
            ledger.run_dir / "episodes" / episode.episode_id / "private_steps.jsonl"
        ).read_text()
        assert '"code": "pass"' in private
    assert len((results_root / "index.jsonl").read_text().splitlines()) == 3
