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
