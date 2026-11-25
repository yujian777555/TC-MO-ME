import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tc_mo_me.ledger import DuplicateEpisodeError, EpisodeRecord, ResultIndexer, RunLedger
from tc_mo_me.model import CostBasis, UsageRecord
from tc_mo_me.provenance import (
    DirtyRepositoryError,
    RunProvenance,
    capture_provenance,
    sha256_file,
)
from tc_mo_me.trajectory import EventType, Trajectory

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
CONFIG_HASH = "a" * 64
PROMPT_HASH = "b" * 64
ARTIFACT_HASH = "c" * 64


def usage_record(
    *,
    model_id: str = "model-a",
    model_version: str | None = "v1",
    price_id: str | None = "price-a",
    phase: str | None = "generation",
    prompt_tokens: int | None = 3,
    completion_tokens: int | None = 2,
    total_tokens: int | None = 5,
    calls: int = 1,
    retries: int = 0,
    cost_basis: CostBasis = CostBasis.ACTUAL,
    api_cost: Decimal | None = Decimal("0.25"),
) -> UsageRecord:
    return UsageRecord(
        model_id=model_id,
        model_version=model_version,
        price_id=price_id,
        phase=phase,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        calls=calls,
        retries=retries,
        elapsed_seconds=1.5,
        api_cost=api_cost,
        cost_basis=cost_basis,
    )


def episode_data() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "experiment_id": "experiment-1",
        "episode_id": "episode-1",
        "task_id": "task-1",
        "task_family": "family-1",
        "parent_id": None,
        "child_id": None,
        "operator_id": "control",
        "model_id": "model-a",
        "model_version": "v1",
        "environment_seed": 7,
        "model_seed_supported": True,
        "input_tokens": 6,
        "output_tokens": 4,
        "api_calls": 2,
        "retry_count": 0,
        "api_cost": Decimal("0.30"),
        "wall_time": 1.5,
        "task_score": 0.0,
        "run_status": "succeeded",
        "git_sha": GIT_SHA,
        "config_hash": CONFIG_HASH,
        "prompt_hash": PROMPT_HASH,
        "parent_ids": [],
        "generation_cost": Decimal("0.25"),
        "evaluation_cost": Decimal("0.05"),
        "total_cost": Decimal("0.30"),
        "execution_time": 1.5,
        "timestamp_utc": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "artifact_hashes": {"answer.py": ARTIFACT_HASH},
        "usage_records": [
            usage_record(),
            usage_record(phase="evaluation", api_cost=Decimal("0.05")),
        ],
    }


def test_episode_record_requires_core_fields_and_rejects_unknown_fields() -> None:
    data = episode_data()
    del data["task_id"]

    with pytest.raises(ValidationError):
        EpisodeRecord.model_validate(data)

    data = episode_data()
    data["unexpected"] = "not allowed"
    with pytest.raises(ValidationError):
        EpisodeRecord.model_validate(data)


@pytest.mark.parametrize("field", ["api_cost", "generation_cost", "evaluation_cost", "total_cost"])
def test_episode_record_rejects_negative_costs(field: str) -> None:
    data = episode_data()
    data[field] = Decimal("-0.01")

    with pytest.raises(ValidationError):
        EpisodeRecord.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("git_sha", "not-a-git-sha"),
        ("config_hash", "A" * 64),
        ("prompt_hash", "f" * 63),
        ("artifact_hashes", {"answer.py": "not-a-sha256"}),
    ],
)
def test_episode_record_rejects_invalid_hashes(field: str, value: object) -> None:
    data = episode_data()
    data[field] = value

    with pytest.raises(ValidationError):
        EpisodeRecord.model_validate(data)


def test_control_has_no_parent_and_o3_preserves_primary_and_multiple_parents() -> None:
    control = EpisodeRecord.model_validate(episode_data())
    o3_data = episode_data()
    o3_data.update(
        parent_id="parent-primary",
        parent_ids=["parent-secondary", "parent-primary"],
        operator_id="O3",
    )

    o3 = EpisodeRecord.model_validate(o3_data)

    assert control.parent_id is None
    assert control.parent_ids == []
    assert o3.parent_id == "parent-primary"
    assert o3.parent_ids == ["parent-secondary", "parent-primary"]

    o3_data["parent_id"] = "not-a-parent"
    with pytest.raises(ValidationError, match="parent_id"):
        EpisodeRecord.model_validate(o3_data)


def test_score_zero_and_unknown_scores_survive_failed_statuses_and_json() -> None:
    scored = EpisodeRecord.model_validate(episode_data())
    failed_data = episode_data()
    failed_data.update(
        episode_id="generation-failed",
        run_status="generation_failed",
        task_score=None,
        input_tokens=3,
        output_tokens=2,
        api_calls=1,
        api_cost=None,
        generation_cost=None,
        evaluation_cost=Decimal("0"),
        total_cost=None,
        usage_records=[
            usage_record(cost_basis=CostBasis.UNKNOWN, api_cost=None),
        ],
    )

    failed = EpisodeRecord.model_validate(failed_data)
    restored = EpisodeRecord.model_validate_json(failed.model_dump_json())

    assert scored.task_score == 0
    assert scored.task_score is not None
    assert failed.run_status == "generation_failed"
    assert failed.task_score is None
    assert restored.task_score is None
    assert restored.run_status == "generation_failed"


def test_usage_records_with_different_models_versions_and_prices_are_not_merged() -> None:
    records = [
        usage_record(),
        usage_record(model_version="v2", price_id="price-b", api_cost=Decimal("0.5")),
        usage_record(
            model_id="model-b",
            price_id=None,
            phase="evaluation",
            api_cost=Decimal("0.75"),
        ),
    ]
    data = episode_data()
    data.update(
        input_tokens=9,
        output_tokens=6,
        api_calls=3,
        api_cost=Decimal("1.50"),
        generation_cost=Decimal("0.75"),
        evaluation_cost=Decimal("0.75"),
        total_cost=Decimal("1.50"),
        usage_records=records,
    )

    episode = EpisodeRecord.model_validate(data)

    assert len(episode.usage_records) == 3
    assert list(episode.usage_records) == records
    assert (
        len({(item.model_id, item.model_version, item.price_id) for item in episode.usage_records})
        == 3
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_calls", 3),
        ("retry_count", 1),
        ("input_tokens", 7),
        ("output_tokens", 5),
        ("api_cost", Decimal("0.31")),
    ],
)
def test_episode_totals_must_match_usage_records(field: str, value: object) -> None:
    data = episode_data()
    data[field] = value

    with pytest.raises(ValidationError, match=field):
        EpisodeRecord.model_validate(data)


def test_unknown_usage_tokens_require_unknown_episode_totals() -> None:
    data = episode_data()
    data.update(
        input_tokens=None,
        usage_records=[
            usage_record(prompt_tokens=None, total_tokens=None),
            usage_record(phase="evaluation", api_cost=Decimal("0.05")),
        ],
    )

    episode = EpisodeRecord.model_validate(data)

    assert episode.input_tokens is None
    assert episode.output_tokens == 4

    data["input_tokens"] = 0
    with pytest.raises(ValidationError, match="input_tokens"):
        EpisodeRecord.model_validate(data)


def test_any_unknown_cost_makes_total_unknown_and_incomplete() -> None:
    data = episode_data()
    data.update(
        api_cost=None,
        generation_cost=None,
        evaluation_cost=Decimal("0"),
        total_cost=None,
        usage_records=[
            usage_record(),
            usage_record(cost_basis=CostBasis.UNKNOWN, api_cost=None),
        ],
    )

    episode = EpisodeRecord.model_validate(data)

    assert episode.total_cost is None
    assert episode.cost_complete is False
    assert episode.cost_basis is CostBasis.UNKNOWN
    assert episode.total_cost != 0

    data["total_cost"] = Decimal("0")
    with pytest.raises(ValidationError, match="total_cost"):
        EpisodeRecord.model_validate(data)


def test_complete_cost_is_summed_and_basis_is_conservative() -> None:
    data = episode_data()
    data.pop("total_cost")
    data["usage_records"] = [
        usage_record(cost_basis=CostBasis.ESTIMATED, api_cost=Decimal("0.25")),
        usage_record(phase="evaluation", api_cost=Decimal("0.05")),
    ]

    episode = EpisodeRecord.model_validate(data)

    assert episode.total_cost == Decimal("0.30")
    assert episode.cost_complete is True
    assert episode.cost_basis is CostBasis.ESTIMATED

    dumped = json.loads(episode.model_dump_json())
    assert dumped["total_cost"] == "0.30"
    assert dumped["cost_complete"] is True


def test_missing_usage_phase_prevents_cost_completeness() -> None:
    data = episode_data()
    data.update(
        total_cost=None,
        usage_records=[
            usage_record(phase=None),
            usage_record(phase="evaluation", api_cost=Decimal("0.05")),
        ],
    )

    episode = EpisodeRecord.model_validate(data)

    assert episode.cost_complete is False
    assert episode.cost_basis is CostBasis.UNKNOWN
    assert episode.total_cost is None

    data["cost_complete"] = True
    with pytest.raises(ValidationError, match="cost_complete"):
        EpisodeRecord.model_validate(data)


@pytest.mark.parametrize("phase", ["generation", "evaluation"])
def test_phase_costs_must_match_usage_records(phase: str) -> None:
    data = episode_data()
    data.update(
        generation_cost=Decimal("0.26") if phase == "generation" else Decimal("0.25"),
        evaluation_cost=Decimal("0.06") if phase == "evaluation" else Decimal("0.05"),
        total_cost=Decimal("0.31"),
    )

    with pytest.raises(ValidationError, match=f"{phase}_cost"):
        EpisodeRecord.model_validate(data)


def test_model_calls_without_usage_reject_zero_cost_completeness() -> None:
    data = episode_data()
    data.update(
        input_tokens=0,
        output_tokens=0,
        api_cost=Decimal("0"),
        generation_cost=Decimal("0"),
        evaluation_cost=Decimal("0"),
        total_cost=Decimal("0"),
        usage_records=[],
        cost_complete=True,
        cost_basis=CostBasis.ACTUAL,
    )

    with pytest.raises(ValidationError, match="usage_records"):
        EpisodeRecord.model_validate(data)


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "output_tokens", "api_cost", "generation_cost", "evaluation_cost"],
)
def test_model_calls_without_usage_require_unknown_aggregates(field: str) -> None:
    data = episode_data()
    data.update(
        input_tokens=None,
        output_tokens=None,
        api_cost=None,
        generation_cost=None,
        evaluation_cost=None,
        total_cost=None,
        usage_records=[],
    )

    episode = EpisodeRecord.model_validate(data)

    assert episode.cost_complete is False
    assert episode.cost_basis is CostBasis.UNKNOWN
    assert episode.total_cost is None

    data[field] = Decimal("0") if "cost" in field else 0
    with pytest.raises(ValidationError, match=field):
        EpisodeRecord.model_validate(data)


def test_offline_zero_cost_requires_explicit_cost_basis() -> None:
    data = episode_data()
    data.update(
        model_version=None,
        input_tokens=0,
        output_tokens=0,
        api_calls=0,
        retry_count=0,
        api_cost=Decimal("0"),
        generation_cost=Decimal("0"),
        evaluation_cost=Decimal("0"),
        total_cost=Decimal("0"),
        usage_records=[],
        cost_complete=True,
        cost_basis=CostBasis.ACTUAL,
    )

    offline = EpisodeRecord.model_validate(data)

    assert offline.cost_complete is True
    assert offline.cost_basis is CostBasis.ACTUAL
    assert offline.total_cost == Decimal("0")

    data["cost_basis"] = CostBasis.UNKNOWN
    with pytest.raises(ValidationError, match="cost_basis"):
        EpisodeRecord.model_validate(data)


def run_provenance() -> RunProvenance:
    return RunProvenance(
        git_sha=GIT_SHA,
        config_hash=CONFIG_HASH,
        git_dirty=False,
        development_mode=False,
        captured_at_utc=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )


def test_capture_provenance_hashes_config_and_accepts_clean_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_bytes(b"seed: 7\n")
    calls: list[tuple[str, ...]] = []

    def fake_git(repo_root: Path, *args: str) -> str:
        assert repo_root == tmp_path
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return GIT_SHA
        assert args == ("status", "--porcelain")
        return ""

    monkeypatch.setattr("tc_mo_me.provenance._git_output", fake_git)

    provenance = capture_provenance(config, repo_root=tmp_path)

    assert provenance.git_sha == GIT_SHA
    assert provenance.config_hash == hashlib.sha256(b"seed: 7\n").hexdigest()
    assert provenance.git_dirty is False
    assert provenance.development_mode is False
    assert provenance.captured_at_utc.utcoffset() == datetime.now(UTC).utcoffset()
    assert calls == [("rev-parse", "HEAD"), ("status", "--porcelain")]
    assert sha256_file(config) == provenance.config_hash


def test_capture_provenance_rejects_dirty_repository_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setattr(
        "tc_mo_me.provenance._git_output",
        lambda _root, *args: GIT_SHA if args == ("rev-parse", "HEAD") else " M config.yaml",
    )

    with pytest.raises(DirtyRepositoryError):
        capture_provenance(config, repo_root=tmp_path)


@pytest.mark.parametrize("options", [{"development": True}, {"allow_dirty": True}])
def test_capture_provenance_records_explicit_dirty_development_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, bool],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setattr(
        "tc_mo_me.provenance._git_output",
        lambda _root, *args: GIT_SHA if args == ("rev-parse", "HEAD") else " M config.yaml",
    )

    provenance = capture_provenance(config, repo_root=tmp_path, **options)

    assert provenance.git_dirty is True
    assert provenance.development_mode is True


def test_run_ledger_exclusively_creates_run_directory_and_appends_jsonl(
    tmp_path: Path,
) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    episode = EpisodeRecord.model_validate(episode_data())

    ledger.append(episode)

    expected_dir = tmp_path / "results" / "raw" / "run-1"
    assert ledger.run_dir == expected_dir
    assert ledger.episodes_path == expected_dir / "episodes.jsonl"
    assert ledger.read_episodes() == [episode]
    lines = ledger.episodes_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["episode_id"] == "episode-1"
    assert list(expected_dir.glob("*.tmp")) == []

    with pytest.raises(FileExistsError):
        RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())


def test_run_ledger_deduplicates_across_reopen_without_rewriting_history(
    tmp_path: Path,
) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    episode = EpisodeRecord.model_validate(episode_data())
    ledger.append(episode)
    original_bytes = ledger.episodes_path.read_bytes()
    original_manifest = ledger.manifest_path.read_bytes()

    with pytest.raises(DuplicateEpisodeError):
        ledger.append(episode)

    assert ledger.episodes_path.read_bytes() == original_bytes
    assert ledger.manifest_path.read_bytes() == original_manifest

    reopened = RunLedger(tmp_path / "results", "run-1", allow_existing=True)
    with pytest.raises(DuplicateEpisodeError):
        reopened.append_episode(episode)
    assert reopened.episodes_path.read_bytes() == original_bytes


def test_run_ledger_reopen_ignores_provenance_capture_timestamp(tmp_path: Path) -> None:
    RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    recaptured = run_provenance().model_copy(
        update={"captured_at_utc": datetime(2027, 2, 3, 4, 5, 6, tzinfo=UTC)}
    )

    reopened = RunLedger(
        tmp_path / "results",
        "run-1",
        provenance=recaptured,
        allow_existing=True,
    )

    assert reopened.provenance is not None
    assert reopened.provenance.model_dump(exclude={"captured_at_utc"}) == (
        run_provenance().model_dump(exclude={"captured_at_utc"})
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"git_sha": "f" * 40},
        {"config_hash": "f" * 64},
        {"git_dirty": True, "development_mode": True},
        {"development_mode": True},
    ],
)
def test_run_ledger_reopen_rejects_provenance_identity_mismatch(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    changed = run_provenance().model_copy(update=overrides)

    with pytest.raises(ValueError, match="provenance"):
        RunLedger(
            tmp_path / "results",
            "run-1",
            provenance=changed,
            allow_existing=True,
        )


def test_run_ledger_reopen_rejects_new_provenance_for_unprovenanced_run(
    tmp_path: Path,
) -> None:
    RunLedger(tmp_path / "results", "run-1")

    with pytest.raises(ValueError, match="provenance"):
        RunLedger(
            tmp_path / "results",
            "run-1",
            provenance=run_provenance(),
            allow_existing=True,
        )


@pytest.mark.parametrize("field", ["git_sha", "config_hash"])
def test_run_ledger_reopen_validates_existing_episodes_against_manifest(
    tmp_path: Path,
    field: str,
) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    ledger.append(EpisodeRecord.model_validate(episode_data()))
    manifest = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"][field] = "f" * 40 if field == "git_sha" else "f" * 64
    ledger.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        RunLedger(tmp_path / "results", "run-1", allow_existing=True)


def test_run_ledger_preserves_failed_records_and_zero_or_null_scores(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    failed_data = episode_data()
    failed_data.update(episode_id="failed", run_status="failed", task_score=0.0)
    generation_failed_data = episode_data()
    generation_failed_data.update(
        episode_id="generation-failed",
        run_status="generation_failed",
        task_score=None,
    )

    ledger.append(EpisodeRecord.model_validate(failed_data))
    ledger.append(EpisodeRecord.model_validate(generation_failed_data))

    restored = ledger.read_episodes()
    assert [item.run_status for item in restored] == ["failed", "generation_failed"]
    assert restored[0].task_score == 0
    assert restored[0].task_score is not None
    assert restored[1].task_score is None
    assert len(ledger.episodes_path.read_text(encoding="utf-8").splitlines()) == 2


def test_run_ledger_updates_manifest_by_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("tc_mo_me.ledger.os.replace", recording_replace)
    ledger.append(EpisodeRecord.model_validate(episode_data()))

    manifest = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-1"
    assert manifest["episode_count"] == 1
    assert manifest["episodes_file"] == "episodes.jsonl"
    assert manifest["provenance"]["git_sha"] == GIT_SHA
    assert replace_calls
    assert replace_calls[-1][0].parent == ledger.run_dir
    assert replace_calls[-1][1] == ledger.manifest_path
    assert list(ledger.run_dir.glob("*.tmp")) == []


def test_run_ledger_reopen_atomically_repairs_stale_episode_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    episode = EpisodeRecord.model_validate(episode_data())
    original_manifest = ledger.manifest_path.read_bytes()
    real_replace = os.replace

    def failing_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("injected manifest replace failure")

    monkeypatch.setattr("tc_mo_me.ledger.os.replace", failing_replace)
    with pytest.raises(OSError, match="injected"):
        ledger.append(episode)

    episodes_bytes = ledger.episodes_path.read_bytes()
    assert len(episodes_bytes.splitlines()) == 1
    assert ledger.manifest_path.read_bytes() == original_manifest
    assert list(ledger.run_dir.glob("*.tmp")) == []

    with pytest.raises(OSError, match="injected"):
        RunLedger(tmp_path / "results", "run-1", allow_existing=True)

    assert ledger.episodes_path.read_bytes() == episodes_bytes
    assert ledger.manifest_path.read_bytes() == original_manifest
    assert list(ledger.run_dir.glob("*.tmp")) == []

    replace_calls: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("tc_mo_me.ledger.os.replace", recording_replace)
    reopened = RunLedger(tmp_path / "results", "run-1", allow_existing=True)

    repaired_manifest = json.loads(reopened.manifest_path.read_text(encoding="utf-8"))
    assert repaired_manifest["episode_count"] == 1
    assert reopened.episodes_path.read_bytes() == episodes_bytes
    assert replace_calls[-1][0].parent == reopened.run_dir
    assert replace_calls[-1][1] == reopened.manifest_path
    assert list(reopened.run_dir.glob("*.tmp")) == []

    with pytest.raises(DuplicateEpisodeError):
        reopened.append(episode)
    assert reopened.episodes_path.read_bytes() == episodes_bytes


def test_run_ledger_rejects_episode_from_another_run(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    data = episode_data()
    data["run_id"] = "run-2"

    with pytest.raises(ValueError, match="run_id"):
        ledger.append(EpisodeRecord.model_validate(data))
    assert ledger.episodes_path.read_bytes() == b""


@pytest.mark.parametrize("field", ["git_sha", "config_hash"])
def test_run_ledger_rejects_episode_provenance_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    ledger = RunLedger(tmp_path / "results", "run-1", provenance=run_provenance())
    data = episode_data()
    data[field] = "f" * 40 if field == "git_sha" else "f" * 64

    with pytest.raises(ValueError, match=field):
        ledger.append(EpisodeRecord.model_validate(data))

    assert ledger.episodes_path.read_bytes() == b""
    manifest = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
    assert manifest["episode_count"] == 0


def test_trajectory_requires_strictly_increasing_steps() -> None:
    trajectory = Trajectory()
    trajectory.record_code(0, "print('ok')")
    trajectory.record_output(2, {"stdout": "done"})

    with pytest.raises(ValueError, match="increasing"):
        trajectory.record_score(2, 1.0)
    with pytest.raises(ValueError, match="increasing"):
        trajectory.record_score(1, 1.0)

    assert [event.step for event in trajectory.events] == [0, 2]


def test_trajectory_serializes_code_output_exception_retry_and_scores() -> None:
    trajectory = Trajectory()
    trajectory.record_code(1, "return answer")
    trajectory.record_output(2, {"stdout": "answer", "exit_code": 0})
    trajectory.record_exception(
        3,
        TimeoutError("slow provider"),
        retry_count=2,
        will_retry=True,
    )
    trajectory.record_score(4, 0.0, details={"reason": "incorrect"})
    trajectory.record_score(5, None, details={"reason": "not evaluated"})

    serialized = trajectory.to_jsonl()
    restored = Trajectory.from_jsonl(serialized)
    events = restored.events

    assert restored.events == trajectory.events
    assert [event.kind for event in events] == [
        EventType.CODE,
        EventType.OUTPUT,
        EventType.EXCEPTION,
        EventType.SCORE,
        EventType.SCORE,
    ]
    assert events[0].code == "return answer"
    assert events[1].output == {"stdout": "answer", "exit_code": 0}
    assert events[2].exception_type == "TimeoutError"
    assert events[2].message == "slow provider"
    assert events[2].retry_count == 2
    assert events[2].will_retry is True
    assert events[3].score == 0
    assert events[3].score is not None
    assert events[4].score is None
    assert len(serialized.splitlines()) == 5


def test_trajectory_redacts_sensitive_fields_and_common_secret_strings() -> None:
    trajectory = Trajectory()
    trajectory.record_output(
        1,
        {
            "api_key": "fake-secret-value",
            "nested": {"authorization": "Bearer fake-token-value"},
            "safe": "visible",
        },
    )
    trajectory.record_exception(2, RuntimeError("api_key=fake-secret-value"))
    trajectory.record_code(3, 'token = "fake-secret-value"')

    serialized = trajectory.to_jsonl()

    assert "fake-secret-value" not in serialized
    assert "fake-token-value" not in serialized
    assert "<redacted>" in serialized
    assert "visible" in serialized


def test_result_indexer_writes_metadata_and_hashes_but_not_trajectory(
    tmp_path: Path,
) -> None:
    trajectory = Trajectory()
    trajectory.record_code(1, "raw-code-sentinel")
    trajectory.record_output(2, "raw-output-sentinel")
    index_path = tmp_path / "results" / "index.jsonl"
    indexer = ResultIndexer(index_path)
    episode = EpisodeRecord.model_validate(episode_data())

    entry = indexer.append(episode)

    assert entry["episode_id"] == "episode-1"
    assert entry["artifact_hashes"] == {"answer.py": ARTIFACT_HASH}
    assert entry["git_sha"] == GIT_SHA
    assert entry["config_hash"] == CONFIG_HASH
    assert entry["prompt_hash"] == PROMPT_HASH
    serialized = index_path.read_text(encoding="utf-8")
    assert "raw-code-sentinel" not in serialized
    assert "raw-output-sentinel" not in serialized
    assert "trajectory" not in serialized
    assert "api_key" not in serialized
    assert "secret" not in serialized
    assert json.loads(serialized) == entry


def test_result_indexer_retains_failed_records_and_score_nullability(tmp_path: Path) -> None:
    indexer = ResultIndexer(tmp_path / "index.jsonl")
    failed_data = episode_data()
    failed_data.update(episode_id="failed", run_status="failed", task_score=0.0)
    generation_failed_data = episode_data()
    generation_failed_data.update(
        episode_id="generation-failed",
        run_status="generation_failed",
        task_score=None,
    )

    indexer.index_episode(EpisodeRecord.model_validate(failed_data))
    indexer.index_episode(EpisodeRecord.model_validate(generation_failed_data))

    entries = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["run_status"] for entry in entries] == ["failed", "generation_failed"]
    assert entries[0]["task_score"] == 0
    assert entries[0]["task_score"] is not None
    assert entries[1]["task_score"] is None
