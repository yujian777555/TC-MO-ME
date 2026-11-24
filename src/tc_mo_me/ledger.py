from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from tc_mo_me.model import CostBasis, UsageRecord
from tc_mo_me.provenance import RunProvenance

__all__ = ["DuplicateEpisodeError", "EpisodeRecord", "ResultIndexer", "RunLedger"]

GitSha = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    ),
]
Sha256 = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    ),
]
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    run_id: NonEmptyString
    experiment_id: NonEmptyString
    episode_id: NonEmptyString
    task_id: NonEmptyString
    task_family: NonEmptyString
    parent_id: NonEmptyString | None
    child_id: NonEmptyString | None
    operator_id: NonEmptyString | None
    model_id: NonEmptyString
    model_version: NonEmptyString | None
    environment_seed: int | None
    model_seed_supported: bool
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    api_calls: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    api_cost: Decimal | None = Field(ge=0)
    wall_time: float = Field(ge=0, allow_inf_nan=False)
    task_score: float | None = Field(allow_inf_nan=False)
    run_status: NonEmptyString
    git_sha: GitSha
    config_hash: Sha256
    prompt_hash: Sha256
    parent_ids: list[NonEmptyString] = Field(default_factory=list)
    generation_cost: Decimal | None = Field(default=None, ge=0)
    evaluation_cost: Decimal | None = Field(default=None, ge=0)
    total_cost: Decimal | None = Field(default=None, ge=0)
    execution_time: float = Field(ge=0, allow_inf_nan=False)
    timestamp_utc: datetime
    artifact_hashes: dict[NonEmptyString, Sha256] = Field(default_factory=dict)
    usage_records: list[UsageRecord] = Field(default_factory=list)
    cost_complete: bool = False
    cost_basis: CostBasis = CostBasis.UNKNOWN

    @field_validator("parent_ids")
    @classmethod
    def validate_parent_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("parent_ids must not contain duplicates")
        return value

    @field_validator("timestamp_utc")
    @classmethod
    def validate_timestamp_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must be an aware UTC datetime")
        return value

    @model_validator(mode="after")
    def validate_lineage_and_cost(self) -> Self:
        if self.parent_id is not None and not self.parent_ids:
            object.__setattr__(self, "parent_ids", [self.parent_id])
        if self.parent_ids and self.parent_id not in self.parent_ids:
            raise ValueError("parent_id must identify one primary parent in parent_ids")

        complete = (
            self.generation_cost is not None
            and self.evaluation_cost is not None
            and all(
                usage.api_cost is not None and usage.cost_basis is not CostBasis.UNKNOWN
                for usage in self.usage_records
            )
        )
        expected_total = (
            self.generation_cost + self.evaluation_cost
            if complete and self.generation_cost is not None and self.evaluation_cost is not None
            else None
        )
        if self.total_cost is not None and self.total_cost != expected_total:
            raise ValueError(
                "total_cost must equal known generation and evaluation costs, or be null"
            )
        if self.cost_complete and not complete:
            raise ValueError("cost_complete cannot be true while any cost is unknown")

        expected_basis = CostBasis.UNKNOWN
        if complete:
            expected_basis = (
                CostBasis.ACTUAL
                if all(usage.cost_basis is CostBasis.ACTUAL for usage in self.usage_records)
                else CostBasis.ESTIMATED
            )
        if self.cost_basis is not CostBasis.UNKNOWN and self.cost_basis is not expected_basis:
            raise ValueError("cost_basis does not describe the completeness of recorded costs")

        object.__setattr__(self, "total_cost", expected_total)
        object.__setattr__(self, "cost_complete", complete)
        object.__setattr__(self, "cost_basis", expected_basis)
        return self


class DuplicateEpisodeError(ValueError):
    pass


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunLedger:
    def __init__(
        self,
        results_root: str | Path,
        run_id: str,
        *,
        provenance: RunProvenance | None = None,
        allow_existing: bool = False,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
            raise ValueError("run_id must be a portable single path component")
        self.run_id = run_id
        self.run_dir = Path(results_root) / "raw" / run_id
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.provenance = provenance
        self._lock = threading.Lock()

        if allow_existing:
            self._open_existing()
            return

        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir()
        self.episodes_path.open("x", encoding="utf-8").close()
        self._episode_ids: set[str] = set()
        self._manifest: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "episodes_file": self.episodes_path.name,
            "episode_count": 0,
            "provenance": (
                provenance.model_dump(mode="json") if provenance is not None else None
            ),
        }
        _atomic_write_json(self.manifest_path, self._manifest)

    def _open_existing(self) -> None:
        if not self.run_dir.is_dir():
            raise FileNotFoundError(self.run_dir)
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self._manifest.get("run_id") != self.run_id:
            raise ValueError("manifest run_id does not match the requested run")
        if self.provenance is None:
            raw_provenance = self._manifest.get("provenance")
            if raw_provenance is not None:
                self.provenance = RunProvenance.model_validate_json(
                    json.dumps(raw_provenance)
                )
        episodes = self._read_episodes_unlocked()
        self._episode_ids = {episode.episode_id for episode in episodes}
        if len(self._episode_ids) != len(episodes):
            raise ValueError("episodes.jsonl contains duplicate episode_id values")

    def _read_episodes_unlocked(self) -> list[EpisodeRecord]:
        with self.episodes_path.open("r", encoding="utf-8") as source:
            return [
                EpisodeRecord.model_validate_json(line)
                for line in source
                if line.strip()
            ]

    def read_episodes(self) -> list[EpisodeRecord]:
        with self._lock:
            return self._read_episodes_unlocked()

    def append(self, episode: EpisodeRecord) -> None:
        if episode.run_id != self.run_id:
            raise ValueError("episode run_id does not match this ledger")
        serialized = episode.model_dump_json()
        with self._lock:
            if episode.episode_id in self._episode_ids:
                raise DuplicateEpisodeError(f"duplicate episode_id: {episode.episode_id}")
            with self.episodes_path.open("a", encoding="utf-8", newline="\n") as destination:
                destination.write(serialized)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            self._episode_ids.add(episode.episode_id)
            self._manifest["episode_count"] = len(self._episode_ids)
            _atomic_write_json(self.manifest_path, self._manifest)

    def append_episode(self, episode: EpisodeRecord) -> None:
        self.append(episode)


_INDEX_FIELDS = {
    "run_id",
    "experiment_id",
    "episode_id",
    "task_id",
    "task_family",
    "parent_id",
    "parent_ids",
    "child_id",
    "operator_id",
    "model_id",
    "model_version",
    "environment_seed",
    "model_seed_supported",
    "input_tokens",
    "output_tokens",
    "api_calls",
    "retry_count",
    "api_cost",
    "wall_time",
    "task_score",
    "run_status",
    "git_sha",
    "config_hash",
    "prompt_hash",
    "generation_cost",
    "evaluation_cost",
    "total_cost",
    "execution_time",
    "timestamp_utc",
    "artifact_hashes",
    "usage_records",
    "cost_complete",
    "cost_basis",
}


class ResultIndexer:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)

    @staticmethod
    def build_entry(episode: EpisodeRecord) -> dict[str, Any]:
        return episode.model_dump(mode="json", include=_INDEX_FIELDS)

    def append(self, episode: EpisodeRecord) -> dict[str, Any]:
        entry = self.build_entry(episode)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8", newline="\n") as destination:
            json.dump(
                entry,
                destination,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        return entry

    def index_episode(self, episode: EpisodeRecord) -> dict[str, Any]:
        return self.append(episode)
