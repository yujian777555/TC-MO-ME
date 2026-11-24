from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "DirtyRepositoryError",
    "RunProvenance",
    "capture_provenance",
    "sha256_file",
]

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


class DirtyRepositoryError(RuntimeError):
    pass


class RunProvenance(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    git_sha: GitSha
    config_hash: Sha256
    git_dirty: bool
    development_mode: bool = False
    captured_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("captured_at_utc")
    @classmethod
    def validate_captured_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("captured_at_utc must be an aware UTC datetime")
        return value

    @model_validator(mode="after")
    def validate_dirty_mode(self) -> Self:
        if self.git_dirty and not self.development_mode:
            raise ValueError("dirty provenance requires explicit development mode")
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def capture_provenance(
    config_path: str | Path,
    repo_root: str | Path | None = None,
    *,
    development: bool = False,
    allow_dirty: bool = False,
) -> RunProvenance:
    repository = Path.cwd() if repo_root is None else Path(repo_root)
    git_sha = _git_output(repository, "rev-parse", "HEAD")
    git_dirty = bool(_git_output(repository, "status", "--porcelain"))
    development_mode = development or allow_dirty
    if git_dirty and not development_mode:
        raise DirtyRepositoryError(
            "repository has uncommitted changes; commit them or use explicit development mode"
        )
    return RunProvenance(
        git_sha=git_sha,
        config_hash=sha256_file(config_path),
        git_dirty=git_dirty,
        development_mode=development_mode,
    )
