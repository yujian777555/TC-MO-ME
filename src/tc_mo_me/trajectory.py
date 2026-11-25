from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

__all__ = ["EventType", "Trajectory", "TrajectoryEvent"]


class EventType(StrEnum):
    CODE = "code"
    OUTPUT = "output"
    EXCEPTION = "exception"
    SCORE = "score"


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    step: int = Field(ge=0)
    kind: EventType
    code: str | None = None
    output: JsonValue = None
    exception_type: str | None = None
    message: str | None = None
    retry_count: int | None = Field(default=None, ge=0)
    will_retry: bool | None = None
    score: float | None = Field(default=None, allow_inf_nan=False)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("code", "exception_type", "message")
    @classmethod
    def redact_text(cls, value: str | None) -> str | None:
        return None if value is None else _redact_text(value)

    @field_validator("output", "details")
    @classmethod
    def redact_structured_data(cls, value: JsonValue) -> JsonValue:
        return _redact(value)

    @model_validator(mode="after")
    def validate_event_payload(self) -> Self:
        if self.kind is EventType.CODE and self.code is None:
            raise ValueError("code events require code")
        if self.kind is not EventType.CODE and self.code is not None:
            raise ValueError("only code events may contain code")
        if self.kind is EventType.EXCEPTION:
            if self.exception_type is None or self.message is None:
                raise ValueError("exception events require a type and message")
            if self.retry_count is None or self.will_retry is None:
                raise ValueError("exception events require retry metadata")
        elif any(
            value is not None
            for value in (
                self.exception_type,
                self.message,
                self.retry_count,
                self.will_retry,
            )
        ):
            raise ValueError("only exception events may contain exception metadata")
        if self.kind is not EventType.SCORE and (self.score is not None or self.details):
            raise ValueError("only score events may contain score data")
        if self.kind is not EventType.OUTPUT and self.output is not None:
            raise ValueError("only output events may contain output")
        return self


_SENSITIVE_KEYS = {
    "api_key",
    "access_key",
    "auth_token",
    "authorization",
    "client_secret",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
_SENSITIVE_KEY_PATTERN = "|".join(
    r"[_-]?".join(re.escape(part) for part in key.split("_"))
    for key in sorted(_SENSITIVE_KEYS, key=len, reverse=True)
)
_SECRET_ASSIGNMENT = re.compile(
    rf"((?<![\w-])(?:{_SENSITIVE_KEY_PATTERN})\b[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)",
    re.IGNORECASE,
)
_BEARER_TOKEN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_COMMON_API_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def _redact_text(value: str) -> str:
    value = _BEARER_TOKEN.sub("Bearer <redacted>", value)
    value = _COMMON_API_TOKEN.sub("<redacted>", value)

    def replace_assignment(match: re.Match[str]) -> str:
        prefix, recorded = match.groups()
        if recorded[:1] in {'"', "'"} and recorded[-1:] == recorded[:1]:
            return f"{prefix}{recorded[0]}<redacted>{recorded[0]}"
        return f"{prefix}<redacted>"

    return _SECRET_ASSIGNMENT.sub(replace_assignment, value)


def _redact(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        redacted: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            redacted[key] = (
                "<redacted>"
                if normalized in _SENSITIVE_KEYS or normalized.endswith("_api_key")
                else _redact(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


class Trajectory:
    def __init__(self) -> None:
        self._events: list[TrajectoryEvent] = []

    @property
    def events(self) -> tuple[TrajectoryEvent, ...]:
        return tuple(self._events)

    def append(self, event: TrajectoryEvent) -> TrajectoryEvent:
        if self._events and event.step <= self._events[-1].step:
            raise ValueError("trajectory steps must be strictly increasing")
        self._events.append(event)
        return event

    def record_code(self, step: int, code: str) -> TrajectoryEvent:
        return self.append(TrajectoryEvent(step=step, kind=EventType.CODE, code=code))

    def record_output(self, step: int, output: JsonValue) -> TrajectoryEvent:
        return self.append(TrajectoryEvent(step=step, kind=EventType.OUTPUT, output=output))

    def record_exception(
        self,
        step: int,
        error: BaseException | str,
        *,
        retry_count: int = 0,
        will_retry: bool = False,
    ) -> TrajectoryEvent:
        exception_type = type(error).__name__ if isinstance(error, BaseException) else "Exception"
        return self.append(
            TrajectoryEvent(
                step=step,
                kind=EventType.EXCEPTION,
                exception_type=exception_type,
                message=str(error),
                retry_count=retry_count,
                will_retry=will_retry,
            )
        )

    def record_score(
        self,
        step: int,
        score: float | None,
        *,
        details: dict[str, JsonValue] | None = None,
    ) -> TrajectoryEvent:
        return self.append(
            TrajectoryEvent(
                step=step,
                kind=EventType.SCORE,
                score=score,
                details={} if details is None else details,
            )
        )

    def to_jsonl(self) -> str:
        if not self._events:
            return ""
        return "\n".join(event.model_dump_json() for event in self._events) + "\n"

    @classmethod
    def from_jsonl(cls, serialized: str) -> Self:
        trajectory = cls()
        for line in serialized.splitlines():
            if line.strip():
                trajectory.append(TrajectoryEvent.model_validate_json(line))
        return trajectory
