from typing import Any, Protocol, TypedDict


class Observation(TypedDict):
    task_id: str
    instruction: str


class Evaluation(TypedDict):
    success: bool
    report: str
    details: dict[str, Any]


class Environment(Protocol):
    def reset(self, task_id: str) -> Observation: ...

    def execute(self, code: str) -> str: ...

    def evaluate(self) -> Evaluation: ...

    def trajectory(self) -> list[dict[str, str]]: ...

    def close(self) -> None: ...
