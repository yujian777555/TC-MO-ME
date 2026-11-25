from collections.abc import Callable
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from tc_mo_me.environments.base import Evaluation, Observation


def _validate_path_component(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or PureWindowsPath(value).drive
    ):
        raise ValueError(f"{field_name} must be a non-empty portable path component")
    return value


class AppWorldEnvironment:
    def __init__(
        self,
        *,
        root: str | Path,
        experiment_name: str | None = None,
        seed: int = 0,
        timeout_seconds: int | None = None,
        world_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.experiment_name = (
            f"tc-mo-me-{uuid4().hex}"
            if experiment_name is None
            else _validate_path_component(experiment_name, "experiment_name")
        )
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.output_directory: Path | None = None
        self._world_factory = world_factory
        self._world: Any | None = None

    def reset(self, task_id: str) -> Observation:
        task_id = _validate_path_component(task_id, "task_id")
        self.close()
        tasks_root = (
            self.root / "experiments" / "outputs" / self.experiment_name / "tasks"
        ).resolve()
        output_directory = (tasks_root / task_id).resolve()
        if (
            not tasks_root.is_relative_to(self.root)
            or output_directory == tasks_root
            or not output_directory.is_relative_to(tasks_root)
        ):
            raise ValueError("AppWorld output directory must remain within the tasks root")
        self.output_directory = output_directory
        if self.output_directory.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing AppWorld output: {self.output_directory}"
            )
        factory = self._world_factory
        if factory is None:
            from appworld import AppWorld, update_root

            update_root(str(self.root))
            factory = AppWorld
        self._world = factory(
            task_id=task_id,
            experiment_name=self.experiment_name,
            random_seed=self.seed,
            timeout_seconds=self.timeout_seconds,
        )
        return {"task_id": task_id, "instruction": self._world.task.instruction}

    def close(self) -> None:
        if self._world is None:
            return
        try:
            self._world.close()
        finally:
            self._world = None

    def _active_world(self) -> Any:
        if self._world is None:
            raise RuntimeError("Call reset() before using the environment.")
        return self._world

    def execute(self, code: str) -> str:
        return self._active_world().execute(code)

    def trajectory(self) -> list[dict[str, str]]:
        return deepcopy(self._active_world().environment_io)

    def evaluate(self) -> Evaluation:
        tracker = self._active_world().evaluate()
        return {
            "success": bool(tracker.success),
            "report": tracker.report(print_it=False, colorize=False),
            "details": tracker.to_dict(),
        }
