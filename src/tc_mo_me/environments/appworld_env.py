from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from tc_mo_me.environments.base import Evaluation, Observation


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
        self.experiment_name = experiment_name or f"tc-mo-me-{uuid4().hex}"
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.output_directory: Path | None = None
        self._world_factory = world_factory
        self._world: Any | None = None

    def reset(self, task_id: str) -> Observation:
        self.close()
        self.output_directory = (
            self.root
            / "experiments"
            / "outputs"
            / self.experiment_name
            / "tasks"
            / task_id
        )
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
