import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tc_mo_me.environments.appworld_env import AppWorldEnvironment


class FakeTask:
    instruction = "Complete the task."


class FakeTracker:
    success = True

    def to_dict(self) -> dict[str, Any]:
        return {"success": True, "num_tests": 2}

    def report(self, *, print_it: bool, colorize: bool) -> str:
        assert not print_it
        assert not colorize
        return "2/2 tests passed"


class FakeWorld:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.task = FakeTask()
        self.environment_io: list[dict[str, str]] = []
        self.close_calls = 0

    def execute(self, code: str) -> str:
        output = "execution output"
        self.environment_io.append({"number": "1", "input": code, "output": output})
        return output

    def evaluate(self) -> FakeTracker:
        return FakeTracker()

    def close(self) -> None:
        self.close_calls += 1


class WorldFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.worlds: list[FakeWorld] = []

    def __call__(self, **kwargs: Any) -> FakeWorld:
        self.calls.append(kwargs)
        world = FakeWorld(**kwargs)
        self.worlds.append(world)
        return world


def test_reset_creates_world_with_seed_and_returns_instruction(tmp_path: Path) -> None:
    factory = WorldFactory()
    environment = AppWorldEnvironment(
        root=tmp_path,
        experiment_name="run",
        seed=17,
        timeout_seconds=9,
        world_factory=factory,
    )

    observation = environment.reset("task-1")

    assert observation == {"task_id": "task-1", "instruction": "Complete the task."}
    assert factory.calls == [
        {
            "task_id": "task-1",
            "experiment_name": "run",
            "random_seed": 17,
            "timeout_seconds": 9,
        }
    ]
    assert environment.output_directory == (
        tmp_path / "experiments" / "outputs" / "run" / "tasks" / "task-1"
    )


def test_default_factory_sets_appworld_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    appworld = ModuleType("appworld")
    factory = WorldFactory()
    roots: list[str] = []
    appworld.AppWorld = factory  # type: ignore[attr-defined]
    appworld.update_root = roots.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld", appworld)

    environment = AppWorldEnvironment(root=tmp_path, experiment_name="run", seed=3)
    environment.reset("task-1")

    assert roots == [str(tmp_path)]
    assert factory.calls[0]["random_seed"] == 3


def test_execute_requires_an_active_world(tmp_path: Path) -> None:
    environment = AppWorldEnvironment(
        root=tmp_path, experiment_name="run", world_factory=WorldFactory()
    )

    with pytest.raises(RuntimeError, match="reset"):
        environment.execute("print('hello')")


def test_execute_returns_output_and_trajectory_is_a_copy(tmp_path: Path) -> None:
    environment = AppWorldEnvironment(
        root=tmp_path, experiment_name="run", world_factory=WorldFactory()
    )
    environment.reset("task-1")

    assert environment.execute("print('hello')") == "execution output"
    trajectory = environment.trajectory()
    assert trajectory == [
        {"number": "1", "input": "print('hello')", "output": "execution output"}
    ]

    trajectory[0]["output"] = "changed"
    assert environment.trajectory()[0]["output"] == "execution output"


def test_evaluate_returns_success_report_and_details(tmp_path: Path) -> None:
    environment = AppWorldEnvironment(
        root=tmp_path, experiment_name="run", world_factory=WorldFactory()
    )
    environment.reset("task-1")

    assert environment.evaluate() == {
        "success": True,
        "report": "2/2 tests passed",
        "details": {"success": True, "num_tests": 2},
    }


def test_close_is_idempotent_and_invalidates_the_world(tmp_path: Path) -> None:
    factory = WorldFactory()
    environment = AppWorldEnvironment(
        root=tmp_path, experiment_name="run", world_factory=factory
    )
    environment.reset("task-1")

    environment.close()
    environment.close()

    assert factory.worlds[0].close_calls == 1
    with pytest.raises(RuntimeError, match="reset"):
        environment.execute("print('hello')")


def test_reset_closes_the_previous_world(tmp_path: Path) -> None:
    factory = WorldFactory()
    environment = AppWorldEnvironment(
        root=tmp_path, experiment_name="run", world_factory=factory
    )
    environment.reset("task-1")

    environment.reset("task-2")

    assert factory.worlds[0].close_calls == 1
    assert factory.worlds[1].close_calls == 0


def test_default_experiment_names_are_unique(tmp_path: Path) -> None:
    first = AppWorldEnvironment(root=tmp_path, world_factory=WorldFactory())
    second = AppWorldEnvironment(root=tmp_path, world_factory=WorldFactory())

    assert first.experiment_name != second.experiment_name


def test_reset_refuses_to_overwrite_existing_task_output(tmp_path: Path) -> None:
    output_directory = (
        tmp_path / "experiments" / "outputs" / "run" / "tasks" / "task-1"
    )
    output_directory.mkdir(parents=True)
    sentinel = output_directory / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")
    factory = WorldFactory()
    environment = AppWorldEnvironment(
        root=tmp_path, experiment_name="run", world_factory=factory
    )

    with pytest.raises(FileExistsError, match="task-1"):
        environment.reset("task-1")

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert factory.calls == []


INVALID_PATH_COMPONENTS = (
    "",
    " ",
    ".",
    "..",
    "../escape",
    "nested/task",
    "nested\\task",
    "/absolute",
    "C:\\absolute",
    "C:relative",
    "\\\\server\\share\\run",
)


@pytest.mark.parametrize("experiment_name", INVALID_PATH_COMPONENTS)
def test_experiment_name_must_be_a_portable_path_component(
    tmp_path: Path, experiment_name: str
) -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        AppWorldEnvironment(
            root=tmp_path,
            experiment_name=experiment_name,
            world_factory=WorldFactory(),
        )


@pytest.mark.parametrize("task_id", INVALID_PATH_COMPONENTS)
def test_task_id_must_be_a_portable_path_component(tmp_path: Path, task_id: str) -> None:
    factory = WorldFactory()
    environment = AppWorldEnvironment(root=tmp_path, experiment_name="run", world_factory=factory)

    with pytest.raises(ValueError, match="task_id"):
        environment.reset(task_id)

    assert environment.output_directory is None
    assert factory.calls == []


def test_reset_rejects_resolved_output_outside_the_tasks_root(tmp_path: Path) -> None:
    tasks_root = tmp_path / "experiments" / "outputs" / "run" / "tasks"
    tasks_root.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    task_link = tasks_root / "task-1"
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(task_link), str(outside)],
            check=True,
            capture_output=True,
        )
    else:
        task_link.symlink_to(outside, target_is_directory=True)
    factory = WorldFactory()
    environment = AppWorldEnvironment(root=tmp_path, experiment_name="run", world_factory=factory)

    with pytest.raises(ValueError, match="within"):
        environment.reset("task-1")

    assert factory.calls == []
