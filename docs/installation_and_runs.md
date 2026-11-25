# AppWorld 安装与运行

## 固定环境

- Python 3.13；项目命令统一使用 `uv run --no-sync ...`。
- AppWorld 官方仓库：`https://github.com/stonybrooknlp/appworld.git`。
- 固定提交：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
- 任务数据版本：`0.2.0`。
- benchmark extra：`fastapi>=0.110,<0.151`、`sqlmodel>=0.0.19,<0.0.45`、`sqlalchemy>=2.0.14,<2.1,<=2.0.55`；当前 `uv.lock` 分别解析为 `0.150.0`、`0.0.44`、`2.0.55`。
- root：`.local/appworld`；payload 测试缓存：`.local/appworld/cache`。两者均被 Git 忽略。

## Git LFS 源码安装

直接从 Git 依赖构建时，`apps.bundle` 可能只是 LFS pointer，不能用它初始化解包。先取得包含真实 LFS payload 的固定官方源码，再从该源码覆盖安装；不提交第三方源码，也不跳过 payload 解包。

```bash
uv sync --frozen --group dev --extra benchmark --python 3.13

git lfs install
git clone https://github.com/stonybrooknlp/appworld.git .local/sources/appworld
git -C .local/sources/appworld checkout 42b5bcf3cd334fee33f0c37c02070a9f5807add5
git -C .local/sources/appworld lfs pull
git -C .local/sources/appworld rev-parse HEAD

# Linux/macOS 使用 .venv/bin/python；Windows Git Bash 使用下一行。
uv pip install --python .venv/bin/python --reinstall --no-deps .local/sources/appworld
# uv pip install --python .venv/Scripts/python.exe --reinstall --no-deps .local/sources/appworld

export APPWORLD_ROOT="$PWD/.local/appworld"
export APPWORLD_CACHE="$APPWORLD_ROOT/cache"
export PYTHONUTF8=1
export NO_PROXY=non-existent-server.com
export no_proxy=non-existent-server.com
```

固定 payload 的部分路由给 `fastapi.Path` 传递只对 Body 有意义的 `embed=True`。签名检查确认 FastAPI 0.150.0 与 0.165.0 都不接受该关键字，因此即使固定 FastAPI 后也仍需既有兼容层。该兼容层通过 lazy import hook 忽略 Path 的这个无效参数，避免在每个子进程启动时预载 FastAPI。上游 cached direct SQLite 连接还会在 remote/非 remote 测试交替时跨线程 backup，因此兼容层只给该专用缓存连接设置 `check_same_thread=False`，保留相同的 cache 容量和 PRAGMA，不改变全局 sqlite 默认行为。

这些适配不修改第三方源码、不绕过 payload 解包，也不跳过任何测试。CI 在 ignored 目录生成同一文件。`NO_PROXY=non-existent-server.com` 防止本机代理把官方“不可达服务器”测试变成空 502，让上游正常处理 DNS ConnectionError。

SQLModel 0.0.45 起会把普通 datetime 字段按 UTC 处理并拒绝 AppWorld 使用的 naive datetime；[上游 issue #229](https://github.com/StonyBrookNLP/appworld/issues/229) 建议 `<0.0.45`。benchmark extra 因此显式约束 `sqlmodel>=0.0.19,<0.0.45`，并以 `sqlalchemy>=2.0.14,<2.1,<=2.0.55` 覆盖 SQLModel 0.0.44 自身的 SQLAlchemy 下限、固定已实测版本，frozen sync 直接得到 `sqlmodel==0.0.44` 和 `sqlalchemy==2.0.55`，无需事后手动降级。

[FastAPI 0.151.0](https://github.com/fastapi/fastapi/releases/tag/0.151.0) 及提交 [`b3e04195abc78b506d5a861425111f36271651a1`](https://github.com/fastapi/fastapi/commit/b3e04195abc78b506d5a861425111f36271651a1) 改变了显式 null 的处理：非空 Body 有默认值时，旧版把 null 当缺失并应用默认，新版则交给 Pydantic 校验。AppWorld 的 Todoist `priority=Body("medium")` 不接受 null，但测试会显式传 `priority=None`，因此 benchmark extra 固定 `fastapi>=0.110,<0.151`。该变化不需要调整 Starlette；实测 frozen sync 保持 `starlette==1.10.0`。

```bash
mkdir -p .local/appworld_compat
cat > .local/appworld_compat/sitecustomize.py <<'PY'
import builtins
from functools import lru_cache, wraps


_original_import = builtins.__import__


def _patch_fastapi(module):
    if getattr(module, "_appworld_path_compat", False) or not hasattr(module, "Path"):
        return
    original_path = module.Path

    @wraps(original_path)
    def _path_without_invalid_embed(*args, **kwargs):
        kwargs.pop("embed", None)
        return original_path(*args, **kwargs)

    module.Path = _path_without_invalid_embed
    module._appworld_path_compat = True


def _patch_cached_db_connections(module):
    original = module.get_direct_cached_sqlite3_connection
    if getattr(original, "_appworld_thread_compat", False):
        return

    @lru_cache(maxsize=1000)
    @wraps(original)
    def _thread_safe_cached_connection(db_app_path):
        connection = module.sqlite3.connect(db_app_path, check_same_thread=False)
        connection.execute("PRAGMA mmap_size = 268435456")
        return connection

    _thread_safe_cached_connection._appworld_thread_compat = True
    module.get_direct_cached_sqlite3_connection = _thread_safe_cached_connection


def _import_with_appworld_compatibility(name, *args, **kwargs):
    module = _original_import(name, *args, **kwargs)
    if name == "fastapi":
        _patch_fastapi(module)
    elif name == "appworld.apps.lib.models.db":
        _patch_cached_db_connections(module)
    return module


builtins.__import__ = _import_with_appworld_compatibility
PY

export PYTHONPATH="$PWD/.local/appworld_compat${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync appworld install
```

`appworld install` 本身没有 `--root` 参数，因此必须在命令前设置 `APPWORLD_ROOT` 和 `APPWORLD_CACHE`。其余官方命令均显式传入 root 和数据版本（适用时）。

## 官方校验

```bash
uv run --no-sync appworld download data \
  --root "$APPWORLD_ROOT" --version 0.2.0

uv run --no-sync appworld verify tests \
  --root "$APPWORLD_ROOT"

uv run --no-sync appworld verify tasks \
  --root "$APPWORLD_ROOT" \
  --include-only-first-n-tasks 3 \
  --num-processes 1
```

任务校验明确为单进程，只取 train 与 dev 合并列表的前 3 个任务，不包含 test 集。CI 对官方 tests 和 tasks 步骤分别设置 20 分钟、10 分钟超时；这不是对全部 benchmark 任务的验证。

## Windows 本机运行

固定 rev 的内存数据库路径在 Windows 会把 SQLAlchemy URL 中的 `/` 与 `os.path.sep` 的 `\` 混用，直接运行会在加载首个任务时失败。本机实测因此使用 Ubuntu WSL 和独立 Python 3.13；CI 使用 Ubuntu，不需要这层适配。WSL 内同样执行上述 LFS 源码重装、payload 解包、下载和校验命令，root/cache 仍指向项目的 `.local/`。

若用 `uv pip install --target` 准备 WSL site-packages，console script 不会生成，可按官方 README 的等价形式把 `appworld ...` 换成 `python -m appworld.cli ...`，例如：

```bash
uv run --no-sync python -m appworld.cli verify tasks \
  --root "$APPWORLD_ROOT" \
  --include-only-first-n-tasks 3 \
  --num-processes 1
```

`--target` 只是本机 WSL 网络故障时准备 Linux wheel 的方式，依赖仍来自 frozen export；标准路径仍是 `uv sync --frozen`。不得用 Windows 的 `.venv` 在 WSL 中运行。

原始数据、解包代码和运行输出只保存在 `.local/`。本机日志命名见 `docs/APPWORLD.md`；不提交 raw 输出，不上传 CI artifact。

## 适配器边界

`AppWorldEnvironment` 延迟导入 AppWorld，向官方构造器传递 `random_seed` 和单次执行 `timeout_seconds`。未指定 `experiment_name` 时生成独立名称；`reset` 在构造 world 前拒绝已存在的任务输出目录，`close` 幂等，重复 reset 会先关闭旧 world。

```python
from tc_mo_me.environments.appworld_env import AppWorldEnvironment

environment = AppWorldEnvironment(root=".local/appworld", seed=0, timeout_seconds=30)
try:
    observation = environment.reset(task_id)
    output = environment.execute(code)
    evaluation = environment.evaluate()
    trajectory = environment.trajectory()
finally:
    environment.close()
```

这里的 timeout 是 AppWorld 单次执行超时。runner 后续仍需提供进程级超时与隔离，不能把适配器当作完整的进程沙箱。示例不启动模型 API；只有实际使用模型时才应记录对应 token 与成本。
