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

## 自研 Agent 的 AppWorld Mock smoke

运行 `e0-appworld-mock-smoke-v1` 使用干净提交 `4f96f12dd7b791affd23259f8e4a8b81d5a19900`、三个 AppWorld 任务和确定性 WorkflowMockAdapter。模型只生成合法 no-op，允许官方零分；它不是真实 LLM，也不是算法性能实验。

主工作区通过 Ubuntu WSL 执行：

```bash
cd /mnt/d/TC-MO-ME
git config --local core.autocrlf true  # 仅在 Windows/WSL 共享工作区对齐换行判定
export APPWORLD_ROOT="$PWD/.local/appworld"
export APPWORLD_CACHE="$APPWORLD_ROOT/cache"
export PYTHONPATH="/mnt/d/TC-MO-ME/src:/mnt/d/TC-MO-ME/.worktrees/appworld/.local/appworld_compat"
export PYTHONUTF8=1
export NO_PROXY=non-existent-server.com
export no_proxy=non-existent-server.com
.worktrees/appworld/.venv-wsl/bin/python -m tc_mo_me.cli run \
  configs/e0/appworld_mock.yaml --results-root results
```

实际结果：3 Episodes、9 次 Mock 模型调用、成本 `0 USD`。`82e2fac_1`、`82e2fac_2`、`82e2fac_3` 的官方分数均为 `0.0`，`run_status=task_failed`，失败均保留。三条记录 `cost_complete=true`，usage phase 均为 `evaluation`，Git SHA 与上述干净提交一致。

每个任务均实际存在官方 `evaluation/report.md`、`logs/environment_io.md` 和 `logs/api_calls.jsonl`。私有账本与轨迹位于 ignored 的 `results/raw/e0-appworld-mock-smoke-v1/`，AppWorld 原始环境输出位于 `.local/appworld/experiments/outputs/e0-appworld-mock-smoke-v1-episode-*/`；公开 `results/index.jsonl` 仅追加三条元数据。

## 真实模型 API 探针状态

在发现本机存在 OpenAI 凭据后，仅尝试以 `gpt-5-mini-2025-08-07` 做模型元数据查询，请求返回 401，确认凭据不可用。没有发起 chat/completion 或其他模型生成，真实模型生成次数为 0，模型生成成本为 `0 USD`。未打印、保存或提交凭据；未使用其他未知凭据重试。

因此真实 API `EXECUTED=NO`，E0 的真实模型成本测量和 E1 模型/价格冻结仍被阻塞。Mock 的显式零价格和官方 ground-truth 的零模型调用成本不能替代真实 API 成本证据。

## E1 Mock dry-run v1（结构诊断）

v1 使用干净提交 `3f9a3115a8a1a80e130204948d51ad062357ae71` 和 `configs/e1/mock_dry_run.yaml`，在主工作区执行：

```bash
uv run --no-sync python -c 'from collections import Counter; from pathlib import Path; from tc_mo_me.e1_mock import run_e1_mock; root=Path.cwd(); r=run_e1_mock(root/"configs/e1/mock_dry_run.yaml", root, "results"); print(len(r.episodes), len(r.pre_child_states), len(r.events), Counter(e.run_status for e in r.episodes))'
```

实际结果：112 Episodes、16 pre-child states、176 protocol events、340 次 Mock 调用、总成本 `0 USD`，全部 112 条记录 `cost_complete=true`。角色为 64 parent controls + 48 children；104 succeeded，8 generation_failed。

8 个失败位于 block 02/03/06/07/10/11/14/15 的 O3，均为 donor/模块映射产生父代复制而触发严格 `no_change`。这不是任务状态异质性或科学结果。v1 原始账本保留在 `results/raw/e1-mock-dry-run-v1/`，公开索引提交为 `193d64b`，不得覆盖。后续 v2 只能使用新 run_id，并需证明修复后的 O3 child 与全部父代不同。

## E1 Mock dry-run v2（结构设计验证通过）

O3 组合规则修复后，v2 使用协议 `1.1`、plan SHA-256 `7dc3253f4102ac0a1d2fe179e8cb0042865f4995f96e4a5dec6885d5dd40b1e0`、干净运行 SHA `5933f9b20449c6c6417b01dd0e0493f61c388e00` 和新 run_id `e1-mock-dry-run-v2`。执行命令与 v1 相同，但配置已更新且运行前 Git 干净。

主工作区实际结果：112/112 succeeded，64 parent controls + 48 children，16 states，176 events，364 Mock calls，6976 输入 tokens、2108 输出 tokens，全部成本完整且总成本 `0 USD`。O3 为 16/16 成功；逐条解析私有 lineage 确认 16 个 child 都使用四个父代，且 child hash 不在任何 parent hash 集合中，结构性 `no_change=0`。

v1 的 112 条记录及 8 个失败保持不变，v2 使用独立原始目录 `results/raw/e1-mock-dry-run-v2/`。公开索引现保留 v1/v2 全部元数据。v2 仍是 synthetic Mock 设计验证，不是实际 AppWorld 状态解析、真实 E1 样本或算子效果证据。

## 最终硬故障与事件语义复验

最终审查后，AgentRunner 增加真实 Windows spawn `os._exit`、模型挂起硬超时和 generation_failed+close 失败回归；E1 增加 21 组事件身份/依赖负例。修复后干净隔离环境为 `383 passed`，Ruff 与 wheel 构建通过。

真实 AppWorld 使用新 run_id `e0-appworld-mock-smoke-v2`、干净 SHA `b1f560d95009e531f43b91cb4970532ce1308117` 重跑三任务，仍为 3 Episodes/9 Mock calls、官方分数 `0/0/0`、成本 `0 USD`，成本完整且失败全部保留。原 v1 未修改。

E1 使用新 run_id `e1-mock-dry-run-v3`、干净 SHA `243880c1a4c2613a85301b165a41031d4d55bc08` 正式执行：112/112 succeeded，O3 16/16，364 Mock calls，全部成本完整且为 `0 USD`。176 个事件严格分为 64 `parent_evaluated`、16 `pre_child_state_serialized`、48 `child_generated` 和 48 `child_evaluated`；全成功运行没有 `child_finalized`。逐条 lineage 复核仍确认 16 个 O3 child 均使用四父代且不是父代复制。

`child_finalized` 仅用于未实际评价的 generation_failed 终态；v1 中的旧式 `child_evaluated` 终止包装只作 legacy 解释，原始 raw 不回写。v1/v2/v3 原始目录均保留，公开索引累计 342 条唯一元数据。上述运行仍不构成真实 E1 或算子效果证据。
