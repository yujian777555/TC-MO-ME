# AppWorld 实测与环境适配

## 固定来源与数据边界

- Python：`3.13.12`。
- 官方源码：`https://github.com/stonybrooknlp/appworld.git`。
- 固定提交：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
- 数据：`0.2.0`；三个任务的 `version/data.txt` 均实际包含该版本。
- root/cache：`.local/appworld`、`.local/appworld/cache`。
- 固定源码克隆：`.local/sources/appworld`。Git LFS 实际拉取了 apps、tests、tasks、data 四个 bundle；随后用 `uv pip install --reinstall --no-deps .local/sources/appworld` 覆盖 uv Git 构建中的 LFS pointer，再执行官方 payload 解包，没有绕过 bundle。
- 原始源码、数据、日志、解包内容和实验输出全部位于 ignored 的 `.local/`，不提交第三方源码或 raw 输出。本次 FastAPI 复验复用原 worktree `D:\TC-MO-ME\.worktrees\appworld\.local` 中的 WSL 环境和数据。

标准安装和 CI 命令见 [installation_and_runs.md](installation_and_runs.md)。本机 Windows 直接加载会遇到固定 rev 的内存 DB 路径分隔符问题，因此本机实测在 Ubuntu WSL 的独立 Python 3.13 环境执行；由于使用 `--target` 准备该环境，本机采用官方 README 允许的 `python -m appworld.cli` 等价入口，CI 使用 `appworld` console script。

## 兼容适配

项目原有 `pydantic>=2.12,<2.14` 约束用于固定 rev 的 `GenerateSchema.match_type` 旧签名兼容。本次实测还确认并记录以下环境约束：

- `sqlmodel==0.0.44`、`sqlalchemy==2.0.55`：SQLModel 0.0.45 起拒绝 AppWorld 的 naive datetime；[上游 issue #229](https://github.com/StonyBrookNLP/appworld/issues/229) 给出了相同根因和 `<0.0.45` 建议，0.0.44 同时要求 SQLAlchemy `<2.1`。
- `fastapi>=0.110,<0.151`（lock 为 `0.150.0`）：[FastAPI 0.151.0](https://github.com/fastapi/fastapi/releases/tag/0.151.0) 与提交 [`b3e04195abc78b506d5a861425111f36271651a1`](https://github.com/fastapi/fastapi/commit/b3e04195abc78b506d5a861425111f36271651a1) 改变显式 null 的默认值语义，导致 Todoist `priority=None` 被交给 Pydantic 并返回 422。只将 FastAPI 栈降至 0.150.0 后，原失败用例转为通过；Starlette 无需调整，保持 `1.10.0`。
- `.local/appworld_compat/sitecustomize.py`：lazy 忽略 `fastapi.Path` 不支持的 `embed` 参数；签名检查确认 0.150.0 仍需要该处理。SQLite 处理只给 AppWorld cached direct 连接启用其既有跨线程 backup 所需的 `check_same_thread=False`。它不预载 FastAPI、不修改全局 sqlite 默认值、不跳过测试。
- `NO_PROXY=non-existent-server.com`：防止代理把官方不可达服务器用例变成空 502，恢复预期的 DNS ConnectionError。

CI AppWorld job 生成同一份兼容文件。本地文件与 CI here-doc 已逐字比较一致。

## FastAPI 根因复验与完整官方 tests

此前 FastAPI 0.165.0 下，Todoist 排序用例因显式 `priority=None` 返回 422。此次只将 FastAPI 从 0.165.0 切换为 **0.150.0**，Starlette 保持 `1.10.0`，SQLModel/SQLAlchemy 和既有兼容 shim 均未改变。原用例随后 **1 passed**，pytest 内部耗时 36.19s、外部耗时 65.000s，原始日志为 `.local/appworld-fastapi-0.150-todoist.log`。这验证了 0.151.0 显式 null 默认值语义变化的单一假设。

`pyproject.toml` 的 benchmark extra 已固定 `fastapi>=0.110,<0.151`，`uv lock` 只将 FastAPI 更新为 0.150.0。新 worktree 的干净 `.venv` 经 `uv sync --frozen --group dev --extra benchmark --python 3.13` 安装后，实测 `fastapi==0.150.0`、`starlette==1.10.0`、`pydantic==2.13.5`，且 `uv lock --check` 通过。签名检查确认 0.150.0 的 `fastapi.Path` 仍不接受 `embed`，因此保留既有 Path shim，不添加新兼容补丁。

在同一 WSL 环境完整执行官方 `verify tests`，最终 **exit 0**：

| 阶段 | 结果 | 耗时 |
|---|---|---:|
| apps | **1652 passed** | 102.31s |
| common | **76 passed** | 85.14s |
| 其他模块 | **110 passed, 2 skipped** | 420.58s |
| 官方命令总计 | 全部通过 | **736.729s** |

两个 skipped 是上游既有 MCP 用例，不是为本次修复新增的过滤。原始日志为 `.local/appworld-fastapi-0.150-verify-tests-official.log`。这次完整通过取代此前 `1651 passed, 1 failed` 的 Todoist 阻塞记录；未推送，不宣称远端 CI 已运行。

## 官方三任务验证

实际执行：

```bash
uv run --no-sync appworld verify tasks \
  --root "$APPWORLD_ROOT" \
  --include-only-first-n-tasks 3 \
  --num-processes 1
```

本机 WSL 使用等价的 `uv run --no-sync python -m appworld.cli ...`。FastAPI 0.150.0 环境下的原始输出 `.local/appworld-fastapi-0.150-verify-tasks.log`：

```text
✅ Passed 3/3 tasks
elapsed_seconds=51.330
```

只验证 train/dev 合并顺序中的前三个 train 任务，不包含 test 集，也不是全量 benchmark。

## 适配器端到端复跑、评分与真实产物

为验证新适配器的 reset/execute/evaluate/trajectory/close，并获取不受 freezegun 影响的逐任务耗时，使用同一批官方 ground-truth solution 通过 `AppWorldEnvironment` 复跑。实验名为 `tc-mo-me-e2620e15665141728d13a7522b63b7e1`；计时采用 Linux `CLOCK_MONOTONIC`。汇总原始 JSONL 位于 `.local/appworld-adapter-verification-osclock.jsonl`，适配器复跑外部总耗时 45.614s。

下表的文件大小由本次官方 `verification` 输出重新核对；逐任务耗时来自此前适配器复跑。官方命令本身只报告上述 51.330s 总耗时。

| Task ID | success | 评分报告 | `evaluation/report.md` | `logs/environment_io.md` | `logs/api_calls.jsonl` | 适配器耗时 |
|---|---|---|---:|---:|---:|---:|
| `82e2fac_1` | true | 2 passed / 0 failed / 2 total | 存在，821 bytes | 存在，1924 bytes | 存在，6282 bytes | 16.843s |
| `82e2fac_2` | true | 2 passed / 0 failed / 2 total | 存在，821 bytes | 存在，1850 bytes | 存在，4023 bytes | 2.240s |
| `82e2fac_3` | true | 2 passed / 0 failed / 2 total | 存在，821 bytes | 存在，1903 bytes | 存在，3816 bytes | 2.024s |

官方产物根目录为 `.local/appworld/experiments/outputs/verification/tasks/`；适配器产物根目录为 `.local/appworld/experiments/outputs/tc-mo-me-e2620e15665141728d13a7522b63b7e1/tasks/`。三条 trajectory 长度均为 1，因为每题由一次官方 solution 执行完成。

这些 success 是**官方 ground-truth solution 的安装/执行验证**，不是模型或 TC-MO-ME 策略成绩。全程未启动真实模型 API：模型 API 调用数为 0，模型 API 成本为 **0 USD**。这不代表未来真实模型调用的未知成本可以写成 0。

## 代码与 CI 范围

- `src/tc_mo_me/environments/base.py`：Environment、Observation、Evaluation 接口类型。
- `src/tc_mo_me/environments/appworld_env.py`：延迟导入、seed/timeout 透传、唯一默认 experiment_name、reset 前关闭旧 world、同名任务输出拒绝覆盖、执行/评分/轨迹和幂等 close。
- `tests/test_appworld_integration.py`：覆盖适配器生命周期、非法路径和输出防逃逸；完整项目测试数以 `STATUS.md` 的最终验收记录为准。
- `pyproject.toml`、`uv.lock`：benchmark extra 固定 FastAPI、SQLModel 和 SQLAlchemy 兼容范围，frozen sync 直接得到已实测版本，不再事后降级。
- `configs/e0/appworld_verify.yaml`：固定 rev、数据、FastAPI 与其他兼容版本、单进程三任务、seed、超时和零模型调用配置。
- `.github/workflows/ci.yml`：AppWorld job 包含 LFS 固定源码重装、兼容适配、完整官方 tests、显式数据版本与单进程三任务；tests/tasks 分别设 20/10 分钟超时。
- 最终代码验收提交 `9b8b27f0b086115cdb10d34ef00f089505d0233a` 的 [run 45571352088](https://github.com/yujian777555/TC-MO-ME/actions/runs/45571352088) 全部通过，包括 AppWorld、三平台项目测试、构建和全历史 secret scan。后续代码变更仍需重新验收。

两个候选的九项核查维度与实测边界见 [benchmark_selection.md](benchmark_selection.md)。ToolSandbox 备选固定为 `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`，仅验证官方远端 HEAD，未安装或运行。
