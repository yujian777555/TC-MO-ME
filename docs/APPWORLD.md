# AppWorld 实测与环境适配

## 固定来源与数据边界

- Python：`3.13.12`。
- 官方源码：`https://github.com/stonybrooknlp/appworld.git`。
- 固定提交：`42b5bcf3cd334fee33f0c37c02070a9f5807add5`。
- 数据：`0.2.0`；三个任务的 `version/data.txt` 均实际包含该版本。
- root/cache：`.local/appworld`、`.local/appworld/cache`。
- 固定源码克隆：`.local/sources/appworld`。Git LFS 实际拉取了 apps、tests、tasks、data 四个 bundle；随后用 `uv pip install --reinstall --no-deps .local/sources/appworld` 覆盖 uv Git 构建中的 LFS pointer，再执行官方 payload 解包，没有绕过 bundle。
- 原始源码、数据、日志、解包内容和实验输出全部位于 ignored 的 `.local/`，不提交第三方源码或 raw 输出。

标准安装和 CI 命令见 [installation_and_runs.md](installation_and_runs.md)。本机 Windows 直接加载会遇到固定 rev 的内存 DB 路径分隔符问题，因此本机实测在 Ubuntu WSL 的独立 Python 3.13 环境执行；由于使用 `--target` 准备该环境，本机采用官方 README 允许的 `python -m appworld.cli` 等价入口，CI 使用 `appworld` console script。

## 兼容适配

项目原有 `pydantic>=2.12,<2.14` 约束用于固定 rev 的 `GenerateSchema.match_type` 旧签名兼容。本次实测还确认并记录以下环境约束：

- `sqlmodel==0.0.44`、`sqlalchemy==2.0.55`：SQLModel 0.0.45 起拒绝 AppWorld 的 naive datetime；[上游 issue #229](https://github.com/StonyBrookNLP/appworld/issues/229) 给出了相同根因和 `<0.0.45` 建议，0.0.44 同时要求 SQLAlchemy `<2.1`。
- `.local/appworld_compat/sitecustomize.py`：lazy 忽略 `fastapi.Path` 不支持的 `embed` 参数；只给 AppWorld cached direct SQLite 连接启用其既有跨线程 backup 所需的 `check_same_thread=False`。它不预载 FastAPI、不修改全局 sqlite 默认值、不跳过测试。
- `NO_PROXY=non-existent-server.com`：防止代理把官方不可达服务器用例变成空 502，恢复预期的 DNS ConnectionError。

CI AppWorld job 生成同一份兼容文件。本地文件与 CI here-doc 已逐字比较一致。

## 官方 tests：三轮实测与剩余阻塞

每轮均运行官方 `verify tests --root "$APPWORLD_ROOT"`，没有过滤测试。最终结果以第三轮为准：

| 轮次 | apps | common | 其他模块 | 总耗时 | 原始日志（ignored） |
|---|---|---|---|---:|---|
| Windows 初轮 | 1652 errors | 74 passed, 2 skipped | 21 passed, 36 failed, 52 errors, 3 skipped | 152.527s | `.local/appworld-verify-tests.log` |
| WSL 第二轮 | 1634 passed, 18 failed | 74 passed, 2 failed | 106 passed, 4 failed, 2 skipped | 653.628s | `.local/appworld-wsl-verify-tests.log` |
| WSL 第三轮 | **1651 passed, 1 failed** | **76 passed** | **110 passed, 2 skipped** | **554.144s** | `.local/appworld-wsl-verify-tests-round3.log` |

第三轮总阶段耗时分别为 101.06s、86.17s、250.14s。两个 skipped 是上游 MCP 用例。官方命令最终仍以 1 退出，因此**不能宣称官方 tests 全通过**。

唯一剩余失败：

```text
tests/package/apps/test_todoist.py::TestTodoistApp::test_tasks_can_be_sorted_correctly
create_task 对 priority=None / due_date=None 返回 422，测试期望 200
```

该用例随后单独重跑仍失败（40.80s），证据在 `.local/appworld-targeted-todoist.log`。按最多三轮修复约束，不进行第四轮排障。CI 保留这个失败信号，不用跳过测试或 `continue-on-error` 掩盖；在上游兼容修复前，AppWorld job 会在 `verify tests` 步骤保持红色。

定向修复证据：两个 background-server 失败用例在 lazy shim 后 `2 passed`；remote-mode 与 exit-safety 顺序组在 SQLite/NO_PROXY 适配后 `2 passed`。日志分别位于 `.local/appworld-targeted-background-after.log` 和 `.local/appworld-targeted-remote-after.log`。

## 官方三任务验证

实际执行：

```bash
uv run --no-sync appworld verify tasks \
  --root "$APPWORLD_ROOT" \
  --include-only-first-n-tasks 3 \
  --num-processes 1
```

本机 WSL 使用等价的 `uv run --no-sync python -m appworld.cli ...`。原始输出 `.local/appworld-wsl-verify-tasks.log`：

```text
✅ Passed 3/3 tasks
elapsed_seconds=35.931
```

只验证 train/dev 合并顺序中的前三个 train 任务，不包含 test 集，也不是全量 benchmark。

## 适配器端到端复跑、评分与真实产物

为验证新适配器的 reset/execute/evaluate/trajectory/close，并获取不受 freezegun 影响的逐任务耗时，使用同一批官方 ground-truth solution 通过 `AppWorldEnvironment` 复跑。实验名为 `tc-mo-me-e2620e15665141728d13a7522b63b7e1`；计时采用 Linux `CLOCK_MONOTONIC`。汇总原始 JSONL 位于 `.local/appworld-adapter-verification-osclock.jsonl`，适配器复跑外部总耗时 45.614s。

下表的文件大小同时由官方 `verification` 输出核对；逐任务耗时来自适配器复跑。官方命令本身只报告上述 35.931s 总耗时。

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
- `tests/test_appworld_integration.py`：先失败再实现的 9 个适配器测试；项目合计 10 个测试。
- `configs/e0/appworld_verify.yaml`：固定 rev、数据、兼容版本、单进程三任务、seed、超时和零模型调用配置。
- `.github/workflows/ci.yml`：仅 AppWorld job 增加 LFS 固定源码重装、兼容适配、官方 tests、显式数据版本与单进程三任务；tests/tasks 分别设 20/10 分钟超时。未推送，不宣称远端 CI 已运行或通过。

九项候选的文档调研与实测边界见 [benchmark_selection.md](benchmark_selection.md)。ToolSandbox 备选固定为 `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`，仅验证官方远端 HEAD，未安装或运行。
