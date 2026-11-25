# E0 基准环境选择报告

## 结论与证据边界

E1 主环境选择 **AppWorld**，ToolSandbox 保留为明确受阻时的备选。AppWorld 已完成固定源码安装、完整官方 tests、三个官方 ground-truth 任务、官方评分和轨迹核验；这些结果证明环境闭环，不是自研 Agent 成绩。ToolSandbox 仅完成官方文档和源码核查，未在本机安装或执行，以下明确区分“文档确认”和“本机实测”。

候选版本：

- AppWorld：`https://github.com/stonybrooknlp/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5`，包版本 `0.2.0.dev0`、数据 `0.2.0`。这是固定开发快照，不是正式 release；不得与 0.1.x 文档、数据或排行榜结果混用。
- ToolSandbox：`https://github.com/apple-aiml-research/ToolSandbox.git@c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`，包版本 `0.0.1`。原始 `apple/ToolSandbox` 当前重定向到 `apple-aiml-research/ToolSandbox`。

## 九个维度逐项核查

| 核查项 | AppWorld | ToolSandbox |
| --- | --- | --- |
| 1. 官方版本与代码来源 | 使用上述 StonyBrookNLP 官方完整 SHA。最新正式 tag `v0.1.3.post1` 的发布说明称 main 已显著领先；本快照具有当前单任务接口和 Windows 官方 CI 证据，因此固定快照而非浮动分支。 | 使用 Apple 官方重定向仓库的完整 SHA；无 GitHub Releases/tags 可作为更可靠版本，不能只写 `0.0.1` 而不固定 commit。 |
| 2. Python 兼容性 | 源码要求 `>=3.11`；项目实测 Python `3.13.12`。本机原生 Windows 遇到固定 rev 的内存数据库路径分隔符问题，E0 实测改用 Ubuntu WSL；CI 为 Ubuntu。 | 声明 `>=3.9`，分类器覆盖 3.9–3.12；官方安装说明主要是 Apple Silicon + Python 3.9，未找到 Windows 官方验证。备选实测拟用 Python 3.10。 |
| 3. 安装依赖 | `uv.lock` 固定全部传递依赖。兼容性实测要求 FastAPI `<0.151`、Pydantic `<2.14`、SQLModel `<0.0.45`、SQLAlchemy `<=2.0.55`；FastAPI 0.151 的显式 null 语义根因和完整复验见 [APPWORLD.md](APPWORLD.md)。Git LFS payload 必须从固定官方源码重装后再执行 `appworld install`。 | 核心安装为 `pip install .`，测试为 `pip install '.[dev]'`。依赖包含 OpenAI、Anthropic、Polars、Transformers、tree-sitter 等；官方无锁文件，启用时必须另行生成完整锁并实测旧依赖解析。 |
| 4. 任务数据获取 | 官方 `appworld download data --version 0.2.0 --root <root>` 从官方渠道下载；任务 ID 由 `load_task_ids` 获取。下载会重建目标 data 目录，因此使用独立 ignored root，不重复覆盖。 | 场景、初始状态和 milestone 直接定义在仓库 Python 源码中，无独立任务数据包；搜索工具仍访问实时 RapidAPI，本地模型权重是另一种独立下载。 |
| 5. 任务重置 | 没有 Gym 式 `reset()`；为相同 task 新建 `AppWorld` 会重新初始化，支持 `random_seed`、`save_state`/`load_state` 和 `close()`。同名 experiment 会清理旧输出，因此本项目以唯一 run/episode 名称防覆盖。 | `Scenario.play()` 每次深拷贝 `starting_context` 并设为当前上下文，等价于 rollout 重置。当前上下文是进程级单例且非线程安全，应串行或用进程隔离。 |
| 6. 官方评价接口 | `world.evaluate()`、`evaluate_task`、`evaluate_tasks` 和 CLI 可用。单任务 success 是全部测试通过；TGC 是二值成功率，SGC 需要完整 scenario 变体，不能对三任务子集宣称全数据集 SGC。 | `Scenario.play_and_evaluate`、批量 `run_sandbox` 和 CLI 可用。分数由 milestone/minefield 与轨迹匹配计算，允许部分分；不是简单任务通过率。 |
| 7. 执行轨迹 | 本机已实际获得 `logs/environment_io.md`、`logs/api_calls.jsonl` 和 `evaluation/report.md`，官方解析器也可读取。完整模型 prompt/response 由本项目另行私有记录，不上传。 | 官方输出包含 `conversation.json`、`execution_context.json`、`pretty_print.txt` 和 `result_summary.json`；文档核查确认执行上下文不序列化 InteractiveConsole 状态。 |
| 8. 本地独立复现 | **已实测。** 完整官方 tests 最终为 apps `1652 passed`、common `76 passed`、其他 `110 passed, 2 skipped`；三个 ground-truth 任务 `3/3` 通过并取得评分和轨迹。应用、数据库和评价均在本地，初次源码/数据获取需要网络。 | 核心状态环境、非搜索工具和评价可本地运行；真人 CLI 可离线但不是自动 Agent 基线。完整任务集不能原样离线复现，因为搜索工具依赖 RapidAPI，自动用户模拟器也有外部模型依赖。 |
| 9. API 需求与成本 | 环境、官方 ground-truth 和评分不需要模型 API；实测调用数与成本均为 0。自研 Agent 若使用云端模型则另计 token/费用，官方环境 seed 不保证模型确定性。 | Agent 可用本地 vLLM 或真人 CLI，但官方自动用户模拟器固定走 OpenAI；搜索场景另需 RapidAPI。启用备选前必须把 Agent、用户模拟器和外部工具成本分开记录。 |

## AppWorld 实测摘要

固定源码、数据版本、安装日志、FastAPI 根因复验和逐任务产物大小见 [APPWORLD.md](APPWORLD.md)，复现命令见 [installation_and_runs.md](installation_and_runs.md)。三个任务为 `82e2fac_1`、`82e2fac_2`、`82e2fac_3`，均为官方 solution 的环境验证；不得解释为任何模型或元演化算法的任务成功率。

## 备选启用条件

只有 AppWorld 出现经诊断的实质阻塞才启用 ToolSandbox。切换时须固定上述 commit、建立独立 Python 3.10 环境、生成锁文件，并先实测至少三个非搜索场景。自动用户角色若改为自研确定性实现，必须醒目标为自定义适配，不得冒充官方自动用户模拟器或官方 Agent 基线。
