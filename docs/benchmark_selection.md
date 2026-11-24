# 工具调用基准选择报告

## 证据边界

本报告覆盖 9 个候选。**文档声明**来自各项目官方 README、论文或仓库提示，不等同于本机验证；除 AppWorld 外，本任务没有安装或运行其他基准，也没有从排行榜转录成绩作为实测结果。AppWorld 的本机命令、三任务结果和产物检查集中记录在 [APPWORLD.md](APPWORLD.md)。

## 九项候选

| # | 候选 | 文档声明与适配判断 | 本任务实测状态 |
|---|---|---|---|
| 1 | [AppWorld](https://github.com/stonybrooknlp/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5) | 固定源码提供本地、可重置的应用世界、交互式代码执行、数据库状态评分及环境/API 日志，直接对应 runner 所需接口；选为 E0 主基准。 | 固定 SHA、数据 0.2.0、官方 tests 与前 3 个官方任务；详细结果见 APPWORLD.md。 |
| 2 | [ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox/tree/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1) | 文档提供状态依赖、用户模拟、轨迹中间/最终 milestone 评分，适合作为备选。固定 `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`；其文档环境为 Python 3.9，部分场景需要模型、用户模拟器或 RapidAPI。 | 仅用 `git ls-remote` 确认官方仓库 HEAD 等于该 SHA；未安装、未运行任务。 |
| 3 | [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | 文档覆盖可执行 function call、AST、live、multi-turn 等评测，适合函数调用专项检查；类别和评测流程与 AppWorld 的持续应用世界不同，不作为 E0 主环境。 | 仅文档调研，未安装、未运行。 |
| 4 | [τ-bench](https://github.com/sierra-research/tau-bench) | 官方 README 明确旧 airline/retail 任务不再更新并指向新仓库；E0 不固定这个旧版本。 | 仅文档调研，未安装、未运行。 |
| 5 | [τ-bench 2.0 / tau2-bench](https://github.com/sierra-research/tau2-bench) | 文档提供文本/语音、多语言、多域任务及固定任务集；Python 要求 `>=3.12,<3.14`，但 agent/user simulator、可选语音与 judge 增加外部调用和运行复杂度，保留后续专项评估。 | 仅文档调研，未安装、未运行。 |
| 6 | [ToolBench](https://github.com/OpenBMB/ToolBench) | 文档包含单/多工具场景和大量真实 REST API；推理依赖 RapidAPI 服务，官方 ToolEval 还使用模型评估，外部服务与成本不利于 E0 的离线可复现 smoke gate。 | 仅文档调研，未安装、未运行。 |
| 7 | [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | 文档提供可运行评估、73 个 API 工具及规划/检索/调用对话，适合较小规模工具使用研究；不作为本任务的持续状态应用环境。 | 仅文档调研，未安装、未运行。 |
| 8 | [WorkArena](https://github.com/ServiceNow/WorkArena) | 文档是 ServiceNow 浏览器知识工作任务，需要 gated instance 与 Playwright/BrowserGym；评测模态和外部实例要求不同于本地 function-calling 目标。 | 仅文档调研，未申请实例、未安装、未运行。 |
| 9 | [ACEBench](https://github.com/chenchen0103/ACEBench) | 文档将样本分为 Normal、Special、Agent，覆盖基础调用、歧义/不完整指令和多轮交互，适合细粒度错误分析；其数据与推理脚本形态仍需另写环境适配，不作为 E0 主执行环境。 | 仅文档调研，未下载数据、未安装、未运行。 |

## 选择结论

主线采用 AppWorld：它能在不启动真实模型 API 的情况下，用官方 ground-truth solution 验证安装、执行、评分和日志链路；这正适合 E0 环境适配，而不是宣称模型能力提升。ToolSandbox 作为备选并固定上述完整 SHA，后续若启用，需要单独处理 Python 3.9 环境、模拟器成本、外部 API 和超时隔离。

本报告不声称九项均已实测，不声称 AppWorld 全量任务通过，也不把官方 solution 的 3/3（若实测确认）解释为任何模型或元演化算法的 benchmark 成功率。
