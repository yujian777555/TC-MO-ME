# TC-MO-ME

Task-Conditioned Multi-Objective Meta-Evolution（TC-MO-ME）面向 LLM Agent 递归自我改进，研究任务条件化的多目标元演化优化。本仓库当前处于 **E0 环境与基线建设**，不包含双层元演化、配对信用模型、自适应评价预算或跨任务迁移。

## 当前状态

- 已实现 Genome v0.1、canonical hash、parent-child lineage、统一 ModelAdapter、确定性 Mock、O1/O2/O3、append-only Episode 账本、轨迹记录和最小 AgentRunner。
- AppWorld 固定到官方源码提交 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`、数据版本 `0.2.0`；官方 ground-truth 三任务已取得 `Passed 3/3`，完整官方 tests 也已通过。该结果不是自研 Agent 或模型成绩。
- 代码验收提交 `9b8b27f` 的 [GitHub Actions run 45571352088](https://github.com/yujian777555/TC-MO-ME/actions/runs/45571352088) 已全部通过，包括 AppWorld、三平台项目测试、构建和 secret scan。后续变更以 `STATUS.md` 和最新 CI 为准。
- 真实模型 API 尚未执行成功：现有 OpenAI 凭据的元数据请求返回 401，未发起模型生成，模型调用成本为 `0 USD`。这不能被解释为真实 API 验收通过。
- E1 已冻结任务、seed、父代、donor、事件顺序和统计协议；v2 Mock 设计验证为 112/112 succeeded，O3 16/16 且均不同于全部父代。真实模型仍未冻结，真实 E1 尚未运行，也没有任何算子优于其他算子的科学结论。

精确状态与证据见 [STATUS.md](STATUS.md)、[ROADMAP.md](ROADMAP.md)、[NEXT_AGENT.md](NEXT_AGENT.md) 和 [AppWorld 实测记录](docs/APPWORLD.md)。

## 本地开发

项目支持 Python `>=3.11`，本地与 AppWorld 验证使用 Python 3.13 和 uv：

```bash
uv python install 3.13
uv sync --frozen --group dev
uv run --no-sync pytest
uv run --no-sync ruff check .
uv build
```

安装 OpenAI-compatible SDK 和固定 AppWorld 源码依赖：

```bash
uv sync --frozen --all-extras
```

依赖同步不等同于完整 AppWorld 安装。Git LFS payload、任务数据、官方三任务验证及兼容约束见 [安装与运行记录](docs/installation_and_runs.md)。原始数据、完整轨迹、私有日志和凭据不得提交。

## 目录

- `src/tc_mo_me/`：Genome、模型、算子、环境、账本、轨迹与实验执行代码。
- `tests/`：离线单元测试和可替换环境集成测试。
- `configs/e0/`：E0 环境、Mock 与真实 API smoke 配置。
- `configs/e1/`：四个初始 Genome 与 E1 Pilot 协议配置。
- `docs/`：研究问题、创新点、算法规范、基准选择和实验协议。
- `experiments/`：可公开实验入口；私有输出不得提交。
- `results/index.jsonl`：只含可公开元数据的实验索引；`results/raw/` 保持本地并被 Git 忽略。
