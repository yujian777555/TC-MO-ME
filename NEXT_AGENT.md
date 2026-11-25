# NEXT AGENT

## 接管提示词

你是 TC-MO-ME 的主执行 Agent。先阅读 `STATUS.md`、`ROADMAP.md`、`docs/APPWORLD.md`、`docs/installation_and_runs.md` 和 `docs/experiment_protocols/E1.md`，再核对 `git status`、HEAD、CI 与 PR 的实际状态。当前只完成 E0 基础设施和部分执行证据；不得把 Mock、官方 ground-truth、自动审查或打开 PR 分别冒充真实模型实验、Agent 性能、独立人工审查或验收通过。

## 当前位置

- 仓库：`https://github.com/yujian777555/TC-MO-ME`
- 分支：`experiment/e0-bootstrap`；不得直接推 main。
- main 起始 SHA：`a73bc1abcdd0d73e084b615fe9046fa38ded59a5`
- 本文更新时 SHA：`193d64b`；接管时必须以 `git rev-parse HEAD` 和 PR 最新提交为准。
- 草稿 PR：https://github.com/yujian777555/TC-MO-ME/pull/1
- 当前 `336 passed`、Ruff 通过；最新已推送绿色 CI 为提交 `561d2d7` 的 run `45566329857`，不能替代后续提交的 CI。

## 首先完成的工作

1. 查看 O3 donor-clone 修复所在工作单元是否已合并。v1 `results/raw/e1-mock-dry-run-v1/` 中 8 个 O3 `generation_failed` 是结构设计诊断，不得删除或覆盖。
2. 合并后运行新的 `e1-mock-dry-run-v2`，必须确认 112 Episodes、16 states、176 events，16 个 O3 child 均与所有父代不同；v2 使用全新 run 目录。
3. 运行完整 pytest、Ruff、build、干净 frozen 安装和 secret scan，推送实验分支并等待最新 CI。只有实际通过才更新 STATUS。
4. 更新 PR #1 和本文的最终 SHA/证据。真实 API 仍被 401 阻塞，真实 E1 不得启动。

## 环境与复现

```bash
uv python install 3.13
uv sync --frozen --all-extras --group dev --python 3.13
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv build
```

AppWorld 固定 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`、数据 `0.2.0`。Git LFS payload、兼容约束和官方 tests/三任务命令见 `docs/installation_and_runs.md`。本机真实运行使用 Ubuntu WSL；Windows/WSL 共享工作区需要 `git config --local core.autocrlf true` 对齐换行判定，不能用 `--allow-dirty` 绕过正式 provenance。

自研 Agent AppWorld Mock 的完整 WSL 命令和 v1 结果见 `docs/installation_and_runs.md`。E1 v2 合并后可用：

```bash
uv run --no-sync python -c 'from collections import Counter; from pathlib import Path; from tc_mo_me.e1_mock import run_e1_mock; root=Path.cwd(); r=run_e1_mock(root/"configs/e1/mock_dry_run.yaml", root, "results"); print(len(r.episodes), len(r.pre_child_states), len(r.events), Counter(e.run_status for e in r.episodes))'
```

运行前必须确认 Git 干净、配置中的 run_id 未被使用。每次运行后检查 manifest 的 Git SHA/config hash、112 条账本、state/event 数和成本完整性。

## 数据边界

- `results/raw/`、`.local/`、`.worktrees/`、完整 prompt/response 和私有日志不得提交。
- `results/index.jsonl` 只能包含公开元数据；不得改写既有条目。
- v1 AppWorld Mock 的干净运行 SHA 为 `4f96f12dd7b791affd23259f8e4a8b81d5a19900`；E1 v1 的干净运行 SHA 为 `3f9a3115a8a1a80e130204948d51ad062357ae71`。最终文档 SHA 与实验运行 SHA 不同是正常的。
- 原始 AppWorld 证据还保留在 `D:\TC-MO-ME\.worktrees\appworld\.local\`，不要清理该目录。

## 真实模型与 E1 门槛

现有 OpenAI 元数据请求返回 401，真实生成 0 次、生成成本 0 USD。不要重试无效凭据，不要读取或打印密钥，不要擅自选择其他付费账户。

真实 E1 前必须依次完成：有效凭据与确切模型版本、官方价格与成本探针、冻结模型配置、展示 112 Episode 总成本估算、取得项目负责人付费确认。`configs/e1/pilot.yaml` 的 `e1-pilot-model-unpinned` 不是可执行模型。Mock dry-run 不是 E1 科学样本，不能据此开展统计推断或进入 E2。

## 研究禁区

不得提前实现完整双层元演化、配对信用模型、自适应评价预算或跨任务迁移。不得根据一次 Pilot 结果反复调参追求显著，不得丢弃失败 Episode，不得把不同模型/价格版本混为一笔成本，不得宣称独立人工审查已完成。
