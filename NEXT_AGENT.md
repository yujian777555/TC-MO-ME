# NEXT AGENT

## 接管提示词

你是 TC-MO-ME 的主执行 Agent。先阅读 `STATUS.md`、`ROADMAP.md`、`docs/APPWORLD.md`、`docs/installation_and_runs.md` 和 `docs/experiment_protocols/E1.md`，再核对 `git status`、HEAD、CI 与 PR 的实际状态。当前只完成 E0 基础设施和部分执行证据；不得把 Mock、官方 ground-truth、自动审查或打开 PR 分别冒充真实模型实验、Agent 性能、独立人工审查或验收通过。

## 当前位置

- 仓库：`https://github.com/yujian777555/TC-MO-ME`
- 分支：`experiment/e0-bootstrap`；不得直接推 main。
- main 起始 SHA：`a73bc1abcdd0d73e084b615fe9046fa38ded59a5`
- 最终代码验收 SHA：`d12c414`；接管时必须以 `git rev-parse HEAD` 和 PR 最新提交为准。
- 草稿 PR：https://github.com/yujian777555/TC-MO-ME/pull/1
- 当前 `385 passed`、Ruff 通过；最终提交 509a7e0 的 [CI run 45613358196](https://github.com/yujian777555/TC-MO-ME/actions/runs/45613358196) 已全绿。

## 首先完成的工作

1. v3 已完成：112/112 succeeded、16 states、176 events、364 Mock calls，16 个 O3 child 均与全部父代不同，事件依赖和终态语义已严格验证。v1 的 8 个 O3 `generation_failed` 及 v1/v2/v3 原始记录均不得删除或覆盖。
2. 最终 pytest、Ruff、build、干净 frozen 安装已完成；最新 CI 以 PR 为准。如需修改代码，必须重新运行完整回归，不能沿用旧 CI。
3. 真实 API 仍被 401 阻塞，真实 E1 不得启动；下一项科学工作必须先解除该阻塞，再完成成本探针和 E1 配置冻结。

## 环境与复现

```bash
uv python install 3.13
uv sync --frozen --all-extras --group dev --python 3.13
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv build
```

AppWorld 固定 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`、数据 `0.2.0`。Git LFS payload、兼容约束和官方 tests/三任务命令见 `docs/installation_and_runs.md`。本机真实运行使用 Ubuntu WSL；Windows/WSL 共享工作区需要 `git config --local core.autocrlf true` 对齐换行判定，不能用 `--allow-dirty` 绕过正式 provenance。

自研 Agent AppWorld Mock 的完整 WSL 命令和 E1 v1/v2/v3 结果见 `docs/installation_and_runs.md`。需要重跑当前 E1 Mock 协议时可使用以下命令，但必须先换用从未存在的新 run_id，不能覆盖已验收的 v3：

```bash
uv run --no-sync python -c 'from collections import Counter; from pathlib import Path; from tc_mo_me.e1_mock import run_e1_mock; root=Path.cwd(); r=run_e1_mock(root/"configs/e1/mock_dry_run.yaml", root, "results"); print(len(r.episodes), len(r.pre_child_states), len(r.events), Counter(e.run_status for e in r.episodes))'
```

运行前必须确认 Git 干净、配置中的 run_id 未被使用。每次运行后检查 manifest 的 Git SHA/config hash、112 条账本、state/event 数和成本完整性。

## 数据边界

- `results/raw/`、`.local/`、`.worktrees/`、完整 prompt/response 和私有日志不得提交。
- `results/index.jsonl` 只能包含公开元数据；不得改写既有条目。
- AppWorld Mock v1/v2 的干净运行 SHA 分别为 `4f96f12dd7b791affd23259f8e4a8b81d5a19900`、`b1f560d95009e531f43b91cb4970532ce1308117`；E1 v1/v2/v3 分别为 `3f9a3115a8a1a80e130204948d51ad062357ae71`、`5933f9b20449c6c6417b01dd0e0493f61c388e00`、`243880c1a4c2613a85301b165a41031d4d55bc08`。最终文档 SHA 与实验运行 SHA 不同是正常的。
- 原始 AppWorld 证据还保留在 `D:\TC-MO-ME\.worktrees\appworld\.local\`，不要清理该目录。

## 真实模型与 E1 门槛

现有 OpenAI 元数据请求返回 401，真实生成 0 次、生成成本 0 USD。不要重试无效凭据，不要读取或打印密钥，不要擅自选择其他付费账户。

真实 E1 前必须依次完成：有效凭据与确切模型版本、官方价格与成本探针、冻结模型配置、展示 112 Episode 总成本估算、取得项目负责人付费确认。`configs/e1/pilot.yaml` 的 `e1-pilot-model-unpinned` 不是可执行模型。Mock dry-run 不是 E1 科学样本，不能据此开展统计推断或进入 E2。

## 研究禁区

不得提前实现完整双层元演化、配对信用模型、自适应评价预算或跨任务迁移。不得根据一次 Pilot 结果反复调参追求显著，不得丢弃失败 Episode，不得把不同模型/价格版本混为一笔成本，不得宣称独立人工审查已完成。
