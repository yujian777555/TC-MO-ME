# TC-MO-ME

TC-MO-ME 面向 AppWorld 工具调用基准的多目标、双层元演化研究。本仓库当前仅包含 E0 Task 1 的可复现实验基础设施；双层元演化、智能体业务模块和实验结论尚未实现或产生。

## 当前状态

- Python >=3.11，开发环境默认 Python 3.13，使用 uv 管理并锁定依赖。
- Pydantic v2 与 PyYAML 为核心依赖；OpenAI-compatible SDK 和固定提交的 AppWorld 为可选 extra。
- 当前测试只验证包可导入，不包含算法或 benchmark 正确性声明。
- `results/index.jsonl` 目前为空；没有已登记实验结果，也未下载 AppWorld 任务数据。

详见 [STATUS.md](STATUS.md)、[ROADMAP.md](ROADMAP.md) 和 [研究范围](docs/RESEARCH.md)。

## 本地开发

```bash
uv python install 3.13
uv sync --frozen --group dev
uv run --no-sync pytest
uv run --no-sync ruff check .
uv build
```

安装全部可选依赖（不会主动下载 AppWorld 任务数据）：

```bash
uv sync --frozen --all-extras
```

AppWorld 的固定版本、初始化和官方校验命令见 [docs/APPWORLD.md](docs/APPWORLD.md)。可复现命令与 CI 范围见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 目录

- `src/tc_mo_me/`：Python 包；当前只有包级占位模块。
- `tests/`：pytest 测试。
- `configs/`：后续实验配置。
- `experiments/`：后续实验入口；运行输出不得提交。
- `results/index.jsonl`：结构化实验索引；原始产物与运行输出保持本地。
- `docs/`：研究范围与复现说明。
