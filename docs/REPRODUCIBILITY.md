# 可复现性

## 工具链

- Python：项目支持 >=3.11，本地默认 3.13。
- uv：本地和 CI 使用 0.10.10。
- 依赖：`uv.lock` 是唯一锁定结果；安装时使用 `--frozen`，不在安装阶段重写锁文件。

## 本地检查

```bash
uv python install 3.13
uv sync --frozen --group dev
uv run --no-sync pytest
uv run --no-sync ruff check .
uv build
```

需要 OpenAI-compatible SDK 和 AppWorld 源码依赖时：

```bash
uv sync --frozen --all-extras
```

上述同步只安装 Python 依赖，不等同于执行 `appworld install` 或下载任务数据。

## CI 范围

CI 在 `experiment/**` 分支 push 和以 `main` 为目标的 pull request 上触发：

- Ubuntu Python 3.11、3.13 与 Windows Python 3.13 的 frozen 安装、Ruff、pytest、build；
- Ubuntu Python 3.13 的独立 AppWorld 初始化与官方校验任务；
- 对完整 Git 历史运行 secret scan。

测试命令使用 pytest 默认发现机制并读取 `pyproject.toml` 的 `testpaths`，后续测试无需修改 CI 命令。CI 不上传原始 AppWorld 数据或实验产物。
