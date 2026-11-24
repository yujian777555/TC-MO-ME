# AppWorld

## 固定来源

`benchmark` extra 通过 `[tool.uv.sources]` 使用以下来源和完整提交：

```text
https://github.com/stonybrooknlp/appworld.git
42b5bcf3cd334fee33f0c37c02070a9f5807add5
```

安装依赖但不下载任务数据：

```bash
uv sync --frozen --extra benchmark
```

## 兼容性约束

首次 CI 在 Pydantic 2.14.4 下进入 `appworld install` 前失败，错误为 `TypeError: match_type() takes 2 positional arguments but 3 were given`。固定 rev 的 `appworld/common/datetime.py` 将 `GenerateSchema.match_type` 补丁为旧签名 `(self, obj)`，而 Pydantic 2.14 的签名已变为 `(self, obj, origin)`。

项目因此将 Pydantic 约束为 `>=2.12,<2.14`，仍使用 Pydantic v2。本地只改变该依赖后，Pydantic 2.13.5 下 `import appworld` 成功。这是针对导入兼容性的最小修正，不代表 AppWorld 任务校验已通过；完整结果以 CI 为准。

## 官方初始化与校验命令

独立环境任务准备数据时，依次执行：

```bash
git lfs install
uv run --no-sync appworld install
uv run --no-sync appworld download data
uv run --no-sync appworld verify tasks
```

CI 的 AppWorld job 保留了上述安装、数据下载和任务校验三个官方步骤，并缓存 `.local/appworld`。本 Task 1 的本地验证不执行数据下载；CI 中命令的实际结果必须以对应 run 为准。

原始 AppWorld 数据、解包内容和运行输出位于忽略目录，不配置 artifact 上传。官方文档来源：固定提交的 [README](https://github.com/stonybrooknlp/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5)。
