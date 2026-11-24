# NEXT AGENT

## 交接边界

当前分支只承载 E0 Task 1 骨架。请先阅读 `STATUS.md` 和 `ROADMAP.md`，并核对远端 CI 的实际状态；不要把当前包测试解释为算法或 AppWorld benchmark 已通过。

## 接续事项

1. 使用 `uv sync --frozen --group dev` 复现开发环境，再运行 pytest、Ruff 和 build。
2. AppWorld 任务数据由独立环境任务处理；本任务本地未下载，原始产物不得上传或提交。固定 rev 与 Pydantic 2.14 的导入不兼容及 `<2.14` 约束原因见 `docs/APPWORLD.md`，不要未经验证放宽该上限。
3. 后续业务实现必须遵循已批准计划并补充测试，E0 的双层元演化尚未开始。
4. 新实验完成前不要向 `results/index.jsonl` 写入推测性结论。
