# STATUS

## 当前阶段

E0 Task 1（仓库、依赖与 CI 骨架）已在实验分支 `experiment/e0-bootstrap` 上实现。基线为 `main` 的 `a73bc1abcdd0d73e084b615fe9046fa38ded59a5`。

## 已确认

- 包可导入性测试先因模块缺失失败，加入最小 `src/tc_mo_me/__init__.py` 后通过。
- uv 0.10.10 与 Python 3.13.12 下，`uv lock` 成功解析 91 个包，`uv sync --frozen --all-extras` 成功安装全部依赖。
- AppWorld 0.2.0.dev0 从固定提交 `42b5bcf3cd334fee33f0c37c02070a9f5807add5` 构建并安装；本地未执行 AppWorld payload 安装或任务数据下载。
- 首次远端 CI 的三个质量矩阵 job 与全历史 secret scan 通过；AppWorld job 在 Pydantic 2.14.4 的导入兼容性问题处失败。已按 `docs/APPWORLD.md` 记录的根因将 Pydantic 限定为 `>=2.12,<2.14`，并在 2.13.5 下验证 `import appworld` 成功。
- 本地 pytest 为 1 passed，Ruff 为 All checks passed，`uv build` 成功生成 sdist 和 wheel。
- 当前没有双层元演化、智能体业务模块、AppWorld 任务数据或实验结果。

## 后续

远端 CI 状态以实验分支的 GitHub Actions run 为准。后续任务不得把包级 smoke test 解释为算法或 benchmark 已通过，也不在本任务提前实现 Task 2+。
