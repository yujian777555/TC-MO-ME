# ROADMAP

## E0：环境与基线建设（收尾中）

### 已实现并测试

- Python/uv 项目、完整锁文件、GitHub Actions、全历史 secret scan 和公开结果索引。
- Genome v0.1、canonical hash、parent-child lineage、统一 ModelAdapter、确定性 Mock。
- O1 局部策略修复、O2 结构性变异、O3 多父代重组及修改/成本审计。
- append-only Episode 账本、usage phase、失败保留、run provenance 和轨迹脱敏。
- 最小 AgentRunner、进程级超时、Mock 环境及 AppWorld 环境适配。

### 已执行

- AppWorld 固定提交 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`、数据 `0.2.0` 的安装与完整官方 tests：全部通过，两个上游 MCP 用例保持跳过。
- 官方 ground-truth 三任务 `3/3` 通过，评分报告与轨迹存在；不是自研 Agent 成绩。
- 自研 Agent Mock 在真实 AppWorld 上执行 3 Episodes/9 Mock calls，三个官方分数均为 0，失败完整保留。
- GitHub Actions [run 45561814592](https://github.com/yujian777555/TC-MO-ME/actions/runs/45561814592) 全部通过；后续最终提交仍需重新运行 CI。

### 尚未完成或受阻

- E1 的 112 次 Mock 设计验证正在实现与调试，不计为真实 E1。
- 真实 OpenAI 元数据请求返回 401，真实模型生成和成本测量未执行；E1 模型与价格配置因此未冻结。
- E0 只有在剩余 Mock 验证、最终回归、CI 和交接文档完成后，才能按实际证据判定哪些验收项通过；真实 API 阻塞不得被忽略。

## E1：算子异质性 Pilot（协议已冻结，真实实验未启动）

- 8 个 train 任务、2 个环境 seed、4 个初始 Genome、O1/O2/O3 和原始父代对照，共 112 Episode。
- 主父代、O3 donor、组件来源、事件门控、状态特征和统计方案已机器校验；dev 保留独立验证。
- 下一步先完成 112 次 Mock 设计验证，再在有效真实模型/价格与付费授权具备后冻结可执行配置。
- 真实 E1 完成后才进行逐任务/族、效应量、置信区间、多重校正、成本和预测—实测分析。Pilot 不是最终确认性样本。

## E2：配对信用与成本模型（未启动）

只有出现预注册的稳定候选信号后，才开发任务轨迹—种群状态下的配对信用估计；随后研究全生命周期成本的多步元生产率和自适应评价预算。不得在 E0/E1 提前实现完整双层元演化。

## 跨任务迁移（PARKED）

第三创新点保持未启动，不开展实现、实验或结论性描述。
