# 研究范围

## 固定主线

1. 任务轨迹—种群联合状态驱动的多目标算子配对信用估计与元策略演化。
2. 全生命周期成本约束的多步元生产率估计与自适应评价预算分配。
3. 跨任务元演化策略迁移目前为 **PARKED**，不启动。

这些主线是后续研究假设，不代表当前已经实现或验证。完整定义见 [research_questions.md](research_questions.md) 和 [innovation_points.md](innovation_points.md)。

## 阶段边界

- **E0**：仓库、AppWorld、最小 Agent、Genome v0.1、O1/O2/O3、统一账本、测试和 CI。当前只证明已列出的实现与运行证据，不证明算法收益。
- **E1**：三种算子在任务轨迹和种群状态下的异质性 Pilot。先冻结协议，再运行实验；Mock 设计验证不属于真实实验样本。
- **E2**：只有 E1 出现预先定义的稳定候选信号后，才开发配对信用估计模型。多步生产率和预算分配不得在 E0 提前实现。
- **跨任务迁移**：保持未启动。

官方 ground-truth 任务验证用于证明环境、评分和轨迹闭环，不能作为自研 Agent 成绩。项目测试、真实 Episode、统计分析和独立审查分别报告，任何一项都不能替代其他项。

当前字段与算子行为以 [algorithm_specification.md](algorithm_specification.md) 为准，E1 设计以 [experiment_protocols/E1.md](experiment_protocols/E1.md) 为准。
