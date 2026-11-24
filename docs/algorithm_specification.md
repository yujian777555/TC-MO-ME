# E0 算法与数据规范（genome schema v0.1）

## 1. 范围与状态口径

本文以当前 workspace 中的 `src/tc_mo_me/genome.py`、`src/tc_mo_me/model.py`、`src/tc_mo_me/ledger.py` 为实现事实来源，并补充 `src/tc_mo_me/trajectory.py`、`src/tc_mo_me/provenance.py` 的审计行为。

文档严格区分：

- **已实现**：当前源码和测试直接支持的行为；
- **E0 约定**：后续算子与实验必须遵守的数据和研究约束，但当前没有对应执行器；
- **待实现**：RQ1/RQ2 所需的元策略、信用分配和预算调度，仅在另行批准后开发。

E0 只建设基础设施。E1 只允许算子异质性 pilot。本文不声明 O1/O2/O3 已有可执行实现，不声明 RQ1/RQ2 已验证；RQ3 跨任务迁移为 PARKED，不启动。项目不实现双层元演化。

后续源码变更时，主 Agent 应同步更新本文的字段、约束与实现状态，不能沿用本版描述掩盖行为变化。

## 2. 模块职责

| 模块 | 当前已实现职责 | 不承担的研究功能 |
| --- | --- | --- |
| `src/tc_mo_me/genome.py` | genome/schema 校验、workflow graph 结构校验、canonical bytes、内容哈希、lineage 记录 | 自动变异、重组、算子选择、信用学习 |
| `src/tc_mo_me/model.py` | 请求/响应模型、prompt hash、模型适配器、确定性 mock、重试、用量聚合与成本依据 | 任务策略、元生产率计算、自适应评价调度 |
| `src/tc_mo_me/ledger.py` | episode 校验、append-only JSONL、run manifest、结果索引 | 多进程协调、实验统计、谱系收益分摊 |
| `src/tc_mo_me/trajectory.py` | 严格递增事件流、JSONL 往返、敏感字段和常见密钥形式脱敏 | 轨迹特征提取或因果信用判断 |
| `src/tc_mo_me/provenance.py` | Git/config 来源记录、文件 SHA-256、脏工作树保护 | 证明实验结果有效或完整 |

## 3. Genome v0.1

### 3.1 顶层字段

`Genome` 使用 Pydantic strict mode、禁止额外字段并在赋值时重新校验。`Metadata` 是 `dict[str, JsonValue]`；其键和 JSON 值可携带策略细节，但不能借此增加未声明的顶层字段。

| 字段 | 类型/默认值 | 约束与含义 |
| --- | --- | --- |
| `schema_version` | 字面量 `"0.1"`，默认 `"0.1"` | 其他版本被拒绝 |
| `system_prompt` | `str` | 非空；系统提示内容 |
| `planning_policy` | `Metadata` | 必填；规划策略参数，当前不执行其语义 |
| `code_generation_policy` | `Metadata` | 必填；代码生成策略参数，当前不执行其语义 |
| `workflow_graph` | `WorkflowGraph` | 必填；声明式工作流结构 |
| `verification_policy` | `Metadata` | 必填；验证策略参数，当前不执行其语义 |
| `identity` | `Metadata`，默认 `{}` | 身份与展示元数据；不进入 genome hash |
| `lineage` | `Metadata`，默认 `{}` | 基因组内嵌谱系元数据；不进入 genome hash；不是 `LineageRecord` 的替代存储 |

`identity` 可以包含 `genome_id`，但 v0.1 模型不强制该键存在，也不强制全局唯一。实验若使用 `genome_id` 作为主键，必须在实验层另行校验，不能把约定误当作当前模型保证。

### 3.2 WorkflowGraph

节点类型只允许 `planner`、`generator`、`executor`、`verifier`。字段如下：

| 对象 | 字段 | 已实现约束 |
| --- | --- | --- |
| `WorkflowNode` | `id` | 非空字符串；图内唯一 |
|  | `type` | 上述四种 `WorkflowNodeType` |
|  | `config` | JSON 元数据字典，默认 `{}` |
| `WorkflowEdge` | `source`, `target` | 均为非空字符串且必须引用已声明节点 |
| `WorkflowGraph` | `entry_id` | 非空且必须引用已声明节点 |
|  | `exit_ids` | 至少一个、不得重复、均须引用已声明节点 |
|  | `max_nodes` | 默认及最大值为 64，最小为 1；实际节点数不得超过它 |
|  | `nodes` | 至少一个节点 |
|  | `edges` | 默认空列表；有序 `(source,target)` 不得重复 |

所有节点必须能从 `entry_id` 到达，并且每个节点必须能沿有向边到达至少一个 `exit_id`。只要满足上述条件，环是允许的；因此不能把工作流描述为保证终止的 DAG，也不能声称存在自动步数限制。

列表顺序在校验和 canonical JSON 中保留。节点、边或 `exit_ids` 的顺序变化可能改变内容哈希，即使某种外部图语义认为它们等价。

## 4. Canonical genome bytes 与内容哈希

### 4.1 已实现算法

`validate_genome` 接受 `Genome`、mapping、JSON 字符串或 UTF-8 JSON bytes，并重新执行模型校验。已有 `Genome` 也通过 JSON round-trip 再校验。

`canonical_genome_bytes` 的步骤为：

1. 校验并规范化为 `Genome`；
2. 仅保留 `schema_version` 和五个内容组件：`system_prompt`、`planning_policy`、`code_generation_policy`、`workflow_graph`、`verification_policy`；
3. 以 JSON mode 导出，递归排序对象键，保留数组顺序；
4. 使用 UTF-8、`ensure_ascii=False`、分隔符 `(",", ":")`，并禁止 NaN/Infinity；
5. 输出上述紧凑 JSON 的 UTF-8 bytes。

`genome_hash` 对这些 bytes 计算 SHA-256，输出 64 位小写十六进制字符串。内容相同而 `identity` 或 `lineage` 不同的 genome 具有相同 hash；任一被纳入内容发生变化，通常会产生不同 hash。

### 4.2 解释边界

- hash 标识的是 **v0.1 内容投影**，不是 `identity`、创建时间、任务结果或完整原始文件的字节哈希。
- 不能把 hash 相同解释为两个智能体在所有环境中行为完全一致；模型版本、环境、随机种子和未纳入 genome 的运行配置仍可能不同。
- `canonical_genome_bytes` 不是签名校验，不能单独证明文件未被恶意篡改。
- `model.py` 的 `prompt_hash` 是另一项哈希：仅包含 `prompt` 与 `system_prompt` 的规范 JSON，不包含模型 ID、seed、temperature、token 上限或重试配置。

## 5. LineageRecord

### 5.1 字段

| 字段 | 类型与已实现约束 |
| --- | --- |
| `operator_id` | 非空字符串；当前不是 O1/O2/O3 的枚举 |
| `parent_ids` | 至少一个非空 ID，且不得重复 |
| `primary_parent_id` | 非空且必须包含于 `parent_ids` |
| `child_id` | 非空且不得等于任一父代 ID |
| `modified_paths` | 至少一个、不得重复；每项首段必须是五个内容组件之一 |
| `component_sources` | 键必须恰好覆盖五个内容组件；值只能引用 `parent_ids` |
| `parent_hashes` | 键必须恰好等于 `parent_ids`；值为 64 位小写 SHA-256 hex |
| `child_hash` | 64 位小写 SHA-256 hex |
| `generation` | 整数，至少为 1 |
| `usage` | 复用 `src/tc_mo_me/model.py` 的 `UsageRecord` |
| `created_at_utc` | 必须带时区且 UTC offset 为零 |

`modified_paths` 的校验只检查首段所属组件，例如可指向 `workflow_graph` 下的路径；当前不解析或验证更深层路径是否实际存在。组件来源要求覆盖全部五个组件，但未修改组件也必须指定其来源父代。

### 5.2 当前校验不自动证明的事项

当前模型不会读取父代/子代 genome 来重算并比对 `parent_hashes`、`child_hash`，不会核对实际内容差异是否等于 `modified_paths`，也不会检查 `generation` 是否恰为父代最大世代加一。这些关系是 E0 写入方责任，后续执行器应在生成时验证。

`Genome.lineage`、`LineageRecord` 与 `EpisodeRecord.parent_ids` 是不同存储接口。当前代码不会自动在它们之间同步；实验写入流程必须显式保持一致，不能声称已有自动谱系服务。

## 6. O1/O2/O3 的 E0 约定

### 6.1 实现状态与共同约束

当前没有 O1/O2/O3 执行器，`operator_id` 的自由字符串类型本身也不实施下列算子语义。以下内容是未来写入和实现的约束，不是已完成能力：

1. 每次生成必须记录算子 ID、全部父代、主父代、子代、规范内容 hash、修改路径、组件来源、生成用量和 UTC 时间。
2. 子代必须通过 genome v0.1 校验；失败生成不得以伪造的空 lineage 冒充成功子代，但其失败成本仍应进入 episode/实验成本记录。
3. 内容未变化的重复子代必须被标记并由实验协议说明去重方式，不能把去重后的零成本或零改进当作算法收益。
4. O1/O2/O3 的父代数量、修改路径和来源必须与实际变换一致；当前通用模型只提供第 5 节列出的结构校验。
5. E1 只允许用这些类别进行异质性 pilot 和记录流程检查，不据此证明元策略或搜索效果。

### 6.2 分类约束

| 类别 | E0 研究约定 | 当前代码状态 |
| --- | --- | --- |
| O1 | 单父代内容变异：`parent_ids` 恰有一个元素，`primary_parent_id` 等于该父代；全部组件来源指向该父代；`modified_paths` 非空且仅描述实际修改 | 未实现专用执行器或 O1 数量/差异校验 |
| O2 | 单父代策略或工作流变换：父代约束同 O1；若修改 `workflow_graph`，子图仍须满足入口、出口、可达性和节点上限；其他内容组件不得被隐式更换来源 | 未实现专用执行器，也未实现自动工作流变换 |
| O3 | 多父代组件组合：`parent_ids` 至少两个不同父代，主父代必须属于父代集合；每个内容组件显式记录实际来源，至少两个父代应对子代组件有实际贡献 | 通用 lineage 支持记录多父代与主父代，但未实现 O3 执行器、最少父代数或实际贡献校验 |

O1 与 O2 的区别是 E0 的变换范围约定，不代表二者已具有固定算法参数或已证明互补。O3 的“实际贡献”不能仅凭 `parent_ids` 推断；未来执行器需用组件差异和来源记录验证。

通用 `LineageRecord` 目前允许单父代记录使用 `operator_id="O3"`，也允许多父代记录只有一个实际来源。这类值能通过通用模型并不代表满足本节算子约定；专用校验属于待实现工作。

## 7. 模型请求、响应与用量

### 7.1 ModelRequest 与 ModelResponse

`ModelRequest` 字段为 `model_id`、 `prompt`、`system_prompt`、`seed`、`temperature`、`max_tokens`、`timeout_seconds`、`max_retries`、`price_id`。模型 ID 非空；temperature 非负；token 上限和 timeout 为正；重试数非负。若适配器未声明支持 seed，请求提供 seed 会在调用前抛出 `UnsupportedSeedError`。

`ModelResponse` 包含 `content`、模型 ID/版本、64 位小写 `prompt_hash`、`model_seed_supported`、`usage` 和可空 `finish_reason`。`text`、`output_text` 是 `content` 的只读别名。

`ModelAdapter.generate` 统一处理重试和用量中的 `calls=1`、`retries`、`elapsed_seconds`。Mock 适配器是确定性的，使用空格分词构造测试用量并将计时固定为零；它不能作为真实模型效果证据。

### 7.2 UsageRecord 与成本依据

| 字段组 | 字段 |
| --- | --- |
| 模型/价格 | `model_id`、可空 `model_version`、可空 `price_id` |
| token | 可空且非负的 `prompt_tokens`、`completion_tokens`、`total_tokens` |
| 调用 | 非负 `calls`（默认 1）、`retries`（默认 0）、`elapsed_seconds`（默认 0） |
| 成本 | 可空且非负的 `Decimal api_cost`，以及 `cost_basis` |

`CostBasis` 的规范序列化值为：

- `unknown`：成本未知，此时 `api_cost` 必须为 null；
- `estimated`：由 token 与配置价格估算，此时必须有 `api_cost`；别名 `TOKEN_ESTIMATE` 对应同值；
- `actual`：由提供方返回实际成本，此时必须有 `api_cost`；别名 `API_RETURNED` 对应同值。

若输入和输出 token 已知且适配器配置了两种每百万 token 价格，且提供方未返回成本，适配器可计算估算成本。缺少任一必要信息时成本保持未知，不得写成零。

`group_usage`/`aggregate_usage` 按 `(model_id, model_version, price_id, cost_basis)` 分组，避免合并不同模型、版本、价格或成本依据。任一记录的某个 token 字段未知，该组对应聚合 token 字段即为未知；调用、重试和耗时求和；已知成本在组内以 `Decimal` 求和，未知成本组保持 null。

## 8. EpisodeRecord 与成本完整性

### 8.1 字段分组

`EpisodeRecord` 是 frozen、strict、禁止额外字段的模型。字段按职责分组如下：

| 分组 | 字段 |
| --- | --- |
| 运行与任务 | `run_id`、`experiment_id`、`episode_id`、`task_id`、`task_family` |
| 谱系与算子 | 可空 `parent_id`、可空 `child_id`、可空 `operator_id`、默认空列表 `parent_ids` |
| 模型与环境 | `model_id`、可空 `model_version`、可空 `environment_seed`、`model_seed_supported` |
| 汇总用量 | 可空非负 `input_tokens`、`output_tokens`、`api_cost`；非负 `api_calls`、`retry_count` |
| 时间与结果 | 非负有限 `wall_time`、`execution_time`；可空有限 `task_score`；非空 `run_status` |
| 来源 | 40 位小写 `git_sha`、64 位小写 `config_hash` 与 `prompt_hash` |
| 生命周期成本 | 可空非负 `generation_cost`、`evaluation_cost`、`total_cost` |
| 审计 | 零偏移 UTC `timestamp_utc`、`artifact_hashes: dict[str, sha256]`、默认空列表 `usage_records` |
| 派生成本状态 | 默认 `cost_complete=False`、`cost_basis=unknown` |

`task_score=0` 与 `task_score=None` 必须区分；失败状态和生成失败状态也可被记录。当前 `run_status` 只要求非空字符串，并没有封闭状态枚举，实验层应冻结自己的状态词表。

若提供 `parent_id` 而 `parent_ids` 为空，校验器会将后者填成仅含主父代的列表。若 `parent_ids` 非空，则 `parent_id` 必须位于其中。父代 ID 不得重复。该关系支持记录 O3 的多父代和主父代，但不自动验证 genome hash。

### 8.2 派生成本规则

成本“完整”的当前定义是：

1. `generation_cost` 与 `evaluation_cost` 均非 null；
2. `usage_records` 中每条记录都有非 null `api_cost`，且 `cost_basis` 不是 `unknown`。

完整时，模型将 `total_cost` 规范化为 `generation_cost + evaluation_cost`；若所有 usage 都是 `actual`，episode 的 `cost_basis=actual`，否则为 `estimated`，并令 `cost_complete=True`。不完整时，`total_cost=None`、`cost_complete=False`、`cost_basis=unknown`。

若调用方显式提供 `total_cost`，它必须与上述派生值一致；调用方提供的非 unknown `cost_basis` 也必须符合派生状态。`api_cost` 汇总字段本身不参与完整性判定，当前模型也不会自动把它与 generation/evaluation 或 usage 总额对账。

因此：

- 未知成本不是零成本；
- 完整成本只表示当前 episode 声明的两阶段与 usage 完整，不自动证明覆盖全部基础设施或跨 episode 生命周期成本；
- RQ2 使用这些字段前仍需定义跨步骤归因、价格版本和缺失数据处理。

## 9. RunLedger、manifest 与结果索引

### 9.1 写入与重开

`RunLedger(results_root, run_id)` 创建：

```text
<results_root>/raw/<run_id>/episodes.jsonl
<results_root>/raw/<run_id>/manifest.json
```

`run_id` 必须匹配 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`，即单个可移植路径组件。默认排他创建 run 目录与空 episodes 文件；已有目录会报错。`allow_existing=True` 可重开并读取已有 ID，防止重复追加。

manifest 当前含 `schema_version=1`、`run_id`、`created_at_utc`、`episodes_file`、`episode_count` 和可空 `provenance`。每次成功 append 后更新 `episode_count`，manifest 通过同目录临时文件、flush/fsync 和 `os.replace` 原子替换。

追加 episode 时，先核对 `run_id` 和进程内/重开恢复的 `episode_id` 唯一性，再以 JSONL 追加一行并 flush/fsync；重复 ID 抛出 `DuplicateEpisodeError`，不改写历史。锁是 `threading.Lock`，当前实现不承诺多进程或分布式写安全。结果索引也有独立 append 接口，不能与 ledger 的唯一性保证混为一谈。

### 9.2 ResultIndexer

`ResultIndexer` 将 `EpisodeRecord` 的审计字段以排序、紧凑 JSON 追加到指定 JSONL。索引包含 episode/run/task、谱系、模型、用量、成本、得分、来源和产物哈希等字段，但不包含原始轨迹字段。

原始轨迹应由独立产物保存并由 `artifact_hashes` 关联；当前索引不会自动计算或验证这些产物 hash，也不会自动在 `RunLedger.append` 时同步写索引。实验流程须显式调用并处理失败，不能把两个独立 append 描述为分布式事务。

## 10. 轨迹与来源记录

`Trajectory` 要求事件 `step` 严格递增。事件类型及有效负载为：

- `code`：必须有 `code`；
- `output`：可携带 JSON `output`；
- `exception`：必须有异常类型、消息、重试次数和是否将重试；
- `score`：可携带可空有限分数与详情。

不适用的事件负载会被拒绝。文本和结构化字段在构造时执行常见敏感键、密钥赋值、Bearer token 和常见 API token 形式脱敏；这降低泄漏风险，但不是对所有秘密格式的完备证明。

`RunProvenance` 记录 `git_sha`、`config_hash`、`git_dirty`、`development_mode` 和 UTC 捕获时间。脏工作树必须显式使用 development/allow-dirty 模式，不能将开发模式记录冒充干净提交复现。`config_hash` 是配置文件原始 bytes 的 SHA-256，与 genome 内容 hash 和 prompt hash 是不同命名空间。

## 11. 与研究主线的接口边界

RQ1 所需的轨迹特征、种群状态摘要、状态条件下的算子—父代配对键、元策略版本关联、即时/延迟信用和选择策略均待实现；现有字段只提供审计输入。RQ2 所需的多步元生产率、跨 episode 成本归集、自适应评价预算和停止规则均待实现；现有成本字段只提供记录基础。

所有后续实现必须遵守：

- 以实际轨迹和种群状态为依据，不把未来信息泄漏到当前决策；
- 以配对、父代、子代、任务、预算和种子连接信用记录；
- 保留失败、重试、零分、未知得分与未知成本；
- 在相同总成本和独立留出任务上比较研究假设；
- 不把 E1 算子异质性 pilot 写成 RQ1/RQ2 已验证；
- 不启动 PARKED 的 RQ3；
- 不实现双层元演化。

研究问题的详细定义见 [research_questions.md](research_questions.md)，创新声明的口径见 [innovation_points.md](innovation_points.md)。
