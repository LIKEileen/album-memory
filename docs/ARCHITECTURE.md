# 架构说明

本工程是可被 VLM 智能体导入的本地 Python 子模块，不监听端口。PostgreSQL 16 与 pgvector 是唯一持久化后端；模型、数据库连接和可选 LLM 客户端均延迟初始化。

## 写入与处理边界

1. ingest_observation 负责严格校验、幂等判定、不可变观察版本、事实和 processing_jobs 的同事务写入，不加载嵌入模型。
2. process_pending 使用 SKIP LOCKED 领取任务，并通过 PostgreSQL advisory lock 保证同一用户串行处理。失败任务带退避时间重试。
3. worker 延迟加载 BGE-M3，将实体、单图和事件摘要写成 L1、L2、L3 版本节点，同时维护受控边。
4. 画像刷新在事实与事件稳定后发生。短期画像每次重评；长期画像仅在新增独立事件、反证或人工更正时重评。

## 多粒度图

- L1：单个可见实体或动作的摘要，父节点为所属 L2。
- L2：单张图片观察的完整描述，父节点为所属 L3。
- L3：一个事件的跨图片摘要。
- 边类型：parent_of、same_asset、same_event、temporal_adjacent、semantic_similar。
- memory_items 采用不可变版本语义。摘要变化时旧版本归档，新版本通过 supersedes_memory_id 串接。

事件归并首先应用硬门槛：时间差、粗粒度地点、来源可信度与主体确定性。只有通过硬门槛后才使用语义相似度与 GMM 自适应阈值。语义连接不会独自决定事件合并。

## 检索

检索不会把全图装入内存：

1. PostgreSQL 并行形成 L1 词元候选、L2/L3 pgvector 候选和时间/事件过滤候选。
2. 在有限候选集合上融合 BM25、向量、来源可靠度、无下限时间衰减和实际注入频次。
3. 只对候选子图执行 PPR。
4. 事实回顾以事件记忆为主；推荐意图才允许检索已激活且未过期的 L6、L8、L9 画像。
5. retrieve 只登记 recalled。调用 record_injection 后才登记 injected，并强化或恢复被实际使用的记忆。

## 数据一致性

- 每张图片可有多个不可变观察版本，只有一个 current 版本。
- observation_id 加内容哈希保证一致重放；相同 ID 的不同内容直接报冲突。
- idempotency_key 在用户内唯一。
- 原始 observation_facts 不覆盖。规则引擎写冲突状态和替代链。
- 所有表都含 user_id 或通过受控外键归属用户；服务层查询必须携带 user_id。

## 隐私边界

未授权用户不可写入。敏感或 blocked_from_profile 的观察仍可保留证据，但不会生成画像。系统不自动推断敏感人口属性、精确住宅或轨迹、健康心理状态、未确认的人际关系、所有权、职业和人格。撤回授权会冻结在线处理；真正删除应由单独保留期策略完成。
