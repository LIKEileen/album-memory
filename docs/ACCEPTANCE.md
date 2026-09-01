# 验收清单

本工程预置以下轻量样例，但依照本轮约束未执行 pytest、数据库迁移、模型加载、构建脚本或端到端验证。因此当前交付状态是：未运行验证。

机器可读场景清单位于 tests/acceptance_scenarios.yaml。

- 重复 observation_id 且内容一致：幂等返回。
- 重复 observation_id 但内容不同：拒绝并报告冲突。
- 同一图片新 observation_id：生成不可变的新观察版本。
- 事件硬门槛：远时间或冲突地点不因向量近似而误合并。
- 正反证：原始事实不覆盖，保留冲突与替代链。
- 短期门槛：7/30 天窗口、有效期和复核时间。
- 长期门槛：独立事件数、时间跨度、主体确认、来源模式和反证。
- 过期短期 Claim：维护时转为候选并等待重评。
- 软归档与恢复：只影响召回；实际注入或新证据可恢复。
- 跨用户隔离：所有读取、写入和强化均验证 user_id。
- candidate、conflicted、ended、expired Claim 不注入。

数据库相关验收需要 PostgreSQL 16 与 pgvector。模型相关验收需要现有 BGE-M3 路径可读。执行前应复制 config.example.yaml，并通过环境变量注入 DATABASE_URL 与可选 LLM 密钥。
