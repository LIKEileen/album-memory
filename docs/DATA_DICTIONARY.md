# 数据字典

| 表 | 用途 | 关键约束 |
|---|---|---|
| users | 用户、授权、画像注入开关、保留期 | external_subject_key 唯一 |
| media_assets | 图片元数据和粗粒度位置 | 用户内 external_asset_id/sha256 防重复 |
| image_observations | VLM 原始观察与不可变版本 | 用户内 observation_key 唯一；asset + version 唯一；current 唯一 |
| observation_facts | 原子事实、规范化 SPO、证据和冲突状态 | 原始行只增不覆盖 |
| events | 事件范围、归并门槛和摘要版本 | 用户隔离；状态可审计 |
| event_assets | 事件与图片关联 | event + asset 唯一 |
| memory_items | L1/L2/L3 文本、1024 维向量、版本、衰减数据 | 部分唯一索引保证单个 active 版本 |
| memory_edges | 受控图边和权重 | from/to/type 唯一；禁止跨用户写入 |
| profile_claims | S1–S7/L1–L10 画像版本、门槛结果和有效期 | active Claim 部分唯一；替代链保留 |
| claim_evidence | Claim 的支持/反证及事件、事实、记忆引用 | polarity 区分 supporting/counter |
| retrieval_events | recalled、injected、feedback 三阶段审计 | 实际注入内容单独记录 |
| research_runs | 模型、提示词、规则、配置版本 | 支持处理结果复现与审计 |
| processing_jobs | 持久任务、幂等、重试和用户串行处理 | job_type + idempotency_key 唯一 |
| user_confirmations | 确认、纠正和理由 | 用户证据优先级最高 |

## 核心状态

- observation：current、superseded、rejected。
- memory：active、archived、conflicted、ended、rejected。
- claim：candidate、active、conflicted、ended、rejected、archived。
- review_state：auto_passed、needs_review、human_confirmed、human_rejected。
- processing_job：queued、running、succeeded、failed。

## 画像维度

短期画像：S1 当前承诺与过程、S2 近期行为分配、S3 近期内容焦点、S4 近期社会参与、S5 近期空间移动、S6 近期时间节律、S7 可观察情绪表达。短期使用 7 天和 30 天窗口，带有效期与复核时间。

长期画像：L1 稳定身份角色、L2 持久实体关联、L3 稳定社会网络、L4 空间锚点、L5 时间习惯、L6 行为习惯、L7 能力技能、L8 持续兴趣、L9 比较偏好、L10 视觉审美与记录风格。每个维度的事件数、跨度、主体、来源可靠度和来源模式门槛固定在 album_memory/policies.py；L1 只有用户确认后才能激活。

## 时间语义

- captured_at：图片拍摄时间，优先采用可信元数据。
- generated_at：VLM 观察生成时间。
- valid_from/valid_to：Claim 对应事实成立范围。
- expires_at：短期 Claim 的自动重评期限；长期 Claim 不因缺少近期照片失效。
- last_reinforced_at：最后一次实际注入或显式恢复时间，召回不会更新。
