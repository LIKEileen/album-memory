# VLM 接入说明

## 运行前提

- Python 3.11。
- PostgreSQL 16，已安装 pgvector 扩展。
- 配置中的模型路径默认指向 /root/autodl-tmp/vlm/models/BAAI/bge-m3；工程不会复制模型。
- DATABASE_URL 从环境变量提供，不应写入仓库。

本轮没有安装依赖、创建数据库、执行 Alembic、加载模型或运行程序。

## 推荐调用顺序

```python
from uuid import UUID

from album_memory import AlbumMemory, MemoryConfig
from album_memory.enums import ConsentState, RetrievalIntent

memory = AlbumMemory(MemoryConfig.from_yaml("config.yaml"))

registered = memory.register_user(
    external_subject_key="tenant-a:user-42",
    consent_state=ConsentState.GRANTED,
)

result = memory.ingest_observation(
    registered.user_id,
    observation_payload,
    asset=asset_payload,
    idempotency_key="vlm-run:asset-42:prompt-v3",
)

# 在显式 worker 或受控批任务中调用；这里才会加载 BGE。
report = memory.process_pending(limit=8)

context = memory.retrieve(
    registered.user_id,
    "上个月旅行时去了哪里？",
    intent=RetrievalIntent.RECALL,
    top_k=5,
)

# 只记录真正放入模型上下文的条目。
memory.record_injection(
    context.retrieval_id,
    [item.memory_id for item in context.memories[:3]],
    [],
)
```

## 入库语义

- ImageObservation 使用 image_observation v1.1；Python 校验器是最终约束，schemas/image_observation.schema.json 便于其他语言预校验。
- 相同 idempotency_key 或相同 observation_id 且内容一致时，返回原记录并标记 idempotent_replay。
- 相同 observation_id 但内容不同会抛 IdempotencyConflict。
- 同一 asset_id 重解析时必须使用新的 observation_id，系统生成 observation_version + 1，并保留 supersedes_observation_id。
- ingest_observation 不做嵌入、聚类、画像或 LLM 调用。

## 查询语义

- recall 和 timeline 不返回画像 Claim。
- recommendation 只返回 L6、L8、L9 中 active、未过期且允许注入的 Claim。
- 用户级 profile_injection_enabled 默认为 false，需显式开启。
- 候选 Claim 不注入，冲突、过期或被替代的 Claim 不注入。

## 人工纠正

confirm_claim 会提高用户确认的优先级并记录确认事件。correct_claim 不改写旧 Claim，而是创建新版本并把旧版本转成 ended，保留 supersedes_claim_id 与原因。

## 维护

run_maintenance 默认 dry_run=True。它只计算软归档、短期 Claim 到期和撤回授权后的冻结动作。显式 dry_run=False 才修改状态；任何证据事实不会因遗忘而删除。
