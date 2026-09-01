# 旧版 AlbumDoc 导入

album_memory.legacy.iter_legacy_jsonl 将旧 docs.jsonl 映射为 v1.1 观察。命令示例：

```bash
album-memory --config config.yaml import-legacy \
  --user-id 00000000-0000-0000-0000-000000000000 \
  --jsonl /path/to/docs.jsonl
```

如需只生成可审阅的规范 JSONL、不写数据库，可使用 scripts/map_legacy_jsonl.py。该脚本输出 asset、observation、trust 和 profile_activation_allowed 字段。

适配器遵循保守映射：

- source_kind 标为 imported，置信度上限为 0.35。
- 旧文本只进入详细描述，旧对象标签逐条映射为低可信原子事实。
- 不补造 bbox、OCR、人物身份、用户归属、所有权、地点或画像结论。
- user_presence 为 unknown，所有 limitations 保持 true。
- safety.blocked_from_profile 为 true；旧数据不能单独触发画像门槛。

导入只排队，不调用 BGE 或处理任务。需要后续显式执行 process 命令。原型 JSON、NPZ、Chroma 产物不会被读取、修改或依赖。
