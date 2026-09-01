# Album Memory

面向相册图片 VLM 智能体的可审计记忆与用户画像 Python 子模块。项目只接收 VLM 已生成的结构化观察和图片旁路元数据，不复制原图，不提供 HTTP 服务。

核心能力包括：不可变观察与事实、L1/L2/L3 多粒度图记忆、显式异步处理、PostgreSQL/pgvector 检索、短期 S1-S7 与长期 L1-L10 画像、正反证冲突、软归档遗忘、Markdown 画像输出和旧版 AlbumDoc 适配。

所有数据库连接、模型加载、迁移和处理都必须由调用方显式触发。导入 album_memory 不会连接数据库或加载 BGE-M3。

本次交付状态：代码和未执行测试样例已准备；未安装依赖、未创建数据库、未执行迁移、未加载模型、未运行测试。

参见 docs/INTEGRATION.md、docs/ARCHITECTURE.md、docs/DATA_DICTIONARY.md、docs/CONFLICT_AND_FORGETTING.md 和 schemas/。
