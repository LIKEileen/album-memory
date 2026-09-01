# 冲突与遗忘策略

## 事实冲突

冲突键由规范化主体、谓词和值与时间范围构成。原始事实保持不可变，规则按以下顺序处理：

1. 用户确认或更正优先。
2. 可信元数据优先于像素视觉候选。
3. 同一主体的时变谓词若时间不重叠，判为演变。
4. 场景或主体不同的结论允许共存。
5. 仍无法判断时，可选 LLM 只能返回 conflict、evolution、coexist 建议；结果进入 needs_review，不能直接激活 Claim。

被替代 Claim 状态转为 ended 或 rejected，新 Claim 记录 supersedes_claim_id、resolution_reason、支持证据和反证。

## 画像冲突

每个维度先执行固定门槛和安全过滤。门槛不足时只创建 candidate。出现反证时保留正反两条证据链；只有确定的来源优先级或人工确认才能自动消解。

## 遗忘

遗忘强度没有固定下限，由最后强化时间、实际注入次数、来源可靠度和证据量组成。强度低且闲置时间足够的 memory_item 被软归档：

- 不删除 observation、fact、event 或 Claim 证据链。
- active Claim 正在引用的证据、用户确认内容和关键 active L3 事件受保护。
- 新证据、用户确认或再次实际注入会恢复记忆。
- 短期 Claim 到期后重评；长期 Claim 不因近期无照片自动失效。
- 授权撤回与保留期限删除不属于遗忘，由独立治理流程负责。

维护接口默认 dry_run=True，建议先审阅 MaintenanceReport 再显式应用。
