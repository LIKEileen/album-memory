from __future__ import annotations

from album_memory.contracts import ClaimView, ProfileSnapshot


def render_profile_markdown(snapshot: ProfileSnapshot) -> str:
    lines = [
        f"# 用户画像 {snapshot.user_id}",
        "",
        f"生成时间：{snapshot.generated_at.isoformat()}",
        "",
        "以下内容只包含当前可见的候选或已激活 Claim；每条结论均保留状态、有效期和正反证据。",
        "",
    ]
    lines.extend(_section("短期画像 S1-S7", snapshot.short_term))
    lines.extend(_section("长期画像 L1-L10", snapshot.long_term))
    return "\n".join(lines).rstrip() + "\n"


def _section(title: str, claims: list[ClaimView]) -> list[str]:
    lines = [f"## {title}", ""]
    if not claims:
        lines.extend(["暂无可展示结论。", ""])
        return lines
    for claim in claims:
        lines.append(
            f"### {claim.dimension_id} · {claim.status.value} · 置信度 {claim.confidence:.3f}"
        )
        lines.append("")
        lines.append(claim.statement)
        lines.append("")
        lines.append(f"- 审核：{claim.review_state.value}")
        lines.append(
            f"- 有效期：{_time(claim.valid_from)} 至 {_time(claim.valid_to)}"
        )
        if claim.next_review_at:
            lines.append(f"- 下次复核：{_time(claim.next_review_at)}")
        if claim.expires_at:
            lines.append(f"- 到期时间：{_time(claim.expires_at)}")
        if claim.resolution_reason:
            lines.append(f"- 冲突/替代说明：{claim.resolution_reason}")
        lines.append(
            f"- 支持证据：{', '.join(claim.evidence_ids) if claim.evidence_ids else '无'}"
        )
        lines.append(
            f"- 反证：{', '.join(claim.counter_evidence_ids) if claim.counter_evidence_ids else '无'}"
        )
        lines.append("")
    return lines


def _time(value) -> str:
    return value.isoformat() if value else "未限定"
