from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from album_memory.enums import Horizon


class DimensionPolicy(BaseModel):
    dimension_id: str
    horizon: Horizon
    title: str
    allowed_signals: set[str]
    min_independent_events: int
    min_span_days: int = 0
    subject_required: bool = False
    min_source_modes: int = 1
    min_source_reliability: float = 0.55
    user_confirmation_required: bool = False
    max_auto_confidence: float = 0.95
    forbidden_inferences: list[str] = Field(default_factory=list)


COMMON_FORBIDDEN = [
    "身份、关系、所有权、偏好和能力不能由单次出现推出",
    "禁止心理、健康、人格和敏感人口属性推断",
    "禁止精确住宅和实时轨迹",
]


DIMENSION_POLICIES: dict[str, DimensionPolicy] = {
    "S1": DimensionPolicy(
        dimension_id="S1", horizon="short", title="当前承诺与进行中过程",
        allowed_signals={"explicit_commitment", "visible_state", "state_change", "project_artifact"},
        min_independent_events=2, subject_required=True, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "S2": DimensionPolicy(
        dimension_id="S2", horizon="short", title="近期行为分配",
        allowed_signals={"action_observed"}, min_independent_events=2,
        subject_required=True, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "S3": DimensionPolicy(
        dimension_id="S3", horizon="short", title="近期内容注意焦点",
        allowed_signals={"contains_object", "visible_text", "topic_observed"},
        min_independent_events=2, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "S4": DimensionPolicy(
        dimension_id="S4", horizon="short", title="近期社会参与配置",
        allowed_signals={"person_present", "group_size", "co_occurs_with"},
        min_independent_events=2, subject_required=True, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "S5": DimensionPolicy(
        dimension_id="S5", horizon="short", title="近期空间与移动状态",
        allowed_signals={"coarse_location_observed", "coarse_location_provided"},
        min_independent_events=2, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "S6": DimensionPolicy(
        dimension_id="S6", horizon="short", title="近期时间节律",
        allowed_signals={"capture_time_provided", "action_observed"},
        min_independent_events=3, subject_required=True, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "S7": DimensionPolicy(
        dimension_id="S7", horizon="short", title="可观察情绪表达",
        allowed_signals={"observable_expression"}, min_independent_events=2,
        subject_required=True, max_auto_confidence=0.5, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L1": DimensionPolicy(
        dimension_id="L1", horizon="long", title="稳定身份与生活角色",
        allowed_signals={"user_confirmation", "explicit_role_text"},
        min_independent_events=1, subject_required=True,
        min_source_reliability=0.8, user_confirmation_required=True,
        forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L2": DimensionPolicy(
        dimension_id="L2", horizon="long", title="持久实体关联",
        allowed_signals={"reidentified_entity"}, min_independent_events=3,
        min_span_days=90, subject_required=True, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L3": DimensionPolicy(
        dimension_id="L3", horizon="long", title="稳定社会网络",
        allowed_signals={"co_occurs_with", "relationship_confirmation"},
        min_independent_events=3, min_span_days=90, subject_required=True,
        forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L4": DimensionPolicy(
        dimension_id="L4", horizon="long", title="空间锚点与长期环境暴露",
        allowed_signals={"coarse_location_observed", "coarse_location_provided"},
        min_independent_events=3, min_span_days=90, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L5": DimensionPolicy(
        dimension_id="L5", horizon="long", title="稳定时间习惯",
        allowed_signals={"capture_time_provided", "action_observed"},
        min_independent_events=4, min_span_days=90, subject_required=True,
        forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L6": DimensionPolicy(
        dimension_id="L6", horizon="long", title="行为习惯",
        allowed_signals={"action_observed"}, min_independent_events=3,
        min_span_days=90, subject_required=True, forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L7": DimensionPolicy(
        dimension_id="L7", horizon="long", title="能力与技能",
        allowed_signals={"autonomous_step", "completed_outcome", "skill_confirmation"},
        min_independent_events=3, min_span_days=30, subject_required=True,
        forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L8": DimensionPolicy(
        dimension_id="L8", horizon="long", title="持续兴趣",
        allowed_signals={"topic_observed", "contains_object", "visible_text"},
        min_independent_events=3, min_span_days=90, min_source_modes=2,
        forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L9": DimensionPolicy(
        dimension_id="L9", horizon="long", title="比较选择偏好",
        allowed_signals={"comparable_choice", "preference_confirmation"},
        min_independent_events=3, min_span_days=30, subject_required=True,
        forbidden_inferences=COMMON_FORBIDDEN,
    ),
    "L10": DimensionPolicy(
        dimension_id="L10", horizon="long", title="视觉审美与记录风格",
        allowed_signals={"visual_style_observed"}, min_independent_events=3,
        min_span_days=30, min_source_modes=1, forbidden_inferences=COMMON_FORBIDDEN,
    ),
}


BLOCKED_TEXT_MARKERS = {
    "抑郁", "焦虑症", "心理疾病", "人格", "性格内向", "性格外向",
    "住址", "门牌号", "实时位置", "收入", "消费能力", "一定是其",
    "伴侣", "父母", "同事", "朋友", "职业是", "拥有",
}


def statement_is_safe(statement: str) -> bool:
    lowered = statement.lower()
    return not any(marker.lower() in lowered for marker in BLOCKED_TEXT_MARKERS)


def evaluate_gate(policy: DimensionPolicy, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    event_ids = {str(item["event_id"]) for item in evidence if item.get("event_id")}
    times = sorted(item["at"] for item in evidence if isinstance(item.get("at"), datetime))
    source_modes = {item.get("source_kind") for item in evidence if item.get("source_kind")}
    subject_confirmed = any(item.get("user_presence") == "confirmed" for item in evidence)
    user_confirmed = any(
        item.get("evidence_type") == "user_confirmation" for item in evidence
    )
    reliabilities = [
        float(item["source_reliability"])
        for item in evidence
        if item.get("source_reliability") is not None
    ]
    average_source_reliability = (
        sum(reliabilities) / len(reliabilities) if reliabilities else 0.0
    )
    violations = []
    span_days = (times[-1] - times[0]).days if len(times) >= 2 else 0
    if len(event_ids) < policy.min_independent_events:
        violations.append(
            f"independent_event_count<{policy.min_independent_events}"
        )
    if span_days < policy.min_span_days:
        violations.append(f"time_span_days<{policy.min_span_days}")
    if policy.subject_required and not subject_confirmed:
        violations.append("subject_not_confirmed")
    if len(source_modes) < policy.min_source_modes:
        violations.append(f"source_modes<{policy.min_source_modes}")
    if average_source_reliability < policy.min_source_reliability:
        violations.append(
            f"average_source_reliability<{policy.min_source_reliability}"
        )
    if policy.user_confirmation_required and not user_confirmed:
        violations.append("user_confirmation_required")
    return {
        "passed": not violations,
        "independent_event_count": len(event_ids),
        "time_span_days": span_days,
        "subject_confirmed": subject_confirmed,
        "source_modes": sorted(x for x in source_modes if x),
        "average_source_reliability": round(average_source_reliability, 3),
        "user_confirmed": user_confirmed,
        "violations": violations,
    }
