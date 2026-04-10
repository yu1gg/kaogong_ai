"""两阶段评分的后处理与证据校验模块。"""

import difflib
import json
import logging
import re
from numbers import Number
from typing import Any, Dict, Iterable, Sequence, Union

from pydantic import ValidationError

from app.core.config import settings
from app.models.schemas import (
    EvidenceExtractionPayload,
    EvidenceItem,
    EvaluationResult,
    QuestionDefinition,
    ReasonedScoreItem,
    StageTwoScoringPayload,
)

from .keyword_matcher import match_all_categories

logger = logging.getLogger(__name__)

ORAL_EXPRESSION_PHRASES = (
    "我觉得",
    "你想啊",
    "挺好的",
    "没毛病",
    "有点跑偏",
    "硬着头皮",
    "看着办",
    "弄好点",
)
MAX_ORAL_EVIDENCE_ITEMS = 3
ROLE_MARKERS = (
    "统筹协调",
    "政策指导",
    "分类指导",
    "部门协同",
    "长效机制",
    "省直机关",
    "省直属机关",
    "岗位职责",
)
PROVINCE_MARKERS = (
    "河南",
    "一主两副",
    "一圈两带",
    "四域多点",
    "十大战略",
)
MEASURE_MARKERS = (
    "首先",
    "其次",
    "再次",
    "最后",
    "然后",
    "另外",
    "第一",
    "第二",
    "第三",
    "第四",
)
STRUCTURE_MARKERS = MEASURE_MARKERS
POSITIVE_SIGNAL_GROUPS = (
    ("初衷", "积极", "好处", "肯定", "值得肯定", "主动作为", "理性看待"),
    ("公信力", "背书", "提高知名度", "打开销路", "助农", "乡村振兴", "数字经济", "服务经济"),
)
PROBLEM_SIGNAL_GROUPS = (
    ("一刀切", "强制", "必须", "行政命令式", "统一排名"),
    ("形式主义", "盲目跟风", "为了直播而直播", "表演式"),
    ("每个县情况不一样", "地区差异", "县域差异", "因地制宜", "资源禀赋", "功能定位", "不顾本地实际", "适配度"),
    ("排名", "流量", "刷数据", "数字竞赛", "激励导向", "政绩比拼"),
    ("耽误", "主责主业", "主业错位", "日常工作"),
    ("物流", "供应链", "包装", "品牌建设", "冷链", "售后", "产品质量", "质量追溯", "电商基础设施"),
    ("政府形象", "政府公信力", "信任", "长效机制", "可持续", "口碑"),
)
ROOT_CAUSE_SIGNAL_GROUPS = (
    ("政绩观", "错误的政绩观", "急躁冒进", "认识误区"),
    ("科学决策", "缺乏科学决策", "政策制定前未充分调研", "加强调研", "摸清"),
    ("政策执行", "执行上级精神", "统筹协调", "分类指导"),
    ("配套保障缺位", "并未配套", "保障缺位", "基础设施"),
)
MEASURE_SIGNAL_GROUPS = (
    (
        "取消强制",
        "不强制",
        "鼓励而非强制",
        "鼓励探索",
        "让各县自己决定",
        "让各县根据自己的情况来",
        "分类指导",
        "一县一策",
        "因地制宜",
        "自主选择",
    ),
    (
        "优化考核",
        "不搞排名",
        "别搞排名",
        "实际销量",
        "群众满意度",
        "品牌提升效果",
        "实际助农效果",
        "第三方评估",
        "农民收入",
        "带动起来",
    ),
    (
        "完善物流",
        "冷链",
        "基础设施",
        "预处理中心",
        "包装",
        "供应链",
        "质量追溯",
        "公共品牌",
        "物流产业园",
        "产品质量",
    ),
    ("人才培训", "培养", "培训", "专业人才", "电商人才", "技术支持", "专业指导"),
    ("电商平台合作", "市场主体", "平台合作", "交流平台", "现场会", "专题培训", "资源共享", "企业负责"),
    ("省直相关部门", "省级层面", "统筹协调", "牵头", "开展调研", "示范推广", "容错纠错"),
    ("文旅推介", "一播多效", "订单农业", "线下展销", "分类赋能", "政务创新交流平台", "本地网红", "网红"),
)
INNOVATION_SIGNAL_GROUPS = (
    ("第三方评估", "一播多效", "分类赋能", "政务创新交流平台", "一县一策"),
    ("订单农业", "线下展销", "文旅推介", "直播引流", "容错纠错", "本地网红"),
)
RULE_BASED_DIMENSIONS = (
    "现象解读",
    "危害根源分析",
    "科学决策与措施",
    "语言逻辑与岗位适配",
    "创新思维",
)


def _effective_text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, Number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_string_list(values: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def _unique_matches(text: str, markers: Sequence[str]) -> list[str]:
    ordered_unique = []
    seen = set()
    for marker in markers:
        if marker in text and marker not in seen:
            ordered_unique.append(marker)
            seen.add(marker)
    return ordered_unique


def _count_marker_groups(text: str, groups: Sequence[Sequence[str]]) -> int:
    return sum(1 for group in groups if any(marker in text for marker in group))


def _round_score(value: float, maximum: float) -> float:
    return round(min(max(value, 0.0), maximum), 1)


def _parse_raw_result(raw_llm_result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(raw_llm_result, dict):
        return raw_llm_result

    try:
        clean_str = raw_llm_result.strip().strip("`").removeprefix("json").strip()
        return json.loads(clean_str)
    except json.JSONDecodeError as exc:
        logger.error("大模型返回了非标准 JSON: %s", raw_llm_result)
        raise ValueError("系统未能成功解析模型评分结果。") from exc


def _normalize_dimension_scores(
    raw_scores: Dict[str, Any],
    question: QuestionDefinition,
    validation_notes: list[str],
) -> Dict[str, float]:
    expected_dimensions = {item.name: item.score for item in question.dimensions}
    normalized_scores: Dict[str, float] = {}

    for dimension_name, max_score in expected_dimensions.items():
        raw_score = raw_scores.get(dimension_name, 0.0)
        score = _to_float(raw_score, default=0.0)
        clamped_score = round(min(max(score, 0.0), max_score), 1)
        if clamped_score != score:
            validation_notes.append(
                f"维度 [{dimension_name}] 得分超出范围，已自动收敛到 0-{max_score}。"
            )
        normalized_scores[dimension_name] = clamped_score

    unexpected_dimensions = sorted(set(raw_scores) - set(expected_dimensions))
    if unexpected_dimensions:
        validation_notes.append(
            f"模型返回了未定义维度 {unexpected_dimensions}，已自动忽略。"
        )

    missing_dimensions = [
        item.name for item in question.dimensions if item.name not in raw_scores
    ]
    if missing_dimensions:
        validation_notes.append(
            f"模型缺失维度 {missing_dimensions}，已按 0 分补齐。"
        )

    return normalized_scores


def _deduplicate_evidence_items(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    deduplicated: list[EvidenceItem] = []
    seen = set()
    for item in items:
        key = (
            item.evidence_type,
            item.evidence_text.strip(),
            item.claim.strip(),
            item.stance,
            item.dimension_hint,
        )
        if key in seen:
            continue
        deduplicated.append(item)
        seen.add(key)
    return deduplicated


def _extract_salvageable_fragment(evidence_text: str, transcript: str) -> str | None:
    cleaned_text = evidence_text.strip()
    if not cleaned_text:
        return None
    if cleaned_text in transcript:
        return cleaned_text

    fragments = [
        fragment.strip()
        for fragment in re.split(r"[，。；：！？\n\r]+", cleaned_text)
        if len(fragment.strip()) >= 8
    ]
    fragments.sort(key=len, reverse=True)
    for fragment in fragments:
        if fragment in transcript:
            return fragment

    matcher = difflib.SequenceMatcher(a=cleaned_text, b=transcript)
    match = matcher.find_longest_match(0, len(cleaned_text), 0, len(transcript))
    if match.size >= 10:
        candidate = transcript[match.b : match.b + match.size].strip(" ，。；：！？\n\r\t")
        if len(candidate) >= 8:
            return candidate
    return None


def _build_deterministic_quote_evidence(transcript: str) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for phrase in ORAL_EXPRESSION_PHRASES:
        if phrase in transcript:
            items.append(
                EvidenceItem(
                    id="",
                    dimension_hint="语言逻辑与岗位适配",
                    claim=f"原文出现口语化表达“{phrase}”",
                    evidence_text=phrase,
                    evidence_type="quote",
                    stance="language",
                )
            )
            if len(items) >= MAX_ORAL_EVIDENCE_ITEMS:
                break
    return items


def _collect_province_markers(question: QuestionDefinition) -> list[str]:
    markers = list(PROVINCE_MARKERS)
    if question.province:
        markers.insert(0, question.province)
    for keyword in question.weakKeywords:
        if keyword not in markers:
            markers.append(keyword)
    return markers


def _build_absence_evidence(
    transcript: str,
    question: QuestionDefinition,
    extracted_items: Sequence[EvidenceItem],
) -> list[EvidenceItem]:
    transcript_lower = transcript.lower()
    absence_items: list[EvidenceItem] = []

    province_markers = _collect_province_markers(question)
    found_province_markers = [marker for marker in province_markers if marker.lower() in transcript_lower]
    found_role_markers = [marker for marker in ROLE_MARKERS if marker.lower() in transcript_lower]
    if not found_province_markers or not found_role_markers:
        missing_labels: list[str] = []
        if not found_province_markers:
            missing_labels.append("河南省情/发展格局")
        if not found_role_markers:
            missing_labels.append("省直机关岗位视角")

        missing_examples: list[str] = []
        if not found_province_markers:
            missing_examples.extend(province_markers[:4])
        if not found_role_markers:
            missing_examples.extend(ROLE_MARKERS[:4])

        absence_items.append(
            EvidenceItem(
                id="",
                dimension_hint="语言逻辑与岗位适配",
                claim="原文未充分体现" + "、".join(missing_labels),
                evidence_text=(
                    "原文未出现相关标识词，如"
                    + "、".join(missing_examples[:6])
                ),
                evidence_type="absence",
                stance="negative",
            )
        )

    found_measure_markers = [marker for marker in MEASURE_MARKERS if marker in transcript]
    quoted_measure_items = [
        item
        for item in extracted_items
        if item.evidence_type == "quote"
        and item.dimension_hint == "科学决策与措施"
    ]
    if len(set(found_measure_markers)) < 2 and len(quoted_measure_items) < 2:
        absence_items.append(
            EvidenceItem(
                id="",
                dimension_hint="科学决策与措施",
                claim="原文措施展开层次有限，结构化措施不足",
                evidence_text=(
                    "原文措施层次标识不足 2 处，检测到："
                    + ("、".join(sorted(set(found_measure_markers))) if found_measure_markers else "无")
                ),
                evidence_type="absence",
                stance="negative",
            )
        )

    return absence_items


def _scale_scores_to_target(
    scores: Dict[str, float],
    question: QuestionDefinition,
    target_total: float,
) -> Dict[str, float]:
    current_total = round(sum(scores.values()), 1)
    if not scores or abs(current_total - target_total) < 0.1:
        return scores

    max_scores = {item.name: item.score for item in question.dimensions}
    scaled = dict(scores)

    if target_total < current_total:
        factor = target_total / current_total if current_total > 0 else 0.0
        scaled = {
            name: round(min(max(value * factor, 0.0), max_scores.get(name, value)), 1)
            for name, value in scaled.items()
        }
    else:
        remaining = round(target_total - current_total, 1)
        growth_order = (
            "科学决策与措施",
            "危害根源分析",
            "现象解读",
            "语言逻辑与岗位适配",
            "创新思维",
        )
        for name in growth_order:
            if remaining <= 0:
                break
            headroom = round(max_scores.get(name, scaled.get(name, 0.0)) - scaled.get(name, 0.0), 1)
            if headroom <= 0:
                continue
            delta = min(headroom, remaining)
            scaled[name] = round(scaled.get(name, 0.0) + delta, 1)
            remaining = round(target_total - sum(scaled.values()), 1)

    diff = round(target_total - sum(scaled.values()), 1)
    if scaled and diff != 0:
        for name in ("科学决策与措施", "危害根源分析", "现象解读", "语言逻辑与岗位适配", "创新思维"):
            if name not in scaled:
                continue
            adjusted = round(
                min(max(scaled[name] + diff, 0.0), max_scores.get(name, scaled[name])),
                1,
            )
            applied = round(adjusted - scaled[name], 1)
            scaled[name] = adjusted
            diff = round(diff - applied, 1)
            if diff == 0:
                break

    return {
        name: _round_score(value, max_scores.get(name, value))
        for name, value in scaled.items()
    }


def _compute_rule_based_dimension_scores(
    transcript: str,
    question: QuestionDefinition,
    matched_keywords: Dict[str, list[str]],
) -> tuple[Dict[str, float], list[str]]:
    expected_dimensions = {item.name for item in question.dimensions}
    if not set(RULE_BASED_DIMENSIONS).issubset(expected_dimensions):
        return {}, []

    province_markers = list(dict.fromkeys(_collect_province_markers(question)))
    oral_matches = _unique_matches(transcript, [*ORAL_EXPRESSION_PHRASES, "我觉着"])
    role_matches = _unique_matches(transcript, ROLE_MARKERS)
    province_matches = _unique_matches(transcript, province_markers)
    structure_matches = _unique_matches(transcript, STRUCTURE_MARKERS)

    positive_count = _count_marker_groups(transcript, POSITIVE_SIGNAL_GROUPS)
    problem_count = _count_marker_groups(transcript, PROBLEM_SIGNAL_GROUPS)
    root_cause_count = _count_marker_groups(transcript, ROOT_CAUSE_SIGNAL_GROUPS)
    measure_count = _count_marker_groups(transcript, MEASURE_SIGNAL_GROUPS)
    innovation_count = _count_marker_groups(transcript, INNOVATION_SIGNAL_GROUPS)
    effective_length = _effective_text_length(transcript)

    oral_count = len(oral_matches)
    role_count = len(role_matches)
    province_count = len(province_matches)
    structure_count = len(structure_matches)

    scores = {
        "现象解读": _round_score(
            1.0
            + (1.3 if positive_count else 0.0)
            + (1.5 if problem_count >= 1 else 0.0)
            + (0.7 if positive_count and problem_count else 0.0)
            + (0.5 if structure_count >= 2 else 0.0)
            + (0.7 if role_count + province_count >= 2 else 0.0)
            + (0.5 if effective_length >= 600 and problem_count >= 5 and measure_count >= 4 else 0.0)
            + (0.3 if effective_length >= 900 and oral_count == 0 else 0.0)
            - (0.4 if oral_count >= 4 else 0.0)
            - (0.2 if 2 <= oral_count < 4 else 0.0),
            8.0,
        ),
        "危害根源分析": _round_score(
            0.8
            + min(problem_count, 4) * 0.85
            + min(root_cause_count, 2) * 0.7
            + (0.6 if province_count >= 2 or role_count >= 1 else 0.0)
            + (0.7 if effective_length >= 600 and problem_count >= 5 else 0.0)
            + (0.4 if root_cause_count >= 1 and measure_count >= 3 else 0.0)
            - (0.2 if oral_count >= 4 else 0.0),
            7.0,
        ),
        "科学决策与措施": _round_score(
            min(measure_count, 4) * 1.0
            + 0.8
            + (0.8 if measure_count >= 4 and effective_length >= 600 else 0.0)
            + (0.5 if problem_count >= 5 and measure_count >= 3 else 0.0)
            + (0.6 if measure_count >= 3 and structure_count >= 2 else 0.0)
            + (0.8 if role_count >= 1 else 0.0)
            + (0.6 if province_count >= 2 else 0.0),
            8.0,
        ),
        "语言逻辑与岗位适配": _round_score(
            1.2
            + (1.0 if oral_count == 0 else 0.4 if oral_count == 1 else -0.4 if oral_count >= 4 else 0.0)
            + (0.9 if structure_count >= 2 else 0.3 if structure_count >= 1 else 0.0)
            + (0.9 if role_count >= 1 else 0.0)
            + (0.7 if province_count >= 2 else 0.3 if province_count == 1 else 0.0)
            + (0.4 if effective_length >= 900 else 0.0),
            5.0,
        ),
        "创新思维": _round_score(
            (0.6 if innovation_count >= 1 else 0.0)
            + (0.6 if innovation_count >= 2 else 0.0)
            + (0.4 if role_count >= 1 and province_count >= 2 else 0.0)
            + (0.3 if measure_count >= 4 else 0.0)
            + (0.2 if len(matched_keywords.get("bonus", [])) >= 2 else 0.0),
            2.0,
        ),
    }

    if measure_count <= 1:
        scores["科学决策与措施"] = min(scores["科学决策与措施"], 3.5)
    elif measure_count == 2:
        scores["科学决策与措施"] = min(scores["科学决策与措施"], 5.0)

    total_score = round(sum(scores.values()), 1)
    rule_notes: list[str] = []
    target_floor = total_score
    target_cap = question.fullScore

    if effective_length >= 400 and positive_count >= 1 and problem_count >= 4:
        target_floor = max(target_floor, 14.0)

    if effective_length >= 600 and problem_count >= 4 and measure_count >= 4 and oral_count <= 1:
        target_floor = max(target_floor, 19.0)

    if effective_length >= 700 and problem_count >= 5 and measure_count >= 4:
        target_floor = max(target_floor, 20.0)

    if effective_length >= 900 and oral_count == 0 and problem_count >= 5 and measure_count >= 4:
        target_floor = max(target_floor, 23.0)

    if province_count >= 4 and role_count >= 2 and effective_length >= 1000:
        target_floor = max(target_floor, 25.0 + min(innovation_count, 2))

    if role_count == 0 and province_count < 4 and innovation_count == 0 and measure_count <= 4 and total_score > 22.0:
        target_cap = min(target_cap, 22.0)

    if oral_count >= 3 and role_count == 0 and province_count == 0 and measure_count <= 2 and effective_length < 500:
        target_cap = min(target_cap, 16.0)

    if oral_count >= 1 and role_count == 0 and province_count <= 1 and measure_count <= 3 and effective_length < 600:
        target_cap = min(target_cap, 17.5)

    if target_floor > total_score:
        scores = _scale_scores_to_target(scores, question, target_floor)
        rule_notes.append(f"规则校准将总分下限抬至 {target_floor:.1f}，避免高内容强度答案被模型均分化压低。")
        total_score = round(sum(scores.values()), 1)

    if target_cap < total_score:
        scores = _scale_scores_to_target(scores, question, target_cap)
        rule_notes.append(f"规则校准将总分上限压至 {target_cap:.1f}，避免口语化或浅层答案被模型虚高。")

    return scores, rule_notes


def prepare_evidence_packet(
    raw_llm_result: Union[str, Dict[str, Any]],
    transcript: str,
    question: QuestionDefinition,
) -> tuple[EvidenceExtractionPayload, list[str]]:
    """整理第一阶段证据抽取结果，并补充规则型缺失证据。"""

    validation_notes: list[str] = []

    try:
        parsed_result = _parse_raw_result(raw_llm_result)
        extracted = EvidenceExtractionPayload.model_validate(parsed_result)
    except (ValueError, ValidationError) as exc:
        validation_notes.append("第一阶段证据抽取结果无效，已回退到规则型证据整理。")
        extracted = EvidenceExtractionPayload()

    evidence_items: list[EvidenceItem] = []
    expected_dimensions = {item.name for item in question.dimensions}

    for index, item in enumerate(extracted.evidence_items, start=1):
        evidence_text = item.evidence_text.strip()
        if not evidence_text:
            validation_notes.append(f"第一阶段第 {index} 条证据缺少 evidence_text，已忽略。")
            continue
        resolved_evidence_text = _extract_salvageable_fragment(evidence_text, transcript)
        if resolved_evidence_text is None:
            validation_notes.append(
                f"第一阶段证据无法在原文中命中，已忽略: {evidence_text}"
            )
            continue
        if resolved_evidence_text != evidence_text:
            validation_notes.append(
                f"第一阶段证据未逐字命中，已自动对齐为原文片段: {resolved_evidence_text}"
            )
        dimension_hint = item.dimension_hint if item.dimension_hint in expected_dimensions else ""
        if item.dimension_hint and not dimension_hint:
            validation_notes.append(
                f"第一阶段证据使用了未知维度 [{item.dimension_hint}]，已清空维度提示。"
            )
        evidence_items.append(
            EvidenceItem(
                id="",
                dimension_hint=dimension_hint,
                claim=item.claim.strip() or resolved_evidence_text,
                evidence_text=resolved_evidence_text,
                evidence_type="quote",
                stance=item.stance,
            )
        )

    evidence_items.extend(_build_deterministic_quote_evidence(transcript))
    evidence_items.extend(_build_absence_evidence(transcript, question, evidence_items))
    evidence_items = _deduplicate_evidence_items(evidence_items)

    for index, item in enumerate(evidence_items, start=1):
        item.id = f"E{index}"

    coverage_notes = _clean_string_list(extracted.coverage_notes)
    if not evidence_items:
        validation_notes.append("证据包为空，第二阶段评分将只能做极保守判断。")

    return (
        EvidenceExtractionPayload(
            evidence_items=evidence_items,
            coverage_notes=coverage_notes,
            summary=extracted.summary.strip(),
        ),
        validation_notes,
    )


def _normalize_reason_items(
    raw_items: Sequence[Any],
    evidence_map: Dict[str, EvidenceItem],
    expected_dimensions: set[str],
    validation_notes: list[str],
    field_name: str,
) -> list[ReasonedScoreItem]:
    normalized: list[ReasonedScoreItem] = []
    seen = set()

    for raw_item in raw_items or []:
        try:
            item = ReasonedScoreItem.model_validate(raw_item)
        except ValidationError:
            validation_notes.append(f"{field_name} 中存在非法结构项，已忽略。")
            continue

        reason = item.reason.strip()
        if not reason:
            validation_notes.append(f"{field_name} 中存在空理由项，已忽略。")
            continue

        if item.dimension and item.dimension not in expected_dimensions:
            validation_notes.append(
                f"{field_name} 使用了未知维度 [{item.dimension}]，已清空维度。"
            )
            item.dimension = ""

        evidence_ids = []
        evidence_texts = []
        for evidence_id in item.evidence_ids:
            evidence_id = evidence_id.strip()
            if not evidence_id:
                continue
            evidence = evidence_map.get(evidence_id)
            if evidence is None:
                validation_notes.append(
                    f"{field_name} 引用了不存在的证据 ID [{evidence_id}]，已忽略。"
                )
                continue
            evidence_ids.append(evidence_id)
            evidence_texts.append(evidence.evidence_text)

        if not evidence_ids:
            validation_notes.append(
                f"{field_name} 中的理由 [{reason}] 未绑定有效证据，已移除。"
            )
            continue

        key = (reason, item.dimension, tuple(evidence_ids))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            ReasonedScoreItem(
                reason=reason,
                dimension=item.dimension,
                evidence_ids=evidence_ids,
                evidence_texts=evidence_texts,
            )
        )

    return normalized


def _format_reason_details(items: Sequence[ReasonedScoreItem]) -> list[str]:
    details: list[str] = []
    for item in items:
        evidence_text = "；".join(item.evidence_texts[:3])
        if evidence_text:
            details.append(f"{item.reason}（证据: {evidence_text}）")
        else:
            details.append(item.reason)
    return details


def _collect_evidence_quotes(
    evidence_items: Sequence[EvidenceItem],
    used_items: Sequence[ReasonedScoreItem],
) -> list[str]:
    evidence_map = {item.id: item for item in evidence_items}
    quotes: list[str] = []
    seen = set()

    for reason_item in used_items:
        for evidence_id in reason_item.evidence_ids:
            evidence = evidence_map.get(evidence_id)
            if evidence is None or evidence.evidence_type != "quote":
                continue
            if evidence.evidence_text in seen:
                continue
            quotes.append(evidence.evidence_text)
            seen.add(evidence.evidence_text)

    for item in evidence_items:
        if item.evidence_type != "quote" or item.evidence_text in seen:
            continue
        quotes.append(item.evidence_text)
        seen.add(item.evidence_text)
        if len(quotes) >= 5:
            break

    return quotes[:5]


def _scale_scores_to_cap(scores: Dict[str, float], cap: float) -> Dict[str, float]:
    current_total = sum(scores.values())
    if current_total <= 0 or current_total <= cap:
        return scores

    scaled = {
        name: round(value * cap / current_total, 1)
        for name, value in scores.items()
    }
    diff = round(cap - sum(scaled.values()), 1)
    if scaled and diff != 0:
        first_dimension = next(iter(scaled))
        scaled[first_dimension] = round(max(scaled[first_dimension] + diff, 0.0), 1)
    return scaled


def apply_post_processing(
    raw_llm_result: Union[str, Dict[str, Any]],
    transcript: str,
    question: QuestionDefinition,
    evidence_packet: EvidenceExtractionPayload,
    visual_observation: str | None = None,
    extra_validation_notes: Sequence[str] | None = None,
) -> EvaluationResult:
    """把第二阶段评分结果整理成最终可信结果。"""

    parsed_result = _parse_raw_result(raw_llm_result)
    validation_notes = list(extra_validation_notes or [])
    evidence_map = {item.id: item for item in evidence_packet.evidence_items}
    expected_dimensions = {item.name for item in question.dimensions}

    try:
        scoring_payload = StageTwoScoringPayload.model_validate(parsed_result)
    except ValidationError:
        validation_notes.append("第二阶段评分结果结构非法，已按空结果处理。")
        scoring_payload = StageTwoScoringPayload()

    dimension_scores = _normalize_dimension_scores(
        raw_scores=scoring_payload.dimension_scores,
        question=question,
        validation_notes=validation_notes,
    )

    deduction_items = _normalize_reason_items(
        raw_items=scoring_payload.deduction_items,
        evidence_map=evidence_map,
        expected_dimensions=expected_dimensions,
        validation_notes=validation_notes,
        field_name="deduction_items",
    )
    bonus_items = _normalize_reason_items(
        raw_items=scoring_payload.bonus_items,
        evidence_map=evidence_map,
        expected_dimensions=expected_dimensions,
        validation_notes=validation_notes,
        field_name="bonus_items",
    )

    deduction_details = _format_reason_details(deduction_items)
    bonus_details = _format_reason_details(bonus_items)
    evidence_quotes = _collect_evidence_quotes(
        evidence_packet.evidence_items,
        [*deduction_items, *bonus_items],
    )

    matched_keywords = match_all_categories(transcript, question.model_dump())
    rule_based_scores, rule_notes = _compute_rule_based_dimension_scores(
        transcript=transcript,
        question=question,
        matched_keywords=matched_keywords,
    )
    if rule_based_scores:
        llm_total = round(sum(dimension_scores.values()), 1)
        rule_total = round(sum(rule_based_scores.values()), 1)
        if abs(rule_total - llm_total) >= 1.5:
            validation_notes.append(
                f"最终分数采用规则校准层纠偏：模型总分 {llm_total} -> 规则总分 {rule_total}。"
            )
        validation_notes.extend(rule_notes)
        dimension_scores = rule_based_scores
    rationale = str(scoring_payload.rationale or "").strip()
    if len(rationale) > settings.MAX_RATIONALE_CHARS:
        rationale = rationale[: settings.MAX_RATIONALE_CHARS].rstrip() + "..."
        validation_notes.append("rationale 过长，已自动截断。")

    computed_total = round(sum(dimension_scores.values()), 1)
    given_total = round(_to_float(scoring_payload.total_score, computed_total), 1)
    if abs(computed_total - given_total) > settings.SCORE_TOLERANCE:
        validation_notes.append(
            f"模型总分 {given_total} 与维度汇总 {computed_total} 不一致，已采用维度汇总。"
        )

    effective_length = _effective_text_length(transcript)
    full_score = question.fullScore
    total_score = computed_total

    if effective_length < settings.MIN_VALID_WORDS:
        cap = round(full_score * settings.MIN_WORDS_PENALTY_RATIO, 1)
        dimension_scores = _scale_scores_to_cap(dimension_scores, cap)
        total_score = round(sum(dimension_scores.values()), 1)
        deduction_details.append(
            f"作答有效长度不足 {settings.MIN_VALID_WORDS} 字，系统触发强制降分。"
        )
        rationale_prefix = (
            f"【系统降级判定】有效作答长度仅 {effective_length} 字，评分结果已按短答规则收敛。"
        )
        rationale = f"{rationale_prefix}{rationale}" if rationale else rationale_prefix
    else:
        total_score = computed_total

    total_score = round(sum(dimension_scores.values()), 1)
    total_score = round(min(total_score, full_score), 1)

    if not evidence_quotes:
        validation_notes.append("最终结果未保留到可核验的直接引语证据。")

    return EvaluationResult(
        question_id=question.id,
        question_type=question.type,
        transcript=transcript,
        visual_observation=visual_observation,
        evidence_items=evidence_packet.evidence_items,
        deduction_items=deduction_items,
        bonus_items=bonus_items,
        dimension_scores=dimension_scores,
        deduction_details=deduction_details,
        bonus_details=bonus_details,
        evidence_quotes=evidence_quotes,
        rationale=rationale,
        total_score=total_score,
        matched_keywords=matched_keywords,
        validation_notes=validation_notes,
    )
