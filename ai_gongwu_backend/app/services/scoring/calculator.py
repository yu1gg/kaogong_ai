"""Deterministic post-processing for LLM evaluation outputs."""

import json
import logging
import re
from numbers import Number
from typing import Any, Dict, Iterable, Union

from app.core.config import settings
from app.models.schemas import EvaluationResult, QuestionDefinition

from .keyword_matcher import match_all_categories

logger = logging.getLogger(__name__)

DIRECT_QUOTE_PATTERN = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,40})[\"“”'‘’]")
DIRECT_SPEECH_HINTS = (
    "原话",
    "提到",
    "说到",
    "说了",
    "讲到",
    "使用",
    "出现",
    "口语",
    "措辞",
    "表述",
    "词语",
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


def _filter_unsupported_direct_quotes(
    details: list[str],
    transcript: str,
    validation_notes: list[str],
    field_name: str,
) -> list[str]:
    supported: list[str] = []
    for item in details:
        quoted_segments = DIRECT_QUOTE_PATTERN.findall(item)
        if quoted_segments and any(hint in item for hint in DIRECT_SPEECH_HINTS):
            if not all(segment in transcript for segment in quoted_segments):
                validation_notes.append(
                    f"{field_name} 中存在无法在原文核验的直接引语，已自动移除: {item}"
                )
                continue
        supported.append(item)
    return supported


def _validate_evidence_quotes(
    evidence_quotes: list[str],
    transcript: str,
    validation_notes: list[str],
) -> list[str]:
    supported_quotes: list[str] = []
    for quote in evidence_quotes[:3]:
        if quote in transcript:
            supported_quotes.append(quote)
        else:
            validation_notes.append(
                f"模型提供的证据引用无法在原文中命中，已忽略: {quote}"
            )
    return supported_quotes


def _scale_scores_to_cap(
    scores: Dict[str, float],
    cap: float,
) -> Dict[str, float]:
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
    visual_observation: str | None = None,
) -> EvaluationResult:
    """Normalize the LLM output into a stable API contract."""

    parsed_result = _parse_raw_result(raw_llm_result)
    validation_notes: list[str] = []

    dimension_scores = _normalize_dimension_scores(
        raw_scores=parsed_result.get("dimension_scores", {}),
        question=question,
        validation_notes=validation_notes,
    )

    deduction_details = _filter_unsupported_direct_quotes(
        _clean_string_list(parsed_result.get("deduction_details", [])),
        transcript,
        validation_notes,
        "deduction_details",
    )
    bonus_details = _filter_unsupported_direct_quotes(
        _clean_string_list(parsed_result.get("bonus_details", [])),
        transcript,
        validation_notes,
        "bonus_details",
    )
    evidence_quotes = _validate_evidence_quotes(
        _clean_string_list(parsed_result.get("evidence_quotes", [])),
        transcript,
        validation_notes,
    )

    matched_keywords = match_all_categories(transcript, question.model_dump())
    rationale = str(parsed_result.get("rationale", "")).strip()
    if len(rationale) > settings.MAX_RATIONALE_CHARS:
        rationale = rationale[: settings.MAX_RATIONALE_CHARS].rstrip() + "..."
        validation_notes.append("rationale 过长，已自动截断。")

    computed_total = round(sum(dimension_scores.values()), 1)
    given_total = round(_to_float(parsed_result.get("total_score"), computed_total), 1)
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

    total_score = round(min(total_score, full_score), 1)

    if not evidence_quotes:
        validation_notes.append("模型未提供可核验的原文证据引用。")

    return EvaluationResult(
        question_id=question.id,
        question_type=question.type,
        transcript=transcript,
        visual_observation=visual_observation,
        dimension_scores=dimension_scores,
        deduction_details=deduction_details,
        bonus_details=bonus_details,
        evidence_quotes=evidence_quotes,
        rationale=rationale,
        total_score=total_score,
        matched_keywords=matched_keywords,
        validation_notes=validation_notes,
    )
