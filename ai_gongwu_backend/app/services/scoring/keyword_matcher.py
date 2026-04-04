"""Deterministic keyword matching used as a lightweight grounding signal."""

from typing import Any, Dict, List


def _normalize_text(value: str) -> str:
    """Normalize whitespace and case for robust substring matching."""

    return "".join(value.lower().split())


def keyword_match(text: str, keywords: List[str]) -> List[str]:
    """Return deduplicated keywords that appear in the provided text."""

    normalized_text = _normalize_text(text)
    matched: List[str] = []
    seen = set()

    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword)
        if not normalized_keyword or normalized_keyword in seen:
            continue
        if normalized_keyword in normalized_text:
            matched.append(keyword)
            seen.add(normalized_keyword)

    return matched


def match_all_categories(text: str, question_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Match configured keyword groups against the transcript text."""

    categories = {
        "core": question_data.get("coreKeywords", []),
        "strong": question_data.get("strongKeywords", []),
        "weak": question_data.get("weakKeywords", []),
        "bonus": question_data.get("bonusKeywords", []),
    }
    return {category: keyword_match(text, keywords) for category, keywords in categories.items()}
