# 采分点匹配
"""关键词匹配模块，用于在答案中检测各类关键词"""
import re
from typing import List, Dict, Any


def keyword_match(text: str, keywords: List[str]) -> List[str]:
    """
    在文本中匹配关键词（不区分大小写，考虑单词边界）

    Args:
        text: 待匹配文本
        keywords: 关键词列表

    Returns:
        匹配到的关键词列表
    """
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        # 使用正则确保单词边界，防止部分匹配
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(kw)
    return matched


def match_all_categories(text: str, question_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    匹配所有类别关键词

    Args:
        text: 考生答案
        question_data: 题目数据，应包含以下键：
            - coreKeywords
            - strongKeywords
            - weakKeywords
            - bonusKeywords
            - penaltyKeywords

    Returns:
        字典，键为类别名（'core', 'strong', 'weak', 'bonus', 'penalty'），值为匹配到的关键词列表
    """
    categories = {
        'core': question_data.get('coreKeywords', []),
        'strong': question_data.get('strongKeywords', []),
        'weak': question_data.get('weakKeywords', []),
        'bonus': question_data.get('bonusKeywords', []),
        'penalty': question_data.get('penaltyKeywords', [])
    }
    result = {}
    for cat, kw_list in categories.items():
        result[cat] = keyword_match(text, kw_list)
    return result