"""
关键词匹配模块。
基于词法提供正向加分点召回，负向扣分已剥离至 LLM 语义层。
"""
from typing import List, Dict, Any


def keyword_match(text: str, keywords: List[str]) -> List[str]:
    """
    在文本中进行子串匹配召回正向关键词。
    （注：已移除英文边界符 \\b，以兼容中文语境下的无缝匹配）

    Args:
        text: 待匹配文本
        keywords: 关键词列表

    Returns:
        匹配到的关键词列表
    """
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        if not kw:
            continue
        # 中文语境直接使用 in 进行子串包含判定，提升匹配召回率
        if kw.lower() in text_lower:
            matched.append(kw)
    return matched


def match_all_categories(text: str, question_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    匹配所有正向/特征类别关键词。
    严禁在此处匹配 penalty (扣分词)，扣分逻辑已全部委托至大模型进行语义判别。

    Args:
        text: 考生答案
        question_data: 题目结构化数据

    Returns:
        Dict: 仅包含增益与特征维度的匹配结果
    """
    categories = {
        'core': question_data.get('coreKeywords', []),
        'strong': question_data.get('strongKeywords', []),
        'weak': question_data.get('weakKeywords', []),
        'bonus': question_data.get('bonusKeywords', [])
        # OCP原则：已剥离 'penalty'，交由下游 LLM 引擎处理上下文防误杀
    }
    result = {}
    for cat, kw_list in categories.items():
        result[cat] = keyword_match(text, kw_list)
    return result