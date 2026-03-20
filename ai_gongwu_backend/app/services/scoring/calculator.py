"""评分计算与后处理模块，对LLM输出进行校验和修正"""
import copy
import logging
from typing import Dict, Any, List
from .keyword_matcher import match_all_categories

logger = logging.getLogger(__name__)

def apply_post_processing(
    raw_llm_result: Dict[str, Any],
    answer: str,
    question_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    后处理：结合关键词匹配修正评分结果。
    采用深拷贝保证纯函数特性，不污染入参。
    """
    result = copy.deepcopy(raw_llm_result)
    
    # 1. 获取关键词匹配结果并合并
    kw_matches = match_all_categories(answer, question_data)
    if 'matched_keywords' not in result:
        result['matched_keywords'] = {}
        
    for cat, matched in kw_matches.items():
        if cat not in result['matched_keywords']:
            result['matched_keywords'][cat] = matched

    # 2. 根据扣分关键词补充扣分详情（如果缺失）
    penalty_list = kw_matches.get('penalty', [])
    if penalty_list:
        deduction_msg = f"触发硬性扣分关键词: {', '.join(penalty_list)}"
        if 'deduction_details' not in result:
            result['deduction_details'] = []
        if deduction_msg not in result['deduction_details']:
            result['deduction_details'].append(deduction_msg)

    # 3. 校验维度分数之和与总分
    dim_scores = result.get('dimension_scores', {})
    computed_total = sum(dim_scores.values())
    given_total = result.get('total_score', 0)

    # 设置容错阈值为 2 分
    if abs(computed_total - given_total) > 2.0:
        logger.warning(f"逻辑不一致: 维度分总和({computed_total})与模型总分({given_total})差异过大，强制采纳维度总和。")
        result['total_score'] = computed_total

    # 4. 确保总分不超过满分
    full_score = question_data.get('fullScore', 30.0)
    result['total_score'] = min(result['total_score'], full_score)

    return result