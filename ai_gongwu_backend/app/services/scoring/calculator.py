"""
评分计算与后处理模块。
负责大模型输出的 JSON 解析容错、结合关键词匹配修正结果，以及执行硬性业务降级。
"""
import copy
import json
import logging
from typing import Dict, Any, Union

from .keyword_matcher import match_all_categories
from app.core.config import settings

logger = logging.getLogger(__name__)

def apply_post_processing(
    raw_llm_result: Union[str, Dict[str, Any]], 
    answer: str, 
    question_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    后处理：解析输出 -> 提取关键词 -> 容错校验 -> 硬性规则阻断。
    """
    # 1. 安全解析与清洗 (兼容大模型返回字符串或已解析字典的情况)
    if isinstance(raw_llm_result, str):
        try:
            clean_str = raw_llm_result.strip().strip('`').removeprefix('json').strip()
            parsed_result = json.loads(clean_str)
        except json.JSONDecodeError:
            logger.error(f"大模型返回的非标准 JSON: {raw_llm_result}")
            return {
                "dimension_scores": {},
                "deduction_details": ["系统未能成功解析模型评分"],
                "bonus_details": [],
                "rationale": "系统判定出现波动，请重试或联系人工复核。",
                "total_score": 0,
                "matched_keywords": {}
            }
    else:
        parsed_result = raw_llm_result
        
    # 深拷贝保证纯函数特性
    result = copy.deepcopy(parsed_result)
    
    # 2. 获取关键词匹配结果并合并 (保留您原有的严谨逻辑)
    kw_matches = match_all_categories(answer, question_data)
    if 'matched_keywords' not in result:
        result['matched_keywords'] = {}
        
    for cat, matched in kw_matches.items():
        if cat not in result['matched_keywords']:
            result['matched_keywords'][cat] = matched

    # 3. 根据扣分关键词补充扣分详情
    penalty_list = kw_matches.get('penalty', [])
    if penalty_list:
        deduction_msg = f"触发硬性扣分关键词: {', '.join(penalty_list)}"
        if 'deduction_details' not in result:
            result['deduction_details'] = []
        if deduction_msg not in result['deduction_details']:
            result['deduction_details'].append(deduction_msg)

    # 4. 校验维度分数之和与总分 (接入全局配置)
    dim_scores = result.get('dimension_scores', {})
    computed_total = sum(dim_scores.values())
    given_total = result.get('total_score', 0)

    if abs(computed_total - given_total) > settings.SCORE_TOLERANCE:
        logger.warning(f"逻辑不一致: 维度分总和({computed_total})与模型总分({given_total})差异过大，强制采纳维度总和。")
        result['total_score'] = computed_total

    # 5. 业务硬性规则阻断：作答字数过少拦截 (接入全局配置)
    full_score = question_data.get('fullScore', 30.0)
    if len(answer) < settings.MIN_VALID_WORDS:
        result["rationale"] = f"【系统降级判定】考生作答内容仅 {len(answer)} 字，无法进行有效评估。" + result.get("rationale", "")
        result['total_score'] = min(result['total_score'], full_score * settings.MIN_WORDS_PENALTY_RATIO)
        result.setdefault("deduction_details", []).append(f"作答字数不足 {settings.MIN_VALID_WORDS} 字，触发强制降分。")

    # 6. 确保总分不超过满分并保留小数
    result['total_score'] = round(min(result['total_score'], full_score), 1)

    return result