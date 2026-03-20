"""
大模型提示词（Prompt）构建模块。
采用结构化模板设计，约束大模型输出边界。
"""
import json
from typing import Dict, Any


def build_evaluation_prompt(question_data: Dict[str, Any], answer_text: str) -> str:
    """
    构建结构化面试评分的系统提示词。

    Args:
        question_data (Dict[str, Any]): 题目的结构化数据，包含维度、规则等。
        answer_text (str): 考生的作答转录文本。

    Returns:
        str: 组装完成的完整 Prompt 字符串。
    """
    # 提取并格式化评分维度与标准
    dimensions = question_data.get("dimensions", [])
    criteria = question_data.get("scoringCriteria", [])
    deductions = question_data.get("deductionRules", [])
    
    # 提取各类关键词库
    keywords = {
        "core": question_data.get("coreKeywords", []),
        "strong": question_data.get("strongKeywords", []),
        "penalty": question_data.get("penaltyKeywords", [])
    }
    
    prompt = f"""
    # Role
    你是一位严格的公考面试考官。请根据以下提供的标准，对考生的作答进行客观评分。

    # Input Data
    ## 题干
    {question_data.get("question", "")}
    
    ## 评分维度与标准 (满分 {question_data.get("fullScore", 30)})
    {json.dumps(dimensions, ensure_ascii=False, indent=2)}
    具体细则：{json.dumps(criteria, ensure_ascii=False)}
    
    ## 扣分规则
    {json.dumps(deductions, ensure_ascii=False)}
    
    ## 关键词库 (核心、强关联、扣分)
    {json.dumps(keywords, ensure_ascii=False)}
    
    ## 考生作答记录
    {answer_text}

    # Workflow & Constraints
    1. 严格对比考生作答与关键词库，未覆盖核心要素必须扣分。
    2. 触犯扣分规则或使用扣分关键词，必须在 deduction_details 中列出。
    3. 严禁生成任何解释性文本或 Markdown 标记，必须仅返回合法的 JSON 字符串。
    
    # Output Format
    请严格输出以下 JSON 结构：
    {{
        "dimension_scores": {{"维度名称": 数字得分}},
        "deduction_details": ["扣分原因1", "扣分原因2"],
        "bonus_details": ["加分原因"],
        "rationale": "整体评价",
        "total_score": 数字总分
    }}
    """
    return prompt.strip()