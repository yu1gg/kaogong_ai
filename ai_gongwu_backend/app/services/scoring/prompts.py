"""Prompt builders for the interview evaluation pipeline."""

import json

from app.models.schemas import QuestionDefinition


def build_evaluation_prompt(
    question: QuestionDefinition,
    answer_text: str,
    visual_observation: str | None = None,
) -> str:
    """Build a grounded, schema-driven evaluation prompt for the LLM."""

    dimensions = [{item.name: item.score} for item in question.dimensions]
    visual_block = visual_observation or "未提供视频流，无法评估仪态信息。"

    prompt = f"""
# 角色
你是一个只负责结构化评分的公务员面试评分引擎。你的任务不是自由发挥，而是基于给定题干、评分标准和考生作答原文，输出保守、可验证、可落地的 JSON 评分结果。

# 事实边界
1. 【考生作答原文】是内容评分的唯一事实来源。你不能捏造考生说过的话，不能把视觉观察改写成考生原话。
2. 【视觉观察】只能作为表达状态的弱补充，不能作为内容观点、关键词、扣分词、加分词的证据。
3. 如果你要引用考生原话，必须逐字出现在【考生作答原文】中，并同时写入 evidence_quotes。
4. 如果证据不足，宁可保守降分，也不要脑补细节。
5. “一刀切”“形式主义”等词只有在考生明确赞同该做法时才扣分；如果考生是在批评这类现象，不能因此扣分。

# 题目信息
题目 ID: {question.id}
题目类型: {question.type}
题干: {question.question}
满分: {question.fullScore}

# 评分维度
以下维度名称必须逐字照抄，不能新增、不能改写、不能翻译：
{json.dumps(dimensions, ensure_ascii=False, indent=2)}

# 评分细则
{json.dumps(question.scoringCriteria, ensure_ascii=False, indent=2)}

# 扣分规则
{json.dumps(question.deductionRules, ensure_ascii=False, indent=2)}

# 关键词参考
{json.dumps({
    "core": question.coreKeywords,
    "strong": question.strongKeywords,
    "bonus": question.bonusKeywords,
    "penalty": question.penaltyKeywords,
}, ensure_ascii=False, indent=2)}

# 视觉观察
{visual_block}

# 考生作答原文
===== 开始 =====
{answer_text}
===== 结束 =====

# 输出要求
1. 只返回合法 JSON，不要输出 Markdown，不要输出解释文字。
2. 每个维度分必须在对应满分范围内。
3. total_score 必须等于 dimension_scores 各维度得分之和。
4. deduction_details 和 bonus_details 只写有证据支撑的结论。
5. evidence_quotes 提供 1 到 3 条直接证据；如果没有可引用原文，返回空数组。
6. rationale 必须简洁，聚焦“为何得这个分”，不要扩写考生没说过的内容。

# JSON Schema
{{
  "dimension_scores": {{
    "维度名称": 0
  }},
  "deduction_details": ["扣分原因"],
  "bonus_details": ["加分原因"],
  "evidence_quotes": ["考生原话片段"],
  "rationale": "整体评价",
  "total_score": 0
}}
"""
    return prompt.strip()
