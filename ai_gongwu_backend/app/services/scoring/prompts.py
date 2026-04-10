"""两阶段评分 Prompt 构造模块。"""

import json

from app.models.schemas import EvidenceExtractionPayload, QuestionDefinition

EVIDENCE_EXTRACTION_SYSTEM_MESSAGE = (
    "你是一个严格的证据抽取引擎。"
    "你的任务只是抽取可核验的证据，不负责打分。"
    "只输出合法 JSON。"
)

EVIDENCE_SCORING_SYSTEM_MESSAGE = (
    "你是一个严格的结构化评分引擎。"
    "你只能基于给定证据包打分，不能自行补充证据。"
    "只输出合法 JSON。"
)


def build_evidence_extraction_prompt(
    question: QuestionDefinition,
    answer_text: str,
    visual_observation: str | None = None,
) -> str:
    """第一阶段：从原文中抽取可核验的证据。"""

    dimensions = [{item.name: item.score} for item in question.dimensions]
    visual_block = visual_observation or "未提供视频流，无法评估仪态信息。"

    prompt = f"""
# 角色
你现在不是评分官，而是“证据抽取员”。你只能从考生原文中抽取逐字可核验的证据。

# 任务规则
1. 只能抽取在【考生作答原文】中逐字存在的内容。
2. 不要给分，不要写抽象结论，不要写“未结合河南省情”这类缺失型判断。
3. 证据文本 evidence_text 必须是原文中的直接片段。
4. claim 只描述这条证据说明了什么，但不能脱离 evidence_text 自由发挥。
5. 如果是明显口语表达，也可以抽取为语言类证据。
6. 视觉观察不是内容评分证据来源，不得作为 evidence_text。

# 题目信息
题目 ID: {question.id}
题目类型: {question.type}
题干: {question.question}
评分维度:
{json.dumps(dimensions, ensure_ascii=False, indent=2)}

# 视觉观察
{visual_block}

# 考生作答原文
===== 开始 =====
{answer_text}
===== 结束 =====

# 输出要求
1. 只输出 JSON。
2. evidence_items 最多输出 10 条，优先保留最关键的内容和语言证据。
3. dimension_hint 必须从已有评分维度中选择，拿不准时可填空字符串。
4. stance 只能是 positive / negative / language / neutral。
5. evidence_type 固定为 quote。

# JSON Schema
{{
  "evidence_items": [
    {{
      "id": "",
      "dimension_hint": "现象解读",
      "claim": "考生指出该做法存在一刀切问题",
      "evidence_text": "最关键的就是“一刀切”太不合理了",
      "evidence_type": "quote",
      "stance": "negative"
    }}
  ],
  "coverage_notes": ["证据覆盖到积极面、问题面、措施面"],
  "summary": "原文中主要证据概览"
}}
"""
    return prompt.strip()


def build_evidence_scoring_prompt(
    question: QuestionDefinition,
    evidence_packet: EvidenceExtractionPayload,
) -> str:
    """第二阶段：只基于证据包打分。"""

    dimensions = [{item.name: item.score} for item in question.dimensions]

    prompt = f"""
# 角色
你现在是“证据约束评分官”。你只能基于给定证据包打分。

# 核心规则
1. 禁止引用证据包以外的新事实。
2. deduction_items 和 bonus_items 的每一条都必须绑定 evidence_ids，且 evidence_ids 不能为空。
3. 若某项扣分来自“缺失”，也必须引用证据包中的 absence 类证据。
4. 不允许输出 deduction_details / bonus_details 自由文本，只输出结构化 items。
5. total_score 必须等于 dimension_scores 之和。

# 评分顺序
1. 先看“通用内容质量”，判断考生是否识别积极意义、问题本质、危害根源、改进措施。
2. 再看“河南省情 + 省直机关岗位适配度”，判断是否体现本土化、岗位化、政策化表达。
3. 如果答案通用分析完整，但缺少河南省情或省直机关视角，应判为“中上档降档”，不能直接打入低分区。
4. 语言口语化会拉低“语言逻辑与岗位适配”，但不能吞掉已经被证据支持的分析分和措施分。

# 分档锚点
1. 24-30 分：河南省直高分。内容完整，且明显体现河南省情、省委精神、省直机关统筹协调和分类指导视角。
2. 20-23 分：通用高分。内容完整、分析较深、措施较实，但河南本土化或岗位化不足。
3. 16-19 分：基本合格。能答到主要问题，但深度、结构、措施或表达存在明显短板。
4. 0-15 分：明显偏弱。内容单薄、逻辑散乱、措施很少，或口语化严重且缺乏有效分析。

# 题目信息
题目 ID: {question.id}
题目类型: {question.type}
题干: {question.question}
满分: {question.fullScore}
评分维度:
{json.dumps(dimensions, ensure_ascii=False, indent=2)}
评分细则:
{json.dumps(question.scoringCriteria, ensure_ascii=False, indent=2)}
扣分规则:
{json.dumps(question.deductionRules, ensure_ascii=False, indent=2)}

# 证据包
{json.dumps(evidence_packet.model_dump(mode="json"), ensure_ascii=False, indent=2)}

# 输出要求
1. 只输出 JSON。
2. dimension_scores 的维度名称必须与题库完全一致。
3. 若证据不足，宁可保守给分，不要脑补。
4. 抽象理由必须绑定 evidence_ids。

# JSON Schema
{{
  "dimension_scores": {{
    "维度名称": 0
  }},
  "deduction_items": [
    {{
      "reason": "未结合河南省情展开分析",
      "dimension": "现象解读",
      "evidence_ids": ["A1"]
    }}
  ],
  "bonus_items": [
    {{
      "reason": "准确识别形式主义和一刀切风险",
      "dimension": "危害根源分析",
      "evidence_ids": ["E2"]
    }}
  ],
  "rationale": "整体评分理由",
  "total_score": 0
}}
"""
    return prompt.strip()
