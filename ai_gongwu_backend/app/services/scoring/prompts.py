"""
大模型提示词（Prompt）构建模块。
引入 System Override 与严苛的分数通胀压制机制，拉开评分梯队。
"""
import json
from typing import Dict, Any


def build_evaluation_prompt(question_data: Dict[str, Any], answer_text: str) -> str:
    """
    构建结构化面试评分的系统提示词，包含防误杀、反幻觉及严格的分数压制逻辑。
    """
    dimensions = question_data.get("dimensions", [])
    criteria = question_data.get("scoringCriteria", [])
    deductions = question_data.get("deductionRules", [])
    
    keywords = {
        "core": question_data.get("coreKeywords", []),
        "strong": question_data.get("strongKeywords", []),
        "penalty": question_data.get("penaltyKeywords", [])
    }
    
    prompt = f"""
    # Role
    你是一位具备极高政治素养、极其严苛的公考面试主考官。你的任务是精准区分考生的水平，绝不给平庸的答案打高分。你需要穿透字面规则，精准评估考生的真实思维逻辑与政治素养。

    # Input Data
    ## 题干
    {question_data.get("question", "")}
    
    ## 评分维度与标准 (满分 {question_data.get("fullScore", 30)})
    {json.dumps(dimensions, ensure_ascii=False, indent=2)}
    具体细则：{json.dumps(criteria, ensure_ascii=False)}
    
    ## 扣分规则
    {json.dumps(deductions, ensure_ascii=False)}
    
    ## 关键词库
    {json.dumps(keywords, ensure_ascii=False)}
    
    ## 考生作答记录 (100% 真实录音转录，以此为唯一事实依据)
    ====== 开始 ======
    {answer_text}
    ====== 结束 ======

    # System Override & Constraints (最高优先级指令，必须绝对服从)
    
    ## 第一原则：严厉打击分数通胀 (Anti-Inflation)
    大模型通常倾向于给出中间偏高的分数，你必须克服这一倾向！严格执行以下“断崖式降级”标准：
    1. 【语言降级阈值】：如果考生大量使用“我觉得”、“这事儿”、“搞搞别的”、“弄好点”、“硬着头皮”等市井口语，缺乏政府机关公文语境，其【语言逻辑与岗位适配】维度得分**绝对不得超过 2分（满分5分）**。
    2. 【对策降级阈值】：如果考生的对策极其笼统（如仅停留在“改一改”、“取消强制”、“培养网红”），缺乏具体的政务执行路径（如部门协同、机制建设），其【科学决策与措施】维度得分**绝对不得超过 3分（满分8分）**。
    3. 【扣分数学执行】：当考生触发《扣分规则》时（例如：未分析本质扣4-5分），你必须进行真实的数学减法！该维度得分 = 维度满分 - 扣分值。绝不允许在触发了扣分规则后，还给出该维度的及格分。

    ## 第二原则：防误杀与反幻觉 (Semantic Exemption & Anti-Hallucination)
    1. 语义豁免：若考生作答中出现“一刀切”、“形式主义”等词，且语境是在“剖析弊端、指出风险”，这是优秀的批判性思维！**绝对禁止以此作为扣分项**，必须在 bonus_details 中加分。仅当考生提倡“一刀切”做法时才扣分。
    2. 反捏造：在判定口语化或套话时，引用的词汇**必须在考生的作答记录中逐字原样存在**。找不到原文切实证据，则不得捏造扣分理由。

    ## 第三原则：柔性地域考核的前提条件
    “未充分结合省情”的豁免是有条件的：只有当考生给出了逻辑严密、专业术语丰富（如“长效机制”、“产业链带动”）的宏观对策时，才可以宽容其缺乏地域词汇。如果考生不仅没提省情，且全篇是大白话、对策极度肤浅，必须严格执行《扣分规则》中针对“脱离省情、泛泛而谈”的重度扣分（扣6-8分）。

    ## 第四原则：数据契约
    严禁生成任何解释性文本、思维链过程或 Markdown 标记，必须仅返回合法的 JSON 字符串。
    
    # Output Format
    请严格输出以下 JSON 结构：
    {{
        "dimension_scores": {{"维度名称": 数字得分}},
        "deduction_details": ["扣分原因明细1 (附扣减分值)", "扣分原因明细2"],
        "bonus_details": ["加分原因明细1", "加分原因明细2"],
        "rationale": "整体评价与逻辑依据",
        "total_score": 数字总分
    }}
    """
    return prompt.strip()