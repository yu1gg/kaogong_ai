"""Tests for deterministic post-processing of LLM output."""

import unittest

from app.core.config import settings
from app.models.schemas import QuestionDefinition
from app.services.scoring.calculator import apply_post_processing
from app.utils.data_loader import load_json_data


class ScoringCalculatorTestCase(unittest.TestCase):
    """Exercise score normalization and hallucination guardrails."""

    @classmethod
    def setUpClass(cls):
        raw_question = load_json_data(settings.resolve_path(settings.QUESTION_DB_PATH))
        cls.question = QuestionDefinition.model_validate(raw_question)

    def test_post_processing_normalizes_invalid_dimensions_and_quotes(self):
        transcript = "我认为要因地制宜，避免形式主义，不能一刀切，还要建立长效机制。"
        raw_result = {
            "dimension_scores": {
                "现象解读": 12,
                "language逻辑与岗位适配": 4,
                "创新思维": "1.5",
            },
            "deduction_details": [
                "考生说了“为了直播而直播”，说明存在明显偏差",
                "未充分结合河南实际",
            ],
            "bonus_details": ["引用原话“因地制宜”，思路较清晰"],
            "evidence_quotes": ["因地制宜", "不存在的原话"],
            "rationale": "整体表现尚可。" * 80,
            "total_score": 30,
        }

        result = apply_post_processing(
            raw_llm_result=raw_result,
            transcript=transcript,
            question=self.question,
        )

        self.assertIn("现象解读", result.dimension_scores)
        self.assertNotIn("language逻辑与岗位适配", result.dimension_scores)
        self.assertEqual(result.dimension_scores["现象解读"], 8.0)
        self.assertEqual(result.dimension_scores["创新思维"], 1.5)
        self.assertNotIn(
            "考生说了“为了直播而直播”，说明存在明显偏差",
            result.deduction_details,
        )
        self.assertEqual(result.evidence_quotes, ["因地制宜"])
        self.assertLessEqual(len(result.rationale), settings.MAX_RATIONALE_CHARS + 3)
        self.assertTrue(result.validation_notes)


if __name__ == "__main__":
    unittest.main()
