"""题库服务测试。

测试不是越多越好，而是要优先覆盖“最容易出错的关键点”。
这里先验证两件基础事情：
1. 已存在的题目能正确取到
2. 不存在的题目会抛出预期异常
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import settings
from app.services.question_bank import QuestionBank, QuestionNotFoundError


class QuestionBankTestCase(unittest.TestCase):
    """验证题库加载与查询的基本行为。"""

    @classmethod
    def setUpClass(cls):
        # setUpClass 只在整个测试类开始前执行一次，适合做共享初始化。
        cls.bank = QuestionBank(settings.QUESTION_DB_PATH)

    def test_get_existing_question(self):
        # 验证：一个真实存在的 question_id 应该能拿到题目对象。
        question = self.bank.get_question("HN-LX-20200606-01")
        self.assertEqual(question.id, "HN-LX-20200606-01")
        self.assertGreater(len(question.dimensions), 0)

    def test_missing_question_raises(self):
        # 验证：不存在的题目应该抛出清晰异常，而不是返回 None 或直接崩溃。
        with self.assertRaises(QuestionNotFoundError):
            self.bank.get_question("missing-id")

    def test_can_load_questions_from_directory(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "q1.json").write_text(
                """
                {
                  "id": "Q1",
                  "type": "测试题",
                  "province": "河南",
                  "fullScore": 10,
                  "question": "问题1",
                  "dimensions": [{"name": "现象解读", "score": 10}]
                }
                """.strip(),
                encoding="utf-8",
            )
            nested_dir = temp_path / "nested"
            nested_dir.mkdir()
            (nested_dir / "q2.json").write_text(
                """
                {
                  "questions": [
                    {
                      "id": "Q2",
                      "type": "测试题",
                      "province": "河南",
                      "fullScore": 10,
                      "question": "问题2",
                      "dimensions": [{"name": "现象解读", "score": 10}]
                    }
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            bank = QuestionBank(temp_path)
            self.assertEqual(bank.count, 2)
            self.assertEqual(bank.list_ids(), ["Q1", "Q2"])
            self.assertEqual(bank.get_question("Q2").question, "问题2")


if __name__ == "__main__":
    unittest.main()
