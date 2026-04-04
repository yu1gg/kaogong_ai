"""Tests for question bank loading and lookup."""

import unittest

from app.core.config import settings
from app.services.question_bank import QuestionBank, QuestionNotFoundError


class QuestionBankTestCase(unittest.TestCase):
    """Validate question bank behavior against the bundled mock dataset."""

    @classmethod
    def setUpClass(cls):
        cls.bank = QuestionBank(settings.QUESTION_DB_PATH)

    def test_get_existing_question(self):
        question = self.bank.get_question("HN-LX-20200606-01")
        self.assertEqual(question.id, "HN-LX-20200606-01")
        self.assertGreater(len(question.dimensions), 0)

    def test_missing_question_raises(self):
        with self.assertRaises(QuestionNotFoundError):
            self.bank.get_question("missing-id")


if __name__ == "__main__":
    unittest.main()
