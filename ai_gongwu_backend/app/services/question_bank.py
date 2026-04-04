"""Question bank loading and lookup service."""

import logging
from pathlib import Path
from typing import Dict, Iterable, List

from pydantic import ValidationError

from app.core.config import settings
from app.models.schemas import QuestionDefinition
from app.utils.data_loader import load_json_data

logger = logging.getLogger(__name__)


class QuestionNotFoundError(ValueError):
    """Raised when a requested question id does not exist in the bank."""


class QuestionBank:
    """In-memory question repository backed by a JSON file."""

    def __init__(self, source_path: str | Path):
        self.source_path = settings.resolve_path(source_path)
        self._questions = self._load_questions()

    @property
    def count(self) -> int:
        """Total number of questions available."""

        return len(self._questions)

    def list_ids(self) -> List[str]:
        """Return all question ids."""

        return list(self._questions.keys())

    def get_question(self, question_id: str) -> QuestionDefinition:
        """Return a validated question definition by id."""

        if question_id not in self._questions:
            available_ids = ", ".join(self.list_ids()) or "无"
            raise QuestionNotFoundError(
                f"题目不存在: {question_id}。当前可用题目 ID: {available_ids}"
            )
        return self._questions[question_id]

    def _load_questions(self) -> Dict[str, QuestionDefinition]:
        raw_data = load_json_data(self.source_path)
        raw_questions = self._coerce_to_question_list(raw_data)

        questions: Dict[str, QuestionDefinition] = {}
        for raw_question in raw_questions:
            try:
                question = QuestionDefinition.model_validate(raw_question)
            except ValidationError as exc:
                raise ValueError(f"题库数据结构非法: {exc}") from exc

            if question.id in questions:
                raise ValueError(f"题库中存在重复的 question_id: {question.id}")
            questions[question.id] = question

        logger.info("题库加载完成: %s, 共 %s 题", self.source_path, len(questions))
        return questions

    @staticmethod
    def _coerce_to_question_list(raw_data: object) -> Iterable[object]:
        if isinstance(raw_data, list):
            return raw_data
        if isinstance(raw_data, dict):
            questions = raw_data.get("questions")
            if isinstance(questions, list):
                return questions
            return [raw_data]
        raise ValueError("题库 JSON 必须是对象、对象列表，或包含 questions 数组的对象。")
