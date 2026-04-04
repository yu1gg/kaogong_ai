"""Dependency providers shared across FastAPI routes."""

from functools import lru_cache

from app.core.config import settings
from app.services.flow import InterviewFlowService
from app.services.llm.client import LLMClient
from app.services.question_bank import QuestionBank


@lru_cache()
def get_llm_client() -> LLMClient:
    """Return a shared LLM client instance."""

    return LLMClient()


@lru_cache()
def get_question_bank() -> QuestionBank:
    """Return a shared in-memory question bank."""

    return QuestionBank(settings.QUESTION_DB_PATH)


@lru_cache()
def get_flow_service() -> InterviewFlowService:
    """Return a shared workflow service."""

    return InterviewFlowService(
        llm_client=get_llm_client(),
        question_bank=get_question_bank(),
    )
