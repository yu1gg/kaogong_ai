"""Interview workflow orchestration service."""

import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import EvaluationResult, MediaExtractionResult
from app.services.llm.client import LLMClient
from app.services.media.video_processor import process_audio, process_video
from app.services.question_bank import QuestionBank
from app.services.scoring.calculator import apply_post_processing
from app.services.scoring.prompts import build_evaluation_prompt

logger = logging.getLogger(__name__)


class InterviewFlowService:
    """Coordinate media extraction, LLM scoring, and deterministic validation."""

    def __init__(self, llm_client: LLMClient, question_bank: QuestionBank):
        self.llm_client = llm_client
        self.question_bank = question_bank

    def validate_media_suffix(self, filename: str) -> None:
        """Ensure the uploaded media extension is supported."""

        suffix = Path(filename).suffix.lower()
        supported_extensions = (
            set(settings.SUPPORTED_VIDEO_EXTENSIONS)
            | set(settings.SUPPORTED_AUDIO_EXTENSIONS)
        )
        if suffix not in supported_extensions:
            supported = ", ".join(sorted(supported_extensions))
            raise ValueError(f"不支持的媒体格式: {suffix or '无后缀'}。支持格式: {supported}")

    def _extract_from_media(self, file_path: str) -> MediaExtractionResult:
        suffix = Path(file_path).suffix.lower()
        if suffix in settings.SUPPORTED_VIDEO_EXTENSIONS:
            return process_video(file_path)
        if suffix in settings.SUPPORTED_AUDIO_EXTENSIONS:
            return process_audio(file_path)
        raise ValueError(f"不支持的媒体格式: {suffix or '无后缀'}")

    def _execute_evaluation_core(
        self,
        question_id: str,
        extraction_result: MediaExtractionResult,
    ) -> EvaluationResult:
        question = self.question_bank.get_question(question_id)

        prompt = build_evaluation_prompt(
            question=question,
            answer_text=extraction_result.transcript,
            visual_observation=extraction_result.visual_observation,
        )
        raw_llm_result = self.llm_client.generate(prompt)
        if raw_llm_result is None:
            raise RuntimeError("大模型评估引擎响应失败或返回结果无法解析。")

        return apply_post_processing(
            raw_llm_result=raw_llm_result,
            transcript=extraction_result.transcript,
            question=question,
            visual_observation=extraction_result.visual_observation,
        )

    def process_and_evaluate(self, question_id: str, file_path: str) -> EvaluationResult:
        """Evaluate an uploaded audio/video submission."""

        extraction_result = self._extract_from_media(file_path)
        if not extraction_result.transcript.strip():
            raise ValueError("未能从媒体文件中提取到有效语音内容。")
        return self._execute_evaluation_core(question_id, extraction_result)

    def evaluate_text_only(self, question_id: str, text_content: str) -> EvaluationResult:
        """Evaluate a plain-text answer without the media preprocessing stage."""

        logger.info("启动纯文本测评旁路, question_id=%s, 文本长度=%s", question_id, len(text_content))
        if not text_content.strip():
            raise ValueError("文本内容为空，无法进行评估。")

        return self._execute_evaluation_core(
            question_id=question_id,
            extraction_result=MediaExtractionResult(
                transcript=text_content.strip(),
                source="text",
                visual_observation=None,
            ),
        )
