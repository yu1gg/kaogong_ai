"""Audio transcription service built around a lazily loaded Whisper model."""

import abc
import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_WHISPER_MODEL_CACHE = {}


class BaseTranscriber(abc.ABC):
    """Abstract speech-to-text contract."""

    @abc.abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe the provided audio file into text."""


class WhisperLocalTranscriber(BaseTranscriber):
    """Local Whisper implementation with process-level model caching."""

    def __init__(self, model_size: str):
        self.model_size = model_size
        if self.model_size not in _WHISPER_MODEL_CACHE:
            self._load_model()
        self.model = _WHISPER_MODEL_CACHE[self.model_size]

    def _load_model(self) -> None:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai-whisper 依赖，请先安装 requirements.txt 中的依赖。"
            ) from exc

        try:
            import torch

            torch.set_num_threads(max(settings.WHISPER_CPU_THREADS, 1))
        except ImportError:
            logger.warning("未安装 torch，无法设置 Whisper CPU 线程数。")

        logger.info(
            "首次初始化 Whisper 模型: %s (首次运行可能会自动下载权重)",
            self.model_size,
        )
        _WHISPER_MODEL_CACHE[self.model_size] = whisper.load_model(self.model_size)
        logger.info("Whisper 模型加载完成。")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        logger.info("ASR 开始转录: %s", audio_path)
        try:
            result = self.model.transcribe(
                audio_path,
                language=language or settings.WHISPER_LANGUAGE,
                fp16=False,
            )
        except Exception as exc:
            logger.error("Whisper 转录失败: %s", exc)
            raise RuntimeError(f"语音识别失败: {exc}") from exc

        text = result.get("text", "").strip()
        logger.info("ASR 转录完成，文本长度: %s", len(text))
        return text


def get_transcriber() -> BaseTranscriber:
    """Factory returning the configured transcriber implementation."""

    return WhisperLocalTranscriber(model_size=settings.WHISPER_MODEL_SIZE)
