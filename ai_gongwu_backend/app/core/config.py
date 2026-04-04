"""Application settings and shared configuration helpers."""

from functools import lru_cache
import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Settings(BaseSettings):
    """Global settings resolved from environment variables and defaults."""

    PROJECT_NAME: str = "公考面试AI测评系统"
    VERSION: str = "1.1.0"
    QUESTION_DB_PATH: str = "assets/mock/question.json"

    LLM_PROVIDER: str = "QWEN"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME: str = "qwen3-coder-plus"
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000
    LLM_FORCE_JSON_RESPONSE: bool = True

    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_CPU_THREADS: int = 4
    WHISPER_LANGUAGE: str = "zh"
    ENABLE_VISUAL_ANALYSIS: bool = True

    MIN_VALID_WORDS: int = 15
    MIN_WORDS_PENALTY_RATIO: float = 0.2
    SCORE_TOLERANCE: float = 2.0
    MAX_RATIONALE_CHARS: int = 400

    SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".m4a")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        """Return the backend project root path."""

        return Path(__file__).resolve().parents[2]

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a path against the backend root while supporting absolute paths."""

        candidate = Path(value)
        if candidate.is_absolute():
            return candidate

        rooted_candidate = self.project_root / candidate
        if rooted_candidate.exists():
            return rooted_candidate

        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate

        return rooted_candidate


@lru_cache()
def get_settings() -> Settings:
    """Return a cached settings singleton."""

    return Settings()


settings = get_settings()
