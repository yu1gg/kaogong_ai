"""全局配置模块，安全管理密钥与环境变量、业务阈值"""
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# 初始化全局日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Settings(BaseSettings):
    """系统全局配置类"""
    
    # --- 基础配置 ---
    PROJECT_NAME: str = "公考面试AI测评系统"
    VERSION: str = "1.0.0"
    QUESTION_DB_PATH: str = "assets/mock/question.json"
    
    # --- 大模型 API 配置 (默认值可被 .env 文件覆盖) ---
    LLM_PROVIDER: str = "QWEN"
    LLM_API_KEY: str = ""  # 强制要求在 .env 中配置，如 LLM_API_KEY=sk-xxxx
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME: str = "qwen3-coder-plus"
    
    # --- ASR 语音识别性能调优配置 ---
    WHISPER_MODEL_SIZE: str = "base"  # 调优选项: "tiny", "base", "small"
    WHISPER_CPU_THREADS: int = 4      # 限制 PyTorch 占用的 CPU 核心数，防止打满服务器
    
    # --- 评分硬性业务规则配置 ---
    MIN_VALID_WORDS: int = 15         # 有效作答的最低字数阈值
    MIN_WORDS_PENALTY_RATIO: float = 0.2  # 字数不足时的惩罚系数 (最高只能拿满分的 20%)
    SCORE_TOLERANCE: float = 2.0      # 维度分与总分的容错差值
    
    # Pydantic V2 读取根目录下的 .env 文件标准写法
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例，避免重复读取硬盘"""
    return Settings()

settings = get_settings()