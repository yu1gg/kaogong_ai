"""全局配置模块，安全管理密钥与环境变量"""
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# 初始化全局日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Settings(BaseSettings):
    """系统全局配置类"""
    PROJECT_NAME: str = "公考面试AI测评系统"
    VERSION: str = "1.0.0"
    
    # 大模型 API 配置 (默认值可被 .env 文件覆盖)
    LLM_PROVIDER: str = "QWEN"
    LLM_API_KEY: str = ""  # 强制要求在 .env 中配置，如 LLM_API_KEY=sk-xxxx
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME: str = "qwen3-coder-plus"
    
    # 系统路径配置
    QUESTION_DB_PATH: str = "assets/mock/question.json"
    
    # 读取根目录下的 .env 文件
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例，避免重复读取硬盘"""
    return Settings()

settings = get_settings()