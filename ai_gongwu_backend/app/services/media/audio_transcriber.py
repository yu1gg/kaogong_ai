"""
音频转录引擎模块。
基于 OpenAI Whisper 的本地化实现，支持模型常驻内存。
"""
import abc
import logging
import whisper
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# 全局模型缓存池，保证 FastAPI 生命周期内只加载一次模型
_WHISPER_MODEL_CACHE = {}

class BaseTranscriber(abc.ABC):
    """语音识别抽象基类"""
    @abc.abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        pass


class WhisperLocalTranscriber(BaseTranscriber):
    """本地 Whisper 实现类 (单例模式变体)"""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        
        # 懒加载：只有在第一次调用时才将模型读入内存
        if self.model_size not in _WHISPER_MODEL_CACHE:
            logger.info(f"首次初始化：正在加载 Whisper '{self.model_size}' 模型 (首次运行会自动下载权重文件)...")
            # 默认加载模型。如果在纯 CPU 服务器上运行，后续推理会自动使用 CPU
            _WHISPER_MODEL_CACHE[self.model_size] = whisper.load_model(self.model_size)
            logger.info("Whisper 模型加载至内存完毕！")
            
        self.model = _WHISPER_MODEL_CACHE[self.model_size]

    def transcribe(self, audio_path: str, language: Optional[str] = "zh") -> str:
        """
        执行音频转文字。
        明确指定 language="zh" 可以省去模型自动语种探测的时间，提升速度。
        """
        logger.info(f"ASR 引擎开始转录: {audio_path}")
        try:
            # fp16=False 是为了避免纯 CPU 环境下 PyTorch 报半精度警告
            result = self.model.transcribe(audio_path, language=language, fp16=False)
            text = result["text"].strip()
            logger.info(f"ASR 转录完成，字数: {len(text)}")
            return text
        except Exception as e:
            logger.error(f"Whisper 转录过程发生异常: {str(e)}")
            raise RuntimeError(f"语音识别失败: {str(e)}")


def get_transcriber() -> BaseTranscriber:
    """
    ASR 引擎工厂方法。
    目前写死为本地 Whisper，未来如果想切回方案 A (阿里云)，只需在这里改一行代码即可。
    """
    # 默认使用 base 模型，兼顾了准确率与 CPU 推理速度 (模型大小约 140MB)
    return WhisperLocalTranscriber(model_size="base")