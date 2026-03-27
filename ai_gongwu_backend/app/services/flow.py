"""
面试业务流编排模块。
负责统筹媒体解析、大模型调用与后置分数计算的完整生命周期。
"""
import logging
from pathlib import Path
from typing import Dict, Any

from app.services.llm.client import LLMClient
from app.services.media.video_processor import process_video, process_audio
from app.services.scoring.prompts import build_evaluation_prompt
from app.services.scoring.calculator import apply_post_processing
from app.utils.data_loader import load_json_data
from app.core.config import settings

logger = logging.getLogger(__name__)

class InterviewFlowService:
    """面试流程服务类，隔离具体网络协议的纯业务逻辑。"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        # 预加载题库，直接从安全的配置中心读取路径
        self.question_data = load_json_data(settings.QUESTION_DB_PATH)

    def _extract_text_from_media(self, file_path: str) -> str:
        """媒体策略路由"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return process_video(file_path)
        elif ext in ['.mp3', '.wav', '.m4a']:
            return process_audio(file_path)
        else:
            # 容错：默认走视频逻辑
            return process_video(file_path)

    def _execute_evaluation_core(self, answer_text: str) -> Dict[str, Any]:
        """
        核心评估调度引擎 (DRY 原则重构)。
        统筹大模型推理与后处理干预，供多模态流与纯文本流复用。
        """
        # 1. 构建上下文与大模型交互
        prompt = build_evaluation_prompt(self.question_data, answer_text)
        raw_llm_result = self.llm_client.generate(prompt)
        
        if not raw_llm_result:
            raise RuntimeError("大模型评估引擎响应失败或解析 JSON 异常。")

        # 2. 业务规则后置干预与清洗
        final_result = apply_post_processing(
            raw_llm_result=raw_llm_result,
            answer=answer_text,
            question_data=self.question_data
        )

        return final_result

    def process_and_evaluate(self, file_path: str) -> Dict[str, Any]:
        """执行多模态核心评估工作流"""
        
        # 1. 多模态解析 (获得转录文本)
        answer_text = self._extract_text_from_media(file_path)
        if not answer_text.strip():
            raise ValueError("未能从媒体文件中提取到有效语音内容。")

        # 2. 调起核心评估引擎
        return self._execute_evaluation_core(answer_text)

    def evaluate_text_only(self, question_id: str, text_content: str) -> Dict[str, Any]:
        """
        纯文本评估链路 (旁路测试专用)。
        跳过媒体解析层，直接对接核心评估引擎。
        """
        logger.info(f"启动纯文本测评旁路, 目标题目 ID: {question_id}, 文本长度: {len(text_content)}")
        
        if not text_content.strip():
            raise ValueError("文本内容为空，无法进行评估。")
            
        # 注：当前架构通过 __init__ 加载单一题库。
        # 入参 question_id 预留用于后续多题目数据库检索的开闭原则 (OCP) 扩展。
        return self._execute_evaluation_core(text_content)