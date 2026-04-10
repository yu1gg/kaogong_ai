"""业务编排服务。

如果把整个后端想象成一条流水线，这个文件就是“总调度台”。
它不关心 HTTP 细节，也不直接处理数据库，而是专门负责串联核心步骤：

1. 找到题目
2. 解析文本 / 音频 / 视频
3. 生成 Prompt 调模型
4. 对模型结果做后处理和校验

这层设计的好处是：API 层保持轻，业务逻辑更集中，也更容易测试。
"""

import json
import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import EvaluationResult, MediaExtractionResult
from app.services.evaluation_store import EvaluationStore
from app.services.llm.client import LLMClient
from app.services.media.video_processor import process_audio, process_video
from app.services.question_bank import QuestionBank
from app.services.scoring.calculator import apply_post_processing, prepare_evidence_packet
from app.services.scoring.prompts import (
    EVIDENCE_EXTRACTION_SYSTEM_MESSAGE,
    EVIDENCE_SCORING_SYSTEM_MESSAGE,
    build_evidence_extraction_prompt,
    build_evidence_scoring_prompt,
)

logger = logging.getLogger(__name__)


class InterviewFlowService:
    """统筹媒体解析、模型评分和确定性校验的核心服务。"""

    def __init__(
        self,
        llm_client: LLMClient,
        question_bank: QuestionBank,
        evaluation_store: EvaluationStore,
    ):
        self.llm_client = llm_client
        self.question_bank = question_bank
        self.evaluation_store = evaluation_store

    def validate_media_suffix(self, filename: str) -> None:
        """检查上传文件后缀是否在支持列表中。"""

        suffix = Path(filename).suffix.lower()
        supported_extensions = (
            set(settings.SUPPORTED_VIDEO_EXTENSIONS)
            | set(settings.SUPPORTED_AUDIO_EXTENSIONS)
        )
        if suffix not in supported_extensions:
            supported = ", ".join(sorted(supported_extensions))
            raise ValueError(f"不支持的媒体格式: {suffix or '无后缀'}。支持格式: {supported}")

    def _extract_from_media(self, file_path: str) -> MediaExtractionResult:
        """根据文件后缀选择合适的解析路径。

        视频走：提取音频 + ASR + 视觉分析
        音频走：直接 ASR
        """
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
        *,
        persist: bool = True,
    ) -> EvaluationResult:
        """执行评分主流程。

        这里是整个系统最核心的“内容评分链路”：
        1. 根据题目 ID 拿到题目定义
        2. 第一阶段抽取证据
        3. 第二阶段仅基于证据评分
        4. 用确定性规则修正和校验模型输出
        """

        question = self.question_bank.get_question(question_id)

        evidence_prompt = build_evidence_extraction_prompt(
            question=question,
            answer_text=extraction_result.transcript,
            visual_observation=extraction_result.visual_observation,
        )
        evidence_generation = self.llm_client.generate(
            evidence_prompt,
            system_message=EVIDENCE_EXTRACTION_SYSTEM_MESSAGE,
        )
        if evidence_generation is None:
            logger.warning("第一阶段证据抽取失败，已回退到规则型证据整理。")
            evidence_packet, evidence_notes = prepare_evidence_packet(
                raw_llm_result={},
                transcript=extraction_result.transcript,
                question=question,
            )
            evidence_notes.insert(0, "第一阶段证据抽取失败，已回退到规则型证据整理。")
            evidence_raw_content = ""
            evidence_raw_payload = {}
        else:
            evidence_packet, evidence_notes = prepare_evidence_packet(
                raw_llm_result=evidence_generation.parsed_payload,
                transcript=extraction_result.transcript,
                question=question,
            )
            evidence_raw_content = evidence_generation.raw_content
            evidence_raw_payload = evidence_generation.parsed_payload

        scoring_prompt = build_evidence_scoring_prompt(
            question=question,
            evidence_packet=evidence_packet,
        )
        scoring_generation = self.llm_client.generate(
            scoring_prompt,
            system_message=EVIDENCE_SCORING_SYSTEM_MESSAGE,
        )
        if scoring_generation is None:
            raise RuntimeError("第二阶段证据评分失败或返回结果无法解析。")

        final_result = apply_post_processing(
            raw_llm_result=scoring_generation.parsed_payload,
            transcript=extraction_result.transcript,
            question=question,
            evidence_packet=evidence_packet,
            visual_observation=extraction_result.visual_observation,
            extra_validation_notes=evidence_notes,
        )
        final_result = final_result.model_copy(
            update={
                "source": extraction_result.source,
                "source_filename": extraction_result.source_filename,
            }
        )

        if not persist:
            return final_result

        return self.evaluation_store.save_evaluation(
            question_id=question.id,
            question_type=question.type,
            source=extraction_result.source,
            source_filename=extraction_result.source_filename,
            transcript=extraction_result.transcript,
            visual_observation=extraction_result.visual_observation,
            prompt_text=json.dumps(
                {
                    "stage_one_prompt": evidence_prompt,
                    "stage_two_prompt": scoring_prompt,
                },
                ensure_ascii=False,
            ),
            llm_provider=self.llm_client.provider,
            llm_model_name=self.llm_client.model_name,
            raw_llm_content=json.dumps(
                {
                    "stage_one_raw": evidence_raw_content,
                    "stage_two_raw": scoring_generation.raw_content,
                },
                ensure_ascii=False,
            ),
            raw_llm_payload={
                "stage_one": evidence_raw_payload,
                "validated_evidence": evidence_packet.model_dump(mode="json"),
                "stage_two": scoring_generation.parsed_payload,
            },
            final_result=final_result,
        )

    def process_and_evaluate(
        self,
        question_id: str,
        file_path: str,
        source_filename: str | None = None,
        *,
        persist: bool = True,
    ) -> EvaluationResult:
        """处理音频/视频类提交。"""

        # 先把媒体转成统一结构。
        extraction_result = self._extract_from_media(file_path).model_copy(
            update={"source_filename": source_filename}
        )
        if not extraction_result.transcript.strip():
            raise ValueError("未能从媒体文件中提取到有效语音内容。")
        return self._execute_evaluation_core(
            question_id,
            extraction_result,
            persist=persist,
        )

    def evaluate_text_only(
        self,
        question_id: str,
        text_content: str,
        source_filename: str | None = None,
        *,
        persist: bool = True,
    ) -> EvaluationResult:
        """处理纯文本提交通道。

        这个接口很适合：
        - 快速调试评分逻辑
        - 对比不同 Prompt 效果
        - 不依赖 Whisper / ffmpeg 的轻量测试
        """

        logger.info("启动纯文本测评旁路, question_id=%s, 文本长度=%s", question_id, len(text_content))
        if not text_content.strip():
            raise ValueError("文本内容为空，无法进行评估。")

        return self._execute_evaluation_core(
            question_id=question_id,
            extraction_result=MediaExtractionResult(
                transcript=text_content.strip(),
                source="text",
                source_filename=source_filename,
                visual_observation=None,
            ),
            persist=persist,
        )
