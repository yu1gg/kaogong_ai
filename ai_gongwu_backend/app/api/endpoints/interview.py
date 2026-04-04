"""HTTP endpoints for interview evaluation workflows."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.dependencies import get_flow_service
from app.models.schemas import EvaluationAPIResponse
from app.services.flow import InterviewFlowService
from app.services.question_bank import QuestionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/evaluate",
    summary="多模态面试作答测评接口",
    response_model=EvaluationAPIResponse,
)
async def evaluate_interview_submission(
    question_id: str = Form(..., description="目标测评题目的唯一标识符"),
    media_file: UploadFile = File(..., description="考生作答的音视频文件"),
    flow_service: InterviewFlowService = Depends(get_flow_service),
) -> Any:
    """Evaluate an uploaded audio or video submission."""

    temp_path = None
    safe_filename = media_file.filename or "unknown"

    try:
        flow_service.validate_media_suffix(safe_filename)

        file_suffix = Path(safe_filename).suffix.lower()
        fd, temp_path = tempfile.mkstemp(suffix=file_suffix)
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(media_file.file, buffer)

        result = await run_in_threadpool(
            flow_service.process_and_evaluate,
            question_id,
            temp_path,
        )
        return EvaluationAPIResponse(data=result)
    except QuestionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("处理多模态测评请求时发生异常")
        raise HTTPException(status_code=500, detail="服务器内部处理异常。") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("临时文件清理失败: %s", temp_path)


@router.post(
    "/evaluate/text",
    summary="纯文本面试作答测评接口",
    response_model=EvaluationAPIResponse,
)
async def evaluate_text_submission(
    question_id: str = Form(..., description="目标测评题目的唯一标识符"),
    text_file: UploadFile = File(..., description="考生作答的纯文本文件 (.txt)"),
    flow_service: InterviewFlowService = Depends(get_flow_service),
) -> Any:
    """Evaluate a plain-text answer without audio/video preprocessing."""

    filename = (text_file.filename or "").lower()
    if not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="当前接口仅接受 .txt 格式的纯文本文件。")

    try:
        content_bytes = await text_file.read()
        text_content = content_bytes.decode("utf-8-sig").strip()
        if not text_content:
            raise HTTPException(status_code=400, detail="文本内容为空，无法进行评估。")

        result = await run_in_threadpool(
            flow_service.evaluate_text_only,
            question_id,
            text_content,
        )
        return EvaluationAPIResponse(data=result)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="文本编码解析失败，请确保文件为 UTF-8 编码。") from exc
    except QuestionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("处理文本测评请求时发生异常")
        raise HTTPException(status_code=500, detail="内部评分引擎异常。") from exc
