"""
面试测评 API 路由模块。
对外暴露网络接口，处理文件 I/O 与 HTTP 异常态转化。
"""
import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Any

from app.services.flow import InterviewFlowService
from app.services.llm.client import LLMClient

router = APIRouter()


def get_flow_service() -> InterviewFlowService:
    """
    依赖注入提供者：实例化并组装业务流服务。
    便于在单元测试中通过 app.dependency_overrides 替换为 Mock 服务。
    """
    client = LLMClient()
    return InterviewFlowService(llm_client=client)


@router.post("/evaluate", summary="多模态面试作答测评接口")
async def evaluate_interview_submission(
        question_id: str = Form(..., description="目标测评题目的唯一标识符"),
        media_file: UploadFile = File(..., description="考生作答的音视频文件"),
        flow_service: InterviewFlowService = Depends(get_flow_service)
) -> Any:
    """
    接收小程序前端上传的作答媒体流，执行多模态评估并返回雷达图诊断数据。
    """
    # 采用安全临时目录，避免并发下文件命名冲突
    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(media_file.filename)[1])

    try:
        # 1. 异步落盘，释放内存
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(media_file.file, buffer)

        # 2. 调度纯业务逻辑层进行运算
        # 注意：若 process_and_evaluate 包含重度 CPU 计算，可考虑丢入 BackgroundTasks 或 ProcessPool
        result = flow_service.process_and_evaluate(temp_path)
        return {"code": 200, "message": "success", "data": result}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部处理异常: {str(e)}")
    finally:
        # 3. 确保存储空间释放 (极高可维护性保障)
        if os.path.exists(temp_path):
            os.remove(temp_path)