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
    """依赖注入提供者：实例化并组装业务流服务"""
    try:
        client = LLMClient()
        return InterviewFlowService(llm_client=client)
    except Exception as e:
        # 拦截注入层异常，防止服务级 500 崩溃
        raise HTTPException(status_code=503, detail=f"核心服务初始化失败 (可能缺失凭证): {str(e)}")


@router.post("/evaluate", summary="多模态面试作答测评接口")
async def evaluate_interview_submission(
        question_id: str = Form(..., description="目标测评题目的唯一标识符"),
        media_file: UploadFile = File(..., description="考生作答的音视频文件"),
        flow_service: InterviewFlowService = Depends(get_flow_service)
) -> Any:
    """
    接收前端上传的作答媒体流，执行多模态评估并返回结构化数据。
    """
    temp_path = None
    try:
        # 1. 防御性获取文件名，防止 NoneType 异常
        safe_filename = media_file.filename or "unknown.mp4"
        file_ext = os.path.splitext(safe_filename)[1]
        
        # 2. 将 I/O 操作移入 try 块内
        fd, temp_path = tempfile.mkstemp(suffix=file_ext)
        with os.fdopen(fd, "wb") as buffer:
            shutil.copyfileobj(media_file.file, buffer)

        # 3. 调度业务编排层
        result = flow_service.process_and_evaluate(temp_path)
        return {"code": 200, "message": "success", "data": result}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部处理异常: {str(e)}")
    finally:
        # 4. 安全释放资源
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass # 忽略清理阶段的系统级锁占用异常

@router.post("/evaluate/text", summary="纯文本面试作答测评接口 (旁路/测试专用)")
async def evaluate_text_submission(
        question_id: str = Form(..., description="目标测评题目的唯一标识符"),
        text_file: UploadFile = File(..., description="考生作答的纯文本文件 (.txt)"),
        flow_service: InterviewFlowService = Depends(get_flow_service)
) -> Any:
    """
    接收纯文本文件，剥离 ASR 与视觉解析前置依赖，直接推入后处理评分引擎。
    """
    # 1. 后缀与文件类型边界校验
    if not text_file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="非法输入：当前接口仅接受 .txt 格式的纯文本文件")

    try:
        # 2. 内存直读，避免落盘 I/O 开销
        content_bytes = await text_file.read()
        text_content = content_bytes.decode('utf-8').strip()

        if not text_content:
            raise HTTPException(status_code=400, detail="文本内容为空，触发硬性阻断")

        # 3. 调度业务编排层 (纯文本特化分支)
        # 假设 flow_service 已具备处理纯文本的特化方法
        result = flow_service.evaluate_text_only(question_id=question_id, text_content=text_content)
        
        return {"code": 200, "message": "success", "data": result}

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文本编码解析失败，请确保文件为 UTF-8 编码")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部评分引擎异常: {str(e)}")