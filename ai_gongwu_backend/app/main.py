"""FastAPI 应用启动总入口"""
"""FastAPI 应用启动总入口"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse # <-- 新增此行
from app.api.endpoints import interview


def create_app() -> FastAPI:
    """应用工厂模式构建 FastAPI 实例"""
    app = FastAPI(
        title="AI 公考面试测评系统 API",
        version="1.0.0",
        description="支持多模态音视频解析与结构化面试评分的后端引擎"
    )

    # 注册面试核心流程路由
    app.include_router(
        interview.router, 
        prefix="/api/v1/interview", 
        tags=["面试核心引擎"]
    )

# 新增：根路径自动跳转至 API 文档
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    # 新增：存活探针 (Health Check)
    @app.get("/health", tags=["系统监控"])
    async def health_check():
        return {"status": "ok", "service": "ai_gongwu_backend"}
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # uvicorn 启动服务器，开启热重载
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)