"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.endpoints import interview
from app.core.config import settings
from app.core.dependencies import get_question_bank


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="支持多模态解析、结构化评分和评分结果校验的后端引擎",
    )

    app.include_router(
        interview.router,
        prefix="/api/v1/interview",
        tags=["面试核心引擎"],
    )

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["系统监控"])
    async def health_check():
        question_bank = get_question_bank()
        return {
            "status": "ok",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "question_count": question_bank.count,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
