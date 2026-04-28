from fastapi import APIRouter
from app.schemas import HealthResponse
from app.model import is_model_loaded

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        version="1.0.0",
    )
