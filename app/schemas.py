from pydantic import BaseModel, Field
from typing import Optional


class NodePredictRequest(BaseModel):
    node_id: str = Field(..., examples=["102503180"])
    water_level: Optional[int] = Field(0, ge=0, le=3, description="Current sensor water level (0-3)")
    rain_1day: Optional[float] = Field(10.0, ge=0)
    rain_3day: Optional[float] = Field(25.0, ge=0)
    rain_5day: Optional[float] = Field(40.0, ge=0)
    rain_7day: Optional[float] = Field(55.0, ge=0)
    rain_avg: Optional[float] = Field(9.0, ge=0)
    elevation: Optional[float] = Field(15.0)
    slope: Optional[float] = Field(3.0)
    wind_speed: Optional[float] = Field(3.0, ge=0)
    storm_intensity: Optional[float] = Field(0.1, ge=0, le=1)


class NodePredictResponse(BaseModel):
    node_id: str
    predicted_level: int
    probability: float
    risk_label: str
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
