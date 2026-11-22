"""
Esquemas Pydantic para validación estricta de entrada y serialización de salida.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """
    Representa una transacción con sus 30 características numéricas requeridas.
    """
    Time: float = Field(default=0.0, description="Segundos transcurridos desde la primera transacción")
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)
    Amount: float = Field(default=0.0, description="Monto monetario de la transacción")


class PredictionResult(BaseModel):
    """
    Resultado de la clasificación no supervisada con DBSCAN.
    """
    is_fraud: bool = Field(..., description="True si la distancia al core sample supera eps")
    label: str = Field(..., description="'FRAUDE' o 'LEGÍTIMA'")
    anomaly_score: float = Field(..., description="Proporción distancia / eps. Si es > 1.0 es anomalía")
    distance_to_nearest_core: float = Field(..., description="Distancia euclidiana mínima en espacio PCA")
    eps: float = Field(..., description="Radio de vecindad épsilon del modelo")


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionFeatures]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_count: int
    legitimate_count: int
    results: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    num_core_samples: int
    eps: float

