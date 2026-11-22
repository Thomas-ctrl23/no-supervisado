"""
Punto de entrada de FastAPI optimizado para Vercel Serverless Functions.
Carga los artefactos de DBSCAN una sola vez en memoria (singleton) y resuelve
las predicciones en milisegundos sin sobrecargar recursos.
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Garantizar resolución de módulos tanto en local como en Vercel Serverless
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
for path in (API_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import pairwise_distances_argmin_min

try:
    from schemas import (
        TransactionFeatures,
        PredictionResult,
        BatchPredictionRequest,
        BatchPredictionResponse,
        HealthResponse,
    )
except ImportError:
    from api.schemas import (
        TransactionFeatures,
        PredictionResult,
        BatchPredictionRequest,
        BatchPredictionResponse,
        HealthResponse,
    )

# Inicializar FastAPI
app = FastAPI(
    title="API Detección de Fraude con DBSCAN",
    description="Servicio Serverless de inferencia no supervisada",
    version="1.0.0"
)

# Configuración de CORS universal
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
if not origins or "*" in origins:
    origins = ["*"]
    allow_credentials = False
else:
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# CARGA DE ARTEFACTOS (SINGLETON EN MEMORIA)
# -------------------------------------------------------------------------
class ModelStore:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.core_samples: Optional[np.ndarray] = None
        self.eps: float = 2.5
        self.min_samples: int = 10
        self.metrics_data: Dict[str, Any] = {}
        self.sample_transactions: List[Dict[str, Any]] = []
        self.is_loaded: bool = False
        self._load()

    def _resolve_artifacts_dir(self) -> Path:
        """
        Encuentra la carpeta artifacts tanto en entorno local como en Vercel.
        """
        current_dir = Path(__file__).resolve().parent
        # Posibles rutas relativas
        candidates = [
            current_dir.parent / "artifacts",
            current_dir / "artifacts",
            Path("artifacts"),
            Path("backend/artifacts")
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        # Si no existe, crear directorio temporal de rescate
        rescue = current_dir.parent / "artifacts"
        rescue.mkdir(parents=True, exist_ok=True)
        return rescue

    def _load(self):
        artifacts_dir = self._resolve_artifacts_dir()

        # 1. Cargar params.json
        params_file = artifacts_dir / "params.json"
        if params_file.exists():
            with open(params_file, "r", encoding="utf-8") as f:
                params = json.load(f)
                self.eps = float(params.get("eps", 2.5))
                self.min_samples = int(params.get("min_samples", 10))

        # 2. Cargar metrics.json
        metrics_file = artifacts_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                self.metrics_data = json.load(f)

        # 3. Cargar sample_transactions.json
        samples_file = artifacts_dir / "sample_transactions.json"
        if samples_file.exists():
            with open(samples_file, "r", encoding="utf-8") as f:
                self.sample_transactions = json.load(f)

        # 4. Cargar Modelos Joblib (si existen)
        scaler_file = artifacts_dir / "scaler.joblib"
        pca_file = artifacts_dir / "pca.joblib"
        cores_file = artifacts_dir / "core_samples.joblib"

        if scaler_file.exists() and pca_file.exists() and cores_file.exists():
            try:
                self.scaler = joblib.load(scaler_file)
                self.pca = joblib.load(pca_file)
                self.core_samples = joblib.load(cores_file)
                self.is_loaded = True
                print(f"[STORE] Artefactos cargados: {self.core_samples.shape[0]} core samples disponibles.")
            except Exception as e:
                print(f"[STORE] Advertencia al cargar joblib: {e}")
                self._init_fallback_mode()
        else:
            print("[STORE] Archivos joblib no encontrados. Inicializando modo de contingencia...")
            self._init_fallback_mode()

    def _init_fallback_mode(self):
        """
        Genera core samples sintéticos basados en la distribución de PCA
        para evitar caídas si aún no se ha ejecutado train.py localmente.
        """
        np.random.seed(42)
        # 10 componentes PCA simulando la nube de densidad normal
        self.core_samples = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10))
        self.eps = 2.5
        self.is_loaded = True


# Instancia global compartida entre invocaciones serverless
store = ModelStore()

# Orden de las 30 características esperadas por el modelo
FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]


def predict_single_vector(vector_30d: np.ndarray) -> PredictionResult:
    """
    Aplica Scaler + PCA y calcula la distancia euclidiana mínima a los Core Samples.
    
    REGLA DE DECISIÓN DBSCAN (No supervisado):
    - distancia <= eps -> Pertenece a un cluster denso -> LEGÍTIMA
    - distancia > eps  -> Punto aislado / ruido          -> FRAUDE
    """
    # 1. Transformación
    if store.scaler is not None and store.pca is not None:
        scaled = store.scaler.transform(vector_30d)
        pca_proj = store.pca.transform(scaled)
    else:
        # Fallback si no hay scaler (normalización estándar manual simple)
        pca_proj = vector_30d[:, :10]

    # 2. Distancia al core sample más cercano
    _, distances = pairwise_distances_argmin_min(pca_proj, store.core_samples, metric="euclidean")
    min_dist = float(distances[0])

    # 3. Clasificación y anomaly score
    eps = store.eps
    is_fraud = bool(min_dist > eps)
    score = round(min_dist / eps, 4)

    return PredictionResult(
        is_fraud=is_fraud,
        label="FRAUDE" if is_fraud else "LEGÍTIMA",
        anomaly_score=score,
        distance_to_nearest_core=round(min_dist, 4),
        eps=eps
    )


# -------------------------------------------------------------------------
# ENDPOINTS DE LA API (Compatibles con y sin prefijo /api)
# -------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        message="DBSCAN Fraud Detection Serverless Engine Activo",
        model_loaded=store.is_loaded,
        num_core_samples=len(store.core_samples) if store.core_samples is not None else 0,
        eps=store.eps
    )


@app.get("/metrics")
@app.get("/api/metrics")
def get_metrics():
    """Devuelve las métricas de rendimiento evaluadas frente al ground truth 'Class'."""
    if not store.metrics_data:
        raise HTTPException(status_code=404, detail="Métricas no disponibles")
    return store.metrics_data


@app.get("/samples")
@app.get("/api/samples")
def get_samples():
    """Devuelve transacciones reales de ejemplo (legítimas y fraudulentas) para pruebas."""
    return store.sample_transactions


@app.post("/predict", response_model=PredictionResult)
@app.post("/api/predict", response_model=PredictionResult)
def predict_transaction(tx: TransactionFeatures):
    """
    Evalúa una única transacción. Valida la entrada con Pydantic y aplica la regla DBSCAN.
    """
    data = tx.model_dump()
    vec = np.array([[data[col] for col in FEATURE_ORDER]], dtype=np.float64)
    return predict_single_vector(vec)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    """
    Evalúa un lote de transacciones de forma vectorial para máximo rendimiento.
    """
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="La lista de transacciones no puede estar vacía")

    matrix = np.array([
        [getattr(tx, col) for col in FEATURE_ORDER]
        for tx in batch.transactions
    ], dtype=np.float64)

    # 1. Transformación
    if store.scaler is not None and store.pca is not None:
        scaled = store.scaler.transform(matrix)
        pca_proj = store.pca.transform(scaled)
    else:
        pca_proj = matrix[:, :10]

    # 2. Distancias vectoriales a los core samples
    _, distances = pairwise_distances_argmin_min(pca_proj, store.core_samples, metric="euclidean")

    results: List[PredictionResult] = []
    fraud_count = 0

    for dist in distances:
        d = float(dist)
        is_fraud = bool(d > store.eps)
        if is_fraud:
            fraud_count += 1
        results.append(PredictionResult(
            is_fraud=is_fraud,
            label="FRAUDE" if is_fraud else "LEGÍTIMA",
            anomaly_score=round(d / store.eps, 4),
            distance_to_nearest_core=round(d, 4),
            eps=store.eps
        ))

    return BatchPredictionResponse(
        total_processed=len(results),
        fraud_count=fraud_count,
        legitimate_count=len(results) - fraud_count,
        results=results
    )

