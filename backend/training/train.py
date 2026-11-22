"""
Script principal de entrenamiento LOCAL.
- Entrena DBSCAN en modo no supervisado.
- Evalúa el desempeño frente a las etiquetas reales 'Class' (no usadas en entrenamiento).
- Guarda artefactos ligeros (< 2 MB) en backend/artifacts/ para despliegue en Vercel.
"""
import os
import sys
from pathlib import Path

# Asegurar path de búsqueda de módulos
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import json
import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    pairwise_distances_argmin_min
)
from preprocess import load_and_clean_data, sample_balanced_data, fit_scaler_and_pca

# Rutas de artefactos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "artifacts")


def train_and_export_artifacts(
    eps: float = 2.5,
    min_samples: int = 10,
    n_components: int = 10,
    max_core_samples_export: int = 5000
):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO DBSCAN (APRENDIZAJE NO SUPERVISADO)")
    print("=" * 60)

    # 1. Cargar y preparar datos
    csv_file = os.path.join(os.path.dirname(BASE_DIR), "creditcard.csv")
    if not os.path.exists(csv_file):
        csv_file = "creditcard.csv"

    df_clean = load_and_clean_data(csv_file)
    X, y = sample_balanced_data(df_clean, n_normal=20000, random_state=42)

    # Extraer ejemplos de demostración (5 legítimos y 5 fraudes) antes de transformar
    print("[TRAIN] Extrayendo transacciones de prueba para la demo...")
    fraudes_idx = y[y == 1].index[:5]
    legit_idx = y[y == 0].index[:5]

    sample_transactions = []
    for idx in legit_idx:
        row = X.loc[idx].to_dict()
        row["_true_label"] = "LEGÍTIMA"
        row["_is_fraud_ground_truth"] = 0
        sample_transactions.append(row)

    for idx in fraudes_idx:
        row = X.loc[idx].to_dict()
        row["_true_label"] = "FRAUDE"
        row["_is_fraud_ground_truth"] = 1
        sample_transactions.append(row)

    samples_path = os.path.join(ARTIFACTS_DIR, "sample_transactions.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(sample_transactions, f, indent=2)
    print(f"[TRAIN] Guardadas 10 muestras en: {samples_path}")

    # 2. Escalar y Reducir con PCA
    X_pca, scaler, pca = fit_scaler_and_pca(X, n_components=n_components, random_state=42)

    # 3. Entrenar DBSCAN (Solo con X_pca, SIN etiquetas y)
    print(f"[TRAIN] Ajustando DBSCAN con eps={eps}, min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    dbscan.fit(X_pca)

    # En DBSCAN:
    # cluster_id >= 0: pertenece a un cluster denso (Transacciones normales)
    # cluster_id == -1: punto de ruido / anomalía (Fraude sospechoso)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"[TRAIN] Clusters densos descubiertos: {n_clusters}")
    print(f"[TRAIN] Puntos clasificados como ruido/anomalía: {n_noise} de {len(labels)}")

    # 4. Extraer y Submuestrear Core Samples para despliegue ultra-ligero en Vercel
    core_samples = dbscan.components_
    n_cores_total = core_samples.shape[0]
    print(f"[TRAIN] Total de Core Samples generados: {n_cores_total}")

    if n_cores_total > max_core_samples_export:
        print(f"[TRAIN] Submuestreando a {max_core_samples_export} core samples para optimizar latencia serverless...")
        np.random.seed(42)
        indices = np.random.choice(n_cores_total, size=max_core_samples_export, replace=False)
        core_samples_export = core_samples[indices]
    else:
        core_samples_export = core_samples

    # 5. Evaluación de Desempeño
    # Implementamos la regla de inferencia por distancia euclidiana al core sample más cercano
    print("[EVAL] Evaluando el modelo frente a las etiquetas ground truth 'Class'...")
    _, min_distances = pairwise_distances_argmin_min(X_pca, core_samples_export, metric="euclidean")

    # Si distancia > eps -> Ruido/Fraude (1). Si distancia <= eps -> Legítima (0).
    y_pred = (min_distances > eps).astype(int)

    precision = float(precision_score(y, y_pred, zero_division=0))
    recall = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Cálculo de AUC-PR utilizando el anomaly_score continuo (distancia / eps)
    anomaly_scores = min_distances / eps
    precision_curve, recall_curve, _ = precision_recall_curve(y, anomaly_scores)
    pr_auc = float(auc(recall_curve, precision_curve))

    metrics = {
        "algorithm": "DBSCAN (Unsupervised)",
        "eps": eps,
        "min_samples": min_samples,
        "n_pca_components": n_components,
        "total_evaluated_transactions": int(len(y)),
        "actual_fraud_count": int(np.sum(y == 1)),
        "actual_legit_count": int(np.sum(y == 0)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

    print("\n--- RESUMEN DE MÉTRICAS (EVALUACIÓN CONJUNTO MUESTREADO) ---")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:  {metrics['f1_score'] * 100:.2f}%")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Matriz de Confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-----------------------------------------------------------\n")

    # 6. Serializar Artefactos
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"), compress=3)
    joblib.dump(pca, os.path.join(ARTIFACTS_DIR, "pca.joblib"), compress=3)
    joblib.dump(core_samples_export, os.path.join(ARTIFACTS_DIR, "core_samples.joblib"), compress=3)

    params = {
        "eps": eps,
        "min_samples": min_samples,
        "n_pca_components": n_components,
        "num_exported_core_samples": int(core_samples_export.shape[0]),
        "feature_names": list(X.columns)
    }

    with open(os.path.join(ARTIFACTS_DIR, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[EXPORT] Todos los artefactos fueron exportados con éxito a '{ARTIFACTS_DIR}'.")


if __name__ == "__main__":
    train_and_export_artifacts(eps=2.5, min_samples=10, n_components=10)

