"""
Script de apoyo para estimar el valor óptimo de eps mediante la técnica del k-distance plot (codo / elbow).
Genera y guarda un gráfico k_distance_plot.png para documentar la tesis/informe.
"""
import os
import sys
from pathlib import Path

# Asegurar path de búsqueda de módulos
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from preprocess import load_and_clean_data, sample_balanced_data, fit_scaler_and_pca


def generate_k_distance_plot(min_samples: int = 10, output_img: str = "artifacts/k_distance_plot.png"):
    """
    Calcula la distancia al k-ésimo vecino más cercano (k = min_samples)
    y genera el gráfico ordenado para ubicar la rodilla del codo.
    """
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    
    # 1. Obtener datos procesados
    df = load_and_clean_data("creditcard.csv")
    X, _ = sample_balanced_data(df, n_normal=20000, random_state=42)
    X_pca, _, _ = fit_scaler_and_pca(X, n_components=10)

    print(f"[FIND_EPS] Calculando distancias al vecino k={min_samples}...")
    nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean", n_jobs=-1)
    nbrs.fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)

    # Distancia al k-ésimo vecino (última columna)
    k_distances = np.sort(distances[:, -1])

    # 2. Generar gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, color="#2563EB", linewidth=1.5, label=f"k-distancia (k={min_samples})")
    plt.axhline(y=2.5, color="#DC2626", linestyle="--", label="eps sugerido = 2.5")
    plt.title("Método del Codo (k-distance plot) para DBSCAN", fontsize=14, fontweight="bold")
    plt.xlabel("Puntos ordenados por distancia", fontsize=12)
    plt.ylabel(f"Distancia al {min_samples}-ésimo vecino (Epsilon)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"[FIND_EPS] Gráfico guardado exitosamente en: {output_img}")
    print("[FIND_EPS] Inspecciona el punto de máxima curvatura (codo) para refinar eps.")


if __name__ == "__main__":
    generate_k_distance_plot(min_samples=10)

