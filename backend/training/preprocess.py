"""
Pipeline de carga, limpieza y muestreo del dataset Credit Card Fraud Detection.
Se ejecuta localmente antes de entrenar DBSCAN.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_clean_data(csv_path: str = "creditcard.csv") -> pd.DataFrame:
    """
    Carga el dataset creditcard.csv, verifica nulos y duplicados y los limpia.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el archivo {csv_path}. Descárgalo de Kaggle ejecutando: "
            "kaggle datasets download -d mlg-ulb/creditcardfraud --unzip"
        )

    print(f"[PREPROCESS] Cargando {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[PREPROCESS] Dimensiones iniciales: {df.shape}")

    # 1. Verificación de nulos
    n_nulos = df.isnull().sum().sum()
    if n_nulos > 0:
        print(f"[PREPROCESS] Eliminando {n_nulos} valores nulos...")
        df = df.dropna()
    else:
        print("[PREPROCESS] Sin valores nulos detectados.")

    # 2. Verificación y eliminación de duplicados
    n_duplicados = df.duplicated().sum()
    if n_duplicados > 0:
        print(f"[PREPROCESS] Eliminando {n_duplicados} filas duplicadas...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[PREPROCESS] Sin filas duplicadas.")

    print(f"[PREPROCESS] Dimensiones tras limpieza: {df.shape}")
    return df


def sample_balanced_data(df: pd.DataFrame, n_normal: int = 20000, random_state: int = 42):
    """
    DBSCAN tiene complejidad O(N^2) en memoria/tiempo.
    Muestreamos TODOS los fraudes disponibles + un subconjunto representativo de transacciones legítimas.
    """
    fraudes = df[df["Class"] == 1]
    normales = df[df["Class"] == 0]

    n_fraudes = len(fraudes)
    print(f"[PREPROCESS] Total de fraudes encontrados: {n_fraudes}")
    print(f"[PREPROCESS] Muestreando {n_normal} transacciones normales...")

    normales_sample = normales.sample(n=min(n_normal, len(normales)), random_state=random_state)

    # Combinar y barajar
    df_sample = pd.concat([fraudes, normales_sample], axis=0)
    df_sample = df_sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Separar características de la etiqueta
    # IMPORTANTE: 'Class' solo se usará para validación y métricas, NUNCA para entrenar DBSCAN
    X = df_sample.drop(columns=["Class"])
    y = df_sample["Class"].astype(int)

    print(f"[PREPROCESS] Conjunto resultante: {X.shape[0]} registros ({n_fraudes} fraudes, {len(normales_sample)} legítimas).")
    return X, y


def fit_scaler_and_pca(X: pd.DataFrame, n_components: int = 10, random_state: int = 42):
    """
    Ajusta el StandardScaler y reduce la dimensionalidad con PCA a 10 componentes.
    """
    print(f"[PREPROCESS] Ajustando StandardScaler en {X.shape[1]} variables...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[PREPROCESS] Reduciendo dimensionalidad con PCA a {n_components} componentes...")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    varianza_explicada = float(np.sum(pca.explained_variance_ratio_) * 100)
    print(f"[PREPROCESS] Varianza explicada total por los {n_components} componentes: {varianza_explicada:.2f}%")

    return X_pca, scaler, pca

