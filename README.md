# Sistema de Detección de Fraude con Aprendizaje No Supervisado (DBSCAN)

Proyecto universitario de Seguridad Informática y Machine Learning para clasificación de transacciones financieras en tiempo real empleando **DBSCAN** y **PCA**, optimizado para despliegue Serverless en **Vercel**.

---

## 1. Descarga del Dataset con Kaggle CLI

El dataset oficial es **Credit Card Fraud Detection** (ULB Machine Learning Group).
Descárgalo en la raíz del backend con:

```bash
# Configura tus credenciales (~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p backend/
```

Verifica que el archivo `creditcard.csv` quede en `backend/creditcard.csv`.

---

## 2. Ajuste de Hiperparámetros y Entrenamiento Local

> **Aviso de Arquitectura Serverless**: No se entrena en Vercel debido al límite de tiempo de ejecución (10s a 60s) y al sistema de archivos de solo lectura. El entrenamiento se ejecuta localmente y los artefactos ligeros (`.joblib` y `.json`) se suben al repositorio.

### Paso 2.1: Crear entorno virtual e instalar dependencias de entrenamiento
```bash
cd backend
python -m venv .venv

# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
# source .venv/bin/activate

pip install -r requirements-train.txt
```

### Paso 2.2: Ajustar `eps` y `min_samples` (Gráfico del Codo)
Ejecuta el script de apoyo:
```bash
python training/find_eps.py
```
Este script genera `artifacts/k_distance_plot.png`. Abre la imagen e inspecciona el punto de máxima curvatura (codo). Por defecto, `min_samples = 10` y `eps = 2.5`.

### Paso 2.3: Ejecutar el entrenamiento y generación de artefactos
```bash
python training/train.py
```
Este script:
1. Limpia nulos y duplicados.
2. Muestrea todos los fraudes + 20,000 normales.
3. Ajusta `StandardScaler` y `PCA(10)`.
4. Entrena DBSCAN en modo estrictamente no supervisado.
5. Evalúa frente a la columna `Class` y genera `backend/artifacts/metrics.json`.
6. Submuestrea los core samples a un máximo de 5,000 puntos y exporta los `.joblib`.

---

## 3. Pruebas Locales del Backend con Uvicorn

```bash
cd backend
uvicorn api.index:app --reload --port 8000
```
Comprueba los endpoints en tu navegador:
- Salud: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- Métricas: [http://127.0.0.1:8000/api/metrics](http://127.0.0.1:8000/api/metrics)
- Muestras: [http://127.0.0.1:8000/api/samples](http://127.0.0.1:8000/api/samples)
- Documentación Swagger interactiva: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4. Pruebas Locales del Frontend

Abre otra terminal:
```bash
cd frontend
npm install
npm run dev
```
Abre en tu navegador [http://localhost:3000](http://localhost:3000).

---

## 5. Despliegue en Vercel (Paso a Paso)

Dado que es un repositorio con frontend y backend desacoplados, se configuran como **dos proyectos independientes en Vercel** vinculados al mismo repositorio de GitHub:

### Proyecto 1: Desplegar el Backend primero
1. En [Vercel Dashboard](https://vercel.com/new), selecciona **Add New > Project** e importa tu repositorio.
2. Configura el proyecto:
   - **Project Name**: `fraud-dbscan-backend`
   - **Framework Preset**: `Other`
   - **Root Directory**: Haz clic en *Edit* y selecciona `backend`.
3. Variables de Entorno:
   - `ALLOWED_ORIGINS`: `*` (o la URL de tu frontend una vez desplegado).
4. Haz clic en **Deploy**.
5. Copia la URL asignada (ejemplo: `https://fraud-dbscan-backend.vercel.app`).

### Proyecto 2: Desplegar el Frontend
1. En Vercel Dashboard, selecciona **Add New > Project** e importa el mismo repositorio.
2. Configura el proyecto:
   - **Project Name**: `fraud-dbscan-frontend`
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Haz clic en *Edit* y selecciona `frontend`.
3. Variables de Entorno:
   - `NEXT_PUBLIC_API_URL`: Pega la URL del backend desplegado en el paso anterior (sin barra final, ej: `https://fraud-dbscan-backend.vercel.app`).
4. Haz clic en **Deploy**.

---

## 6. Limitaciones Teóricas y Prácticas para el Informe Universitario

1. **DBSCAN es un algoritmo transductivo, no inductivo**:
   - Por definición matemática, DBSCAN no posee una función nativa `.predict()` para puntos nuevos. Se implementa la proyección en PCA y la distancia mínima a los *Core Samples*. Si bien es geométricamente coherente, no modela cadenas arbitrarias de densidad conectadas (*density-reachable chains*).
2. **Supuesto de densidad uniforme**:
   - DBSCAN asume un único radio global $\varepsilon$. El comportamiento legítimo puede presentar densidades heterogéneas según el perfil del usuario.
3. **Complejidad computacional ($O(N^2)$)**:
   - Procesar las 284,807 transacciones simultáneamente requiere excesiva memoria. Por eso es indispensable el muestreo balanceado.
4. **Límites Serverless en Vercel (250 MB)**:
   - Desacoplamos estrictamente `requirements-train.txt` (con pandas y matplotlib) de `requirements.txt` (solo inferencia ligera con numpy, scikit-learn y joblib).

