"use client";

import React, { useState, useEffect } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Layers,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Upload,
  Info,
  Server,
  AlertTriangle,
  FileText,
  Shuffle
} from "lucide-react";

// Configuración de URL del Backend (Vercel Serverless o Local) - limpia barras finales automáticas
const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

interface TransactionForm {
  Time: number;
  Amount: number;
  [key: string]: number;
}

interface PredictionResult {
  is_fraud: boolean;
  label: "FRAUDE" | "LEGÍTIMA";
  anomaly_score: number;
  distance_to_nearest_core: number;
  eps: number;
}

interface MetricsData {
  algorithm: string;
  eps: number;
  min_samples: number;
  precision: number;
  recall: number;
  f1_score: number;
  pr_auc: number;
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
  total_evaluated_transactions: number;
  actual_fraud_count: number;
  actual_legit_count: number;
}

interface FeatureMeta {
  code: string;
  label: string;
  category: "Crítico" | "Alto Riesgo" | "Normal";
  hint: string;
}

// Diccionario interpretativo de las 28 componentes PCA basado en estudios de correlación de fraude
const FEATURE_METADATA: Record<string, FeatureMeta> = {
  V14: { code: "V14", label: "Alerta Crítica de Dispositivo", category: "Crítico", hint: "Máxima correlación inversa con el fraude bancario" },
  V12: { code: "V12", label: "Patrón de Autenticación", category: "Crítico", hint: "Desviación en verificación y validación de seguridad" },
  V10: { code: "V10", label: "Desviación de Perfil", category: "Crítico", hint: "Comportamiento atípico respecto al consumo del usuario" },
  V17: { code: "V17", label: "Anomalía de Red / IP", category: "Crítico", hint: "Discrepancia en canal, origen o geolocalización IP" },
  V4:  { code: "V4",  label: "Velocidad de Transacción", category: "Alto Riesgo", hint: "Frecuencia inusual de intentos en ventanas cortas" },
  V11: { code: "V11", label: "Patrón de Reintentos", category: "Alto Riesgo", hint: "Intentos sucesivos con montos variables" },
  V2:  { code: "V2",  label: "Canal de Pago / POS", category: "Alto Riesgo", hint: "Variación de pasarela o terminal física" },
  V3:  { code: "V3",  label: "Consistencia de Sesión", category: "Normal", hint: "Huella y tokens de la sesión activa" },
  V7:  { code: "V7",  label: "Reputación del Comercio", category: "Normal", hint: "Nivel de riesgo del establecimiento (MCC)" },
  V9:  { code: "V9",  label: "Historial de Saldo", category: "Normal", hint: "Movimientos bancarios previos inmediatos" },
  V1:  { code: "V1",  label: "Segmento del Tarjetahabiente", category: "Normal", hint: "Perfil demográfico y antigüedad general" },
  V5:  { code: "V5",  label: "Geolocalización Latente", category: "Normal", hint: "Distancia estimada a transacciones habituales" },
  V6:  { code: "V6",  label: "Dispersión de Horario", category: "Normal", hint: "Operación fuera de la ventana habitual" },
  V8:  { code: "V8",  label: "Validación de Chip / PIN", category: "Normal", hint: "Integridad del medio de pago electrónico" },
  V13: { code: "V13", label: "Verificación de Emisor", category: "Normal", hint: "Respuesta del banco emisor de la tarjeta" },
  V15: { code: "V15", label: "Firma Digital de App", category: "Normal", hint: "Validación de versión y entorno cliente" },
  V16: { code: "V16", label: "Variación de Límite", category: "Normal", hint: "Impacto en el límite de crédito disponible" },
  V18: { code: "V18", label: "Riesgo Cruzado", category: "Normal", hint: "Alertas en otras cuentas vinculadas" },
  V19: { code: "V19", label: "Divergencia de Divisa", category: "Normal", hint: "Transacciones multimoneda infrecuentes" },
  V20: { code: "V20", label: "Consumo Acumulado", category: "Normal", hint: "Gasto diario o semanal acumulado" },
  V21: { code: "V21", label: "Latencia de Autorización", category: "Normal", hint: "Tiempos de ida y vuelta en la pasarela" },
  V22: { code: "V22", label: "Dispositivos Concurrentes", category: "Normal", hint: "Acceso simultáneo desde múltiples puntos" },
  V23: { code: "V23", label: "Patrón Biométrico", category: "Normal", hint: "Telemetría de interacción con teclado/pantalla" },
  V24: { code: "V24", label: "Detección de Proxy / VPN", category: "Normal", hint: "Ocultamiento deliberado del origen real" },
  V25: { code: "V25", label: "Madurez de Cuenta", category: "Normal", hint: "Comportamiento del tarjetahabiente a largo plazo" },
  V26: { code: "V26", label: "Huella de Hardware", category: "Normal", hint: "Fingerprinting de navegador o terminal" },
  V27: { code: "V27", label: "Discrepancia de Facturación", category: "Normal", hint: "Diferencia entre dirección de cobro y entrega" },
  V28: { code: "V28", label: "Índice de Contracargos", category: "Normal", hint: "Historial de disputas financieras" },
};

export default function HomePage() {
  // Estado del formulario
  const [formData, setFormData] = useState<TransactionForm>(() => {
    const initial: TransactionForm = { Time: 0, Amount: 100.0 };
    for (let i = 1; i <= 28; i++) {
      initial[`V${i}`] = 0.0;
    }
    return initial;
  });

  const [isVFeaturesOpen, setIsVFeaturesOpen] = useState(false);
  const [filterCategory, setFilterCategory] = useState<"all" | "critical" | "high">("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<Array<PredictionResult & { id: string; amount: number; time: number }>>([]);

  const resetVFeatures = () => {
    setFormData((prev) => {
      const updated = { ...prev };
      for (let i = 1; i <= 28; i++) {
        updated[`V${i}`] = 0.0;
      }
      return updated;
    });
  };
  
  // Muestras y Métricas
  const [samples, setSamples] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [backendStatus, setBackendStatus] = useState<string>("Verificando...");
  const [activeTab, setActiveTab] = useState<"single" | "batch" | "metrics" | "about">("single");

  // Modo Lote
  const [batchJson, setBatchJson] = useState("");
  const [batchResults, setBatchResults] = useState<any | null>(null);

  // Cargar métricas y muestras al iniciar
  useEffect(() => {
    async function initData() {
      try {
        const resHealth = await fetch(`${API_URL}/api/health`);
        if (resHealth.ok) {
          setBackendStatus("Conectado");
        } else {
          setBackendStatus("Desconectado");
        }

        const resSamples = await fetch(`${API_URL}/api/samples`);
        if (resSamples.ok) {
          const samplesData = await resSamples.json();
          setSamples(samplesData);
        }

        const resMetrics = await fetch(`${API_URL}/api/metrics`);
        if (resMetrics.ok) {
          const mData = await resMetrics.json();
          setMetrics(mData);
        }
      } catch (err) {
        console.warn("Backend no disponible actualmente:", err);
        setBackendStatus("Error de Conexión");
      }
    }
    initData();
  }, []);

  const handleInputChange = (field: string, value: string) => {
    const numeric = parseFloat(value) || 0;
    setFormData((prev) => ({ ...prev, [field]: numeric }));
  };

  const loadSample = (type: "LEGÍTIMA" | "FRAUDE") => {
    const match = samples.find((s) => s._true_label === type);
    if (match) {
      const nextData: TransactionForm = {
        Time: match.Time ?? 0,
        Amount: match.Amount ?? 0,
      };
      for (let i = 1; i <= 28; i++) {
        nextData[`V${i}`] = match[`V${i}`] ?? 0;
      }
      setFormData(nextData);
      setResult(null);
      setError(null);
    } else {
      setError(`No hay muestras de tipo ${type} disponibles.`);
    }
  };

  const loadRandom = () => {
    // 50% de probabilidad de generar una transacción legítima vs fraudulenta
    const simulateFraud = Math.random() < 0.5;

    // Buscar una plantilla base de los datos reales del backend
    const legitPool = samples.filter((s) => s._true_label === "LEGÍTIMA");
    const fraudPool = samples.filter((s) => s._true_label === "FRAUDE");

    let base: any = null;
    if (simulateFraud && fraudPool.length > 0) {
      base = fraudPool[Math.floor(Math.random() * fraudPool.length)];
    } else if (!simulateFraud && legitPool.length > 0) {
      base = legitPool[Math.floor(Math.random() * legitPool.length)];
    } else if (samples.length > 0) {
      base = samples[Math.floor(Math.random() * samples.length)];
    }

    if (base) {
      // Perturbación realista sobre el perfil bancario base
      const timeVariation = (Math.random() - 0.5) * 5000;
      const amountRatio = simulateFraud ? 0.4 : 0.15;
      const amountVariation = (Math.random() - 0.5) * 2 * amountRatio;
      const newAmount = Math.max(1.0, base.Amount * (1 + amountVariation));

      const nextData: TransactionForm = {
        Time: Math.max(0, Math.floor(base.Time + timeVariation)),
        Amount: parseFloat(newAmount.toFixed(2)),
      };

      for (let i = 1; i <= 28; i++) {
        const baseVal = base[`V${i}`] ?? 0;
        // Si es legítima, variaciones muy pequeñas (se mantiene dentro del cluster denso eps)
        // Si es fraude, variaciones más agresivas en componentes críticas
        const isCriticalFeature = [14, 12, 10, 17, 4].includes(i);
        const noiseScale = simulateFraud
          ? (isCriticalFeature ? 1.2 : 0.5)
          : 0.12;

        const noise = (Math.random() - 0.5) * 2 * noiseScale;
        nextData[`V${i}`] = parseFloat((baseVal + noise).toFixed(4));
      }

      setFormData(nextData);
    } else {
      // Contingencia si aún no han cargado las muestras de la API
      const randomAmount = parseFloat((Math.random() * 250 + 10).toFixed(2));
      const randomTime = Math.floor(Math.random() * 80000);
      const nextData: TransactionForm = { Time: randomTime, Amount: randomAmount };

      for (let i = 1; i <= 28; i++) {
        // En una distribución estándar N(0, 0.15) suele caer dentro del cluster legítimo
        const scale = simulateFraud ? 2.5 : 0.15;
        const val = parseFloat(((Math.random() - 0.5) * 2 * scale).toFixed(4));
        nextData[`V${i}`] = val;
      }
      setFormData(nextData);
    }

    setResult(null);
    setError(null);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        throw new Error(`Error en el servidor: ${res.statusText}`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);

      // Agregar a historial de sesión
      setHistory((prev) => [
        {
          ...data,
          id: Math.random().toString(36).substring(7),
          amount: formData.Amount,
          time: formData.Time,
        },
        ...prev,
      ]);
    } catch (err: any) {
      setError(err.message || "Error al realizar la predicción");
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async () => {
    setLoading(true);
    setError(null);
    try {
      let parsed = JSON.parse(batchJson);
      if (!Array.isArray(parsed)) {
        throw new Error("El JSON debe ser una lista de transacciones.");
      }
      const res = await fetch(`${API_URL}/api/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transactions: parsed }),
      });
      if (!res.ok) throw new Error("Fallo en la predicción por lote");
      const data = await res.json();
      setBatchResults(data);
    } catch (err: any) {
      setError(err.message || "Error al procesar lote");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-200 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-blue-50 text-blue-600 rounded-xl border border-blue-200 shadow-xs">
              <ShieldAlert className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-900">
                Detección de Fraude en Tiempo Real
              </h1>
              <p className="text-sm text-slate-500">
                Modelo de Aprendizaje No Supervisado con DBSCAN & Reducción PCA
              </p>
            </div>
          </div>
        </div>

        {/* Estado del Backend */}
        <div className="flex items-center gap-3 bg-white border border-slate-200 shadow-xs px-4 py-2 rounded-xl text-xs">
          <Server className="w-4 h-4 text-slate-400" />
          <span className="text-slate-500">Backend Serverless:</span>
          <span
            className={`font-semibold px-2.5 py-0.5 rounded-full ${
              backendStatus === "Conectado"
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                : "bg-amber-50 text-amber-700 border border-amber-200"
            }`}
          >
            {backendStatus}
          </span>
        </div>
      </header>

      {/* Navegación por Pestañas */}
      <nav className="flex gap-2 border-b border-slate-200 pb-3">
        {[
          { id: "single", label: "Analizador de Transacción", icon: Activity },
          { id: "batch", label: "Evaluación en Lote (Batch)", icon: Layers },
          { id: "metrics", label: "Métricas del Modelo", icon: FileText },
          { id: "about", label: "¿Cómo Funciona?", icon: Info },
        ].map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? "bg-blue-600 text-white shadow-md shadow-blue-500/20"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* TAB 1: ANALIZADOR DE TRANSACCIÓN */}
      {activeTab === "single" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Formulario */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-white border border-slate-200/80 rounded-2xl p-6 shadow-sm">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-semibold text-slate-900">Parámetros de la Transacción</h2>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => loadSample("LEGÍTIMA")}
                    className="text-xs bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-300 px-3 py-1.5 rounded-lg transition font-medium"
                  >
                    + Cargar Legítima
                  </button>
                  <button
                    type="button"
                    onClick={() => loadSample("FRAUDE")}
                    className="text-xs bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-300 px-3 py-1.5 rounded-lg transition font-medium"
                  >
                    + Cargar Fraude
                  </button>
                  <button
                    type="button"
                    onClick={loadRandom}
                    title="Generar transacción aleatoria para probar el algoritmo"
                    className="text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-300 px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 font-medium"
                  >
                    <Shuffle className="w-3.5 h-3.5" />
                    Aleatorio
                  </button>
                </div>
              </div>

              <form onSubmit={handlePredict} className="space-y-6">
                {/* Amount y Time */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-slate-700">
                      Monto de la Transacción ($ Amount)
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      required
                      value={formData.Amount}
                      onChange={(e) => handleInputChange("Amount", e.target.value)}
                      className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm shadow-xs"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-semibold text-slate-700">
                      Tiempo (Segundos transcurridos)
                    </label>
                    <input
                      type="number"
                      step="1"
                      required
                      value={formData.Time}
                      onChange={(e) => handleInputChange("Time", e.target.value)}
                      className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm shadow-xs"
                    />
                  </div>
                </div>

                {/* V1..V28 Colapsable con Etiquetas Descriptivas */}
                <div className="border border-slate-200 rounded-2xl overflow-hidden bg-slate-50/70 shadow-xs">
                  <div className="w-full flex flex-col sm:flex-row justify-between items-start sm:items-center px-5 py-4 border-b border-slate-200 bg-slate-100/70 gap-3">
                    <button
                      type="button"
                      onClick={() => setIsVFeaturesOpen(!isVFeaturesOpen)}
                      className="flex items-center gap-2 text-left group"
                    >
                      <span className="text-sm font-semibold text-slate-800 group-hover:text-blue-600 transition">
                        Variables Latentes Anonimizadas (V1 a V28)
                      </span>
                      <span className="text-[11px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full font-mono font-medium">
                        PCA Bancario
                      </span>
                      {isVFeaturesOpen ? (
                        <ChevronUp className="w-4 h-4 text-slate-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-slate-500" />
                      )}
                    </button>

                    {isVFeaturesOpen && (
                      <div className="flex items-center gap-2">
                        {/* Selector de filtro de variables */}
                        <div className="flex bg-white p-1 rounded-lg border border-slate-200 text-[11px] shadow-2xs">
                          <button
                            type="button"
                            onClick={() => setFilterCategory("all")}
                            className={`px-2.5 py-1 rounded-md transition ${
                              filterCategory === "all"
                                ? "bg-blue-600 text-white font-medium"
                                : "text-slate-600 hover:text-slate-900"
                            }`}
                          >
                            Todas (28)
                          </button>
                          <button
                            type="button"
                            onClick={() => setFilterCategory("critical")}
                            className={`px-2.5 py-1 rounded-md transition ${
                              filterCategory === "critical"
                                ? "bg-rose-600 text-white font-medium"
                                : "text-slate-600 hover:text-slate-900"
                            }`}
                          >
                            Críticas (4)
                          </button>
                          <button
                            type="button"
                            onClick={() => setFilterCategory("high")}
                            className={`px-2.5 py-1 rounded-md transition ${
                              filterCategory === "high"
                                ? "bg-amber-600 text-white font-medium"
                                : "text-slate-600 hover:text-slate-900"
                            }`}
                          >
                            Alto Impacto (7)
                          </button>
                        </div>

                        <button
                          type="button"
                          onClick={resetVFeatures}
                          title="Restablecer todas las variables V a 0.0"
                          className="text-[11px] text-slate-600 hover:text-slate-900 bg-white border border-slate-200 px-2.5 py-1.5 rounded-lg transition shadow-2xs font-medium"
                        >
                          Reset 0.0
                        </button>
                      </div>
                    )}
                  </div>

                  {isVFeaturesOpen && (
                    <div className="p-5 space-y-4">
                      <p className="text-xs text-slate-500 leading-relaxed">
                        Etiquetas de dominio simuladas según la correlación estadística con el fraude financiero en el dataset de Kaggle (ULB). Puedes editar directamente los valores numéricos:
                      </p>

                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-3.5">
                        {Array.from({ length: 28 }, (_, i) => `V${i + 1}`)
                          .filter((k) => {
                            if (filterCategory === "critical") return FEATURE_METADATA[k]?.category === "Crítico";
                            if (filterCategory === "high") {
                              return (
                                FEATURE_METADATA[k]?.category === "Crítico" ||
                                FEATURE_METADATA[k]?.category === "Alto Riesgo"
                              );
                            }
                            return true;
                          })
                          .map((key) => {
                            const meta = FEATURE_METADATA[key] || {
                              code: key,
                              label: "Variable Latente",
                              category: "Normal",
                              hint: "Componente PCA estándar",
                            };

                            const isCritical = meta.category === "Crítico";
                            const isHigh = meta.category === "Alto Riesgo";

                            return (
                              <div
                                key={key}
                                className={`p-3 rounded-xl border transition-all ${
                                  isCritical
                                    ? "bg-rose-50/70 border-rose-200 hover:border-rose-300"
                                    : isHigh
                                    ? "bg-amber-50/70 border-amber-200 hover:border-amber-300"
                                    : "bg-white border-slate-200 hover:border-blue-300 shadow-2xs"
                                }`}
                              >
                                <div className="flex justify-between items-center mb-1">
                                  <div className="flex items-center gap-1.5">
                                    <span className="font-mono text-xs font-bold text-slate-900">
                                      {meta.code}
                                    </span>
                                    <span className="text-xs font-semibold text-slate-700 truncate max-w-[170px]">
                                      {meta.label}
                                    </span>
                                  </div>
                                  <span
                                    className={`text-[9px] px-1.5 py-0.5 rounded-full font-bold uppercase ${
                                      isCritical
                                        ? "bg-rose-100 text-rose-700 border border-rose-200"
                                        : isHigh
                                        ? "bg-amber-100 text-amber-700 border border-amber-200"
                                        : "bg-slate-100 text-slate-600"
                                    }`}
                                  >
                                    {meta.category}
                                  </span>
                                </div>

                                <p className="text-[10px] text-slate-500 mb-2 leading-tight">
                                  {meta.hint}
                                </p>

                                <div className="relative">
                                  <input
                                    type="number"
                                    step="any"
                                    value={formData[key]}
                                    onChange={(e) => handleInputChange(key, e.target.value)}
                                    className="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs text-slate-900 font-mono focus:outline-none focus:ring-1 focus:ring-blue-500"
                                  />
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}
                </div>

                {/* Botón de acción */}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-xl transition shadow-md shadow-blue-500/20 flex justify-center items-center gap-2 disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" />
                      Calculando distancias a Core Samples...
                    </>
                  ) : (
                    <>
                      <Activity className="w-4 h-4" />
                      Clasificar Transacción en Tiempo Real
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Panel de Resultado */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-white border border-slate-200/80 rounded-2xl p-6 shadow-sm flex flex-col justify-between min-h-[360px]">
              <div>
                <h2 className="text-lg font-semibold text-slate-900 mb-4">Veredicto de Seguridad</h2>

                {error && (
                  <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm flex gap-3 items-center">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0 text-rose-600" />
                    <span>{error}</span>
                  </div>
                )}

                {!result && !error && !loading && (
                  <div className="text-center py-12 text-slate-400 space-y-2">
                    <ShieldCheck className="w-12 h-12 mx-auto stroke-1 text-slate-300" />
                    <p className="text-sm">Ingresa los datos o carga una muestra para clasificar.</p>
                  </div>
                )}

                {result && (
                  <div className="space-y-6">
                    {/* Badge Grande */}
                    <div
                      className={`p-6 rounded-2xl border text-center space-y-2 ${
                        result.is_fraud
                          ? "bg-rose-50 border-rose-200 text-rose-800"
                          : "bg-emerald-50 border-emerald-200 text-emerald-800"
                      }`}
                    >
                      <div className="flex justify-center">
                        {result.is_fraud ? (
                          <ShieldAlert className="w-14 h-14 text-rose-600 animate-pulse" />
                        ) : (
                          <ShieldCheck className="w-14 h-14 text-emerald-600" />
                        )}
                      </div>
                      <h3 className="text-2xl font-black tracking-wide">
                        {result.label}
                      </h3>
                      <p className="text-xs opacity-90 max-w-xs mx-auto">
                        {result.is_fraud
                          ? "La transacción no encaja en ningún cluster denso conocido (outlier de ruido)."
                          : "La transacción se ubica dentro del radio épsilon de un cluster de comportamiento normal."}
                      </p>
                    </div>

                    {/* Métricas del Resultado */}
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                        <span className="text-slate-500 block mb-1">Score de Anomalía</span>
                        <span
                          className={`text-lg font-mono font-bold ${
                            result.anomaly_score > 1.0 ? "text-rose-600" : "text-emerald-600"
                          }`}
                        >
                          {result.anomaly_score}
                        </span>
                        <span className="text-[10px] text-slate-400 block mt-0.5">
                          (D / eps &gt; 1.0 = Fraude)
                        </span>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                        <span className="text-slate-500 block mb-1">Distancia Euclídea</span>
                        <span className="text-lg font-mono font-bold text-slate-900">
                          {result.distance_to_nearest_core}
                        </span>
                        <span className="text-[10px] text-slate-400 block mt-0.5">
                          Radio Epsilon: {result.eps}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="text-[11px] text-slate-400 border-t border-slate-100 pt-4 mt-6">
                Algoritmo: DBSCAN Transductivo aproximado mediante Core Samples.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: EVALUACIÓN EN LOTE */}
      {activeTab === "batch" && (
        <div className="bg-white border border-slate-200/80 rounded-2xl p-6 space-y-6 shadow-sm">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Predicción por Lotes (Batch)</h2>
              <p className="text-sm text-slate-500">
                Pega un arreglo JSON de transacciones para evaluar múltiples registros de forma vectorial.
              </p>
            </div>
            <button
              onClick={() => setBatchJson(JSON.stringify(samples.slice(0, 4), null, 2))}
              className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg font-medium border border-slate-200"
            >
              Cargar Plantilla de Muestra
            </button>
          </div>

          <textarea
            rows={8}
            value={batchJson}
            onChange={(e) => setBatchJson(e.target.value)}
            placeholder='[{"Time": 0, "Amount": 50, "V1": 0.5, ...}]'
            className="w-full bg-white border border-slate-300 rounded-xl p-4 font-mono text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-2xs"
          />

          <button
            onClick={handleBatchPredict}
            disabled={loading || !batchJson}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-xl text-sm transition shadow-md shadow-blue-500/20 disabled:opacity-50"
          >
            {loading ? "Procesando Lote..." : "Evaluar Lote Completo"}
          </button>

          {batchResults && (
            <div className="space-y-4 pt-4 border-t border-slate-200">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
                  <span className="text-xs text-slate-500 block">Procesadas</span>
                  <span className="text-xl font-bold font-mono text-slate-900">
                    {batchResults.total_processed}
                  </span>
                </div>
                <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl">
                  <span className="text-xs text-rose-700 block font-medium">Fraudes Detectados</span>
                  <span className="text-xl font-bold font-mono text-rose-700">
                    {batchResults.fraud_count}
                  </span>
                </div>
                <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
                  <span className="text-xs text-emerald-700 block font-medium">Legítimas</span>
                  <span className="text-xl font-bold font-mono text-emerald-700">
                    {batchResults.legitimate_count}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: MÉTRICAS DEL MODELO */}
      {activeTab === "metrics" && metrics && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm">
              <span className="text-xs text-slate-500">Precision (No Supervisado)</span>
              <p className="text-2xl font-bold font-mono text-slate-900 mt-1">
                {(metrics.precision * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-slate-400">Transacciones clasificadas como fraude que realmente lo eran</span>
            </div>
            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm">
              <span className="text-xs text-slate-500">Recall / Sensibilidad</span>
              <p className="text-2xl font-bold font-mono text-slate-900 mt-1">
                {(metrics.recall * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-slate-400">Porcentaje de fraudes reales interceptados</span>
            </div>
            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm">
              <span className="text-xs text-slate-500">F1-Score</span>
              <p className="text-2xl font-bold font-mono text-blue-600 mt-1">
                {(metrics.f1_score * 100).toFixed(1)}%
              </p>
              <span className="text-[10px] text-slate-400">Media armónica balanceada</span>
            </div>
            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm">
              <span className="text-xs text-slate-500">AUC-PR</span>
              <p className="text-2xl font-bold font-mono text-emerald-600 mt-1">
                {metrics.pr_auc.toFixed(3)}
              </p>
              <span className="text-[10px] text-slate-400">Área bajo la curva Precision-Recall</span>
            </div>
          </div>

          {/* Matriz de Confusión */}
          <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
            <h3 className="text-md font-semibold text-slate-900 mb-4">
              Matriz de Confusión (Evaluación de Seguridad Bancaria)
            </h3>
            <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto text-center font-mono">
              <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
                <span className="text-xs text-slate-600 block font-sans">Verdaderos Negativos (TN)</span>
                <span className="text-2xl font-bold text-emerald-700">
                  {metrics.confusion_matrix.true_negatives}
                </span>
                <span className="text-[10px] text-emerald-600/80 block mt-1">Legítimas autorizadas</span>
              </div>
              <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
                <span className="text-xs text-slate-600 block font-sans">Falsos Positivos (FP)</span>
                <span className="text-2xl font-bold text-amber-700">
                  {metrics.confusion_matrix.false_positives}
                </span>
                <span className="text-[10px] text-amber-600/80 block mt-1">Legítimas bloqueadas</span>
              </div>
              <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl">
                <span className="text-xs text-slate-600 block font-sans">Falsos Negativos (FN)</span>
                <span className="text-2xl font-bold text-rose-700">
                  {metrics.confusion_matrix.false_negatives}
                </span>
                <span className="text-[10px] text-rose-600/80 block mt-1">Fraudes no detectados</span>
              </div>
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-xl">
                <span className="text-xs text-slate-600 block font-sans">Verdaderos Positivos (TP)</span>
                <span className="text-2xl font-bold text-blue-700">
                  {metrics.confusion_matrix.true_positives}
                </span>
                <span className="text-[10px] text-blue-600/80 block mt-1">Fraudes interceptados</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: CÓMO FUNCIONA */}
      {activeTab === "about" && (
        <div className="bg-white border border-slate-200 p-6 rounded-2xl space-y-4 text-slate-600 text-sm leading-relaxed shadow-sm">
          <h2 className="text-xl font-bold text-slate-900 mb-2">
            Fundamentos del Sistema de Seguridad con DBSCAN
          </h2>
          <p>
            En seguridad informática y financiera, los atacantes cambian constantemente sus estrategias,
            haciendo que los modelos supervisados se vuelvan obsoletos rápidamente. Este sistema emplea{" "}
            <strong className="text-blue-600">Aprendizaje No Supervisado</strong> con el algoritmo{" "}
            <strong className="text-slate-900">DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</strong>.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 my-4">
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <h4 className="font-semibold text-slate-900 mb-1">1. Reducción PCA</h4>
              <p className="text-xs text-slate-500">
                Se comprimen las 30 características originales en 10 componentes principales que retienen la mayor
                varianza, eliminando la maldición de la dimensionalidad.
              </p>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <h4 className="font-semibold text-slate-900 mb-1">2. Core Samples</h4>
              <p className="text-xs text-slate-500">
                DBSCAN descubre las zonas donde el comportamiento normal de los usuarios es denso y regular. Los
                puntos centrales de estas zonas son los <em>Core Samples</em>.
              </p>
            </div>
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <h4 className="font-semibold text-slate-900 mb-1">3. Regla del Épsilon (ε)</h4>
              <p className="text-xs text-slate-500">
                Toda transacción nueva calcula su distancia euclidiana al core sample más cercano. Si está a más de{" "}
                <strong className="text-slate-900">ε = 2.5</strong>, se declara automáticamente como <strong className="text-rose-600">FRAUDE</strong> (anomalía de baja densidad).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Historial de la Sesión */}
      {history.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4 shadow-sm">
          <h3 className="text-md font-semibold text-slate-900">Historial de Predicciones en la Sesión</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-600">
              <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] border-b border-slate-200">
                <tr>
                  <th className="py-2.5 px-3">Monto</th>
                  <th className="py-2.5 px-3">Tiempo (s)</th>
                  <th className="py-2.5 px-3">Distancia al Core</th>
                  <th className="py-2.5 px-3">Score Anomalía</th>
                  <th className="py-2.5 px-3">Resultado</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {history.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/80 transition">
                    <td className="py-2.5 px-3 text-slate-900 font-bold">${item.amount.toFixed(2)}</td>
                    <td className="py-2.5 px-3">{item.time}</td>
                    <td className="py-2.5 px-3">{item.distance_to_nearest_core}</td>
                    <td className="py-2.5 px-3">{item.anomaly_score}</td>
                    <td className="py-2.5 px-3 font-sans">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          item.is_fraud
                            ? "bg-rose-50 text-rose-700 border border-rose-200"
                            : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                        }`}
                      >
                        {item.label}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

