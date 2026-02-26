# nozzle-design-benchmark

Framework computacional para diseño y optimización de toberas supersónicas convergente-divergente usando inteligencia de enjambre (Swarm Intelligence) y validación RANS con OpenFOAM.

| Componente | Descripción |
|---|---|
| **MOC** | Geometría base por Método de las Características (referencia clásica) |
| **Optimización CI** | Ensemble de 4 algoritmos de inteligencia computacional: PSO, MOPSO-LF, Firefly, ABC |
| **Multi-objetivo** | Frente de Pareto 3D (empuje ↑, pérdida de presión ↓, longitud ↓) con selección de rodilla |
| **RANS** | Validación de alta fidelidad con OpenFOAM shockFluid para los diseños Pareto |
| **Benchmark** | Comparación MOC vs optimizada con el mismo evaluador CFD |

> Este proyecto es un framework de análisis computacional, no una GUI. Se ejecuta por terminal con archivos de configuración JSON.

---

## Requisitos

### Para correr con evaluador quasi-1D (sin Docker)
```
Python 3.11+
pip install -r requirements.txt
```
Funciona en Windows, Linux y macOS. No necesita OpenFOAM.

### Para correr con OpenFOAM RANS 2D (alta fidelidad)
```
Docker Desktop instalado y corriendo
```
El contenedor ya incluye OpenFOAM 13. No necesitas instalarlo en el host.

---

## Uso rápido

### 1. Test local — sin Docker, ~5 segundos
Verifica que el pipeline completo funciona (MOC → optimización → benchmark → plots):
```bash
python main_pipeline.py --config docs/local_quick_test.json
```
Usa el evaluador quasi-1D. Ideal para desarrollo y verificación de cambios.

### 2. Ensemble CI (4 algoritmos) — sin Docker, ~30 segundos
Corre los 4 algoritmos de swarm intelligence en modo rápido:
```bash
python main_ensemble_pipeline.py --quick
```
PSO + MOPSO-LF + Firefly + ABC compiten en un espacio de diseño 5D. Análisis Pareto tri-objetivo.

### 3. Ensemble CI + RANS — requiere Docker, ~20-40 min
Test de referencia: screening Q1D → validación RANS de frente de Pareto (rodilla + mejor-por-pérdida + top-empuje):
```bash
docker compose build                    # una sola vez
docker compose run --rm nozzle python3 main_ensemble_pipeline.py \
    --config docs/rans_2d_ensemble.json
```

### 4. RANS 2D con pipeline clásico (GA) — requiere Docker
```bash
docker compose run --rm nozzle python3 main_pipeline.py \
    --config docs/rans_2d_quick.json
```

---

## Pipelines disponibles

| Pipeline | Archivo | Algoritmos | Descripción |
|---|---|---|---|
| **Clásico** | `main_pipeline.py` | Random / GA | Pipeline original con optimización evolutiva |
| **Swarm** | `main_swarm_pipeline.py` | PSO | Pipeline con Particle Swarm Optimisation |
| **Ensemble CI** | `main_ensemble_pipeline.py` | PSO + MOPSO-LF + Firefly + ABC | 4 algoritmos compiten, selección Pareto 3-objetivo |

### Fases del Ensemble Pipeline
1. **MOC Baseline** — Geometría de referencia por Prandtl-Meyer isentrópico
2. **Ensemble CI Race** — Los 4 algoritmos exploran el espacio 5D con evaluador Q1D
3. **RANS Validation** — OpenFOAM shockFluid en: MOC, CI-best, rodilla Pareto, mejor-por-pérdida, top-K
4. **Multi-fidelity Table** — Comparación Q1D vs RANS con gap de fidelidad
5. **Plots + Benchmark** — Diagramas comparativos, frente de Pareto, radar, heatmaps

### Algoritmos de Inteligencia Computacional

| Algoritmo | Módulo | Tipo | Objetivos |
|---|---|---|---|
| **PSO** | `SwarmOptimizer.py` | Single-objective | Empuje (escalar) |
| **MOPSO-LF** | `MOPSO.py` | Multi-objetivo (Pareto) | Empuje ↑, pérdida ↓, longitud ↓ |
| **Firefly** | `FireflyOptimizer.py` | Single-objective | Empuje (escalar) |
| **ABC** | `ABCOptimizer.py` | Multi-objetivo (Pareto) | Empuje ↑, pérdida ↓, longitud ↓ |

**MOPSO-LF**: Lévy flights para saltar óptimos locales, archivo externo con crowding distance (NSGA-II), factor de constricción χ.

**ABC** (Artificial Bee Colony): Abejas empleadas explotan fuentes conocidas, abejas observadoras seleccionan proporcionalmente al fitness (roulette-wheel), abejas exploradoras abandonan fuentes agotadas con vuelos Lévy. Archivo Pareto compartido.

---

## Resultados

Los resultados se guardan en `out/<nombre_del_caso>/`, configurado por `out_dir` en el JSON.

### Archivos de salida (ensemble pipeline)

| Archivo | Contenido |
|---|---|
| `summary.json` | Estado, paths, comparación, fidelity table, Pareto |
| `report.md` / `report.json` | Reporte completo con todas las métricas |
| `ensemble/ensemble_summary.json` | Resultados por algoritmo, ganador, Pareto |
| `ensemble/multiobjective/` | Frente de Pareto, rodilla, ranking por score |
| `rans/` | Resultados RANS de cada candidato validado |
| `comp/fidelity_table.json` | Tabla multi-fidelidad Q1D vs RANS |
| `moc/moc_characteristics.png` | Red de líneas características MOC |
| `ensemble/<algo>/` | Plots de convergencia por algoritmo |
| `ensemble/ensemble_*.png` | Comparación entre algoritmos (radar, heatmap, etc.) |
| `comp/compare_*.png` | Comparación MOC vs optimizada |
| `ensemble/multiobjective/pareto_*.png` | Gráficos del frente de Pareto |

---

## Configuración (archivos JSON)

### Configs disponibles

| Archivo | Pipeline | Backend | Duración estimada | Uso |
|---|---|---|---|---|
| `local_quick_test.json` | Clásico | quasi-1D | ~5 s | Verificación local, sin Docker |
| `rans_2d_quick.json` | Clásico | OpenFOAM 2D | ~10-20 min | Smoke RANS rápido |
| `rans_2d_ensemble.json` | Ensemble CI | OpenFOAM 2D | ~20-40 min | **4 algoritmos + RANS Pareto** |
| `design_supersonic.json` | Clásico | OpenFOAM 2D | ~2-6 h | Campaña de diseño (casi shock-free) |
| `overexpanded_sea_level.json` | Clásico | OpenFOAM 2D | ~2-6 h | Campaña sobreexpandida |
| `rans_2d_nightly_smoke.json` | Clásico | OpenFOAM 2D | ~15-40 min | Smoke nightly CI |

### Bloque `moc`
```json
"moc": {
  "mach_exit": 2.4,
  "pressure_ratio": 0.09,
  "gamma": 1.4,
  "geometry": {
    "throat_y": 0.02,
    "exit_y": 0.062,
    "length": 0.22,
    "n_points": 120
  }
}
```
- `mach_exit` — Mach de diseño en la salida
- `pressure_ratio` — relación p_exit / p0 de diseño
- `throat_y` — radio de garganta [m]
- `exit_y` — radio de salida [m]

### Bloque `evaluator`
```json
"evaluator": {
  "backend": "quasi1d",
  "gamma": 1.4,
  "stagnation_pressure": 600000.0,
  "stagnation_temperature": 1000.0,
  "ambient_pressure": 80000.0,
  "n_samples": 60,
  "fallback_on_failure": false,
  "use_gpu": false
}
```

| Parámetro | Valores | Descripción |
|---|---|---|
| `backend` | `"quasi1d"` / `"openfoam"` | Motor de evaluación |
| `campaign` | `"design_supersonic"` / `"overexpanded_sea_level"` | Régimen físico objetivo |
| `fallback_on_failure` | `true` / `false` | Recomendado `false` en campañas oficiales para evitar éxitos falsos |
| `require_rans_converged` | `true` / `false` | Exige backend OpenFOAM convergido |
| `require_converged_series` | `true` / `false` | Exige convergencia por serie temporal de salida |
| `allow_unsteady_overexpanded` | `true` / `false` | En campaña overexpanded acepta series no-estacionarias dentro de límites |
| `use_window_averages_overexpanded` | `true` / `false` | Usa promedio temporal de ventana para métricas de salida en overexpanded |
| `min_writes` / `convergence_window` | int | Control de ventana para convergencia temporal |
| `convergence_tol` | object | Tolerancias relativas de `p_out`, `mdot`, `Ux_out` |
| `overexpanded_unsteady_tol` | object | Límites máximos de oscilación permitidos en overexpanded |
| `enable_delta_t_abort` / `delta_t_abort_threshold` / `delta_t_abort_streak` | bool/float/int | Abortado temprano si `deltaT` colapsa en shockFluid |
| `outlet_bc_mode` | `"wave_transmissive"` / `"fixed_pressure"` | BC de salida según régimen |
| `sampling_nx` | int | Número de puntos muestreados en línea de centro |
| `use_gpu` | `true` / `false` | Habilita CuPy para el solver quasi-1D (requiere CuPy instalado) |
| `dimension` | `"2d_planar"` | Modo de simulación OpenFOAM (solo 2D recomendado) |
| `mesh_nx` / `mesh_ny` | int | Resolución de malla (afecta tiempo y fidelidad) |
| `end_time` | int | Iteraciones máximas del solver RANS |

**Para OpenFOAM RANS** se necesitan además:
```json
"dimension": "2d_planar",
"depth": 0.02,
"mesh_nx": 80, "mesh_ny": 36,
"end_time": 500, "write_interval": 100,
"inlet_velocity": 10.0,
"k_inlet": 0.5, "epsilon_inlet": 10.0,
"p_initial": 270000.0, "u_initial": 1.0, "t_initial": 900.0
```

### Bloque `optimization`
```json
"optimization": {
  "algorithm": "random",
  "seed": 42,
  "bounds": {
    "exit_radius": [0.055, 0.075],
    "length": [0.20, 0.28],
    "shape": [1.4, 2.2]
  },
  "n_samples": 5,
  "n_points": 120,
  "profile": "bezier_like",
  "w_thrust": 1.0,
  "w_pressure_loss": 0.0,
  "use_pareto_knee": false,
  "n_workers": 1
}
```

| Parámetro | Descripción |
|---|---|
| `strategy` | `"single_fidelity"` / `"multifidelity_screen"` |
| `algorithm` | `"random"` (recomendado para empezar), `"ga"` (evolutivo) |
| `n_samples` | Número de candidatos a evaluar. ⚠️ Con RANS cada candidato cuesta tiempo real |
| `low_fidelity_samples` / `high_fidelity_top_k` | Tamaño de screening quasi-1D y shortlist RANS |
| `objective_terms` | Penalizaciones por choque, oscilación de outlet y RMS de presión de pared |
| `w_thrust` / `w_pressure_loss` | Pesos del objetivo multi-criterio (thrust maximize, loss minimize) |
| `use_pareto_knee` | `true` = usar geometría de la rodilla del frente de Pareto como ganadora |
| `n_workers` | Evaluaciones en paralelo. Con quasi-1D se puede subir (ej. 4). Con RANS dejar en 1 |

Ejemplo de `objective_terms`:
```json
"objective_terms": {
  "design_shock_penalty": 0.10,
  "overexpanded_instability_penalty": 0.05,
  "overexpanded_outlet_osc_penalty": 0.05,
  "overexpanded_wall_rms_penalty": 0.03
}
```

---

## Cómo pasar de quasi-1D a RANS

### Paso 1 — Verificar localmente
```bash
python main_pipeline.py --config docs/local_quick_test.json
```
Si `status: ok` en `summary.json`, el pipeline funciona.

### Paso 2 — Levantar Docker
```bash
docker compose build          # solo la primera vez
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_2d_quick.json
```

### Paso 3 — Forzar RANS estricto (sin fallback)
Para asegurarse de que todos los resultados sean RANS-convergidos:
```json
"evaluator": {
  "backend": "openfoam",
  "fallback_on_failure": false,
  "require_rans_converged": true,
  "require_converged_series": true
}
```
Con estos flags, el pipeline rechazará candidatos sin convergencia RANS y sin estabilidad temporal en series de salida.

---

## Aceleración GPU (quasi-1D)

El evaluador quasi-1D soporta GPU a través de **CuPy** (reemplazo de NumPy en GPU):

```bash
# CUDA 12.x (GPUs recientes — RTX 3xxx / 4xxx / A series)
pip install cupy-cuda12x

# CUDA 11.x (GPUs más antiguas)
pip install cupy-cuda11x
```

Luego en el JSON:
```json
"evaluator": {
  "backend": "quasi1d",
  "use_gpu": true
}
```

Si CuPy no está instalado y `use_gpu: true`, cae silenciosamente a NumPy (CPU). Verificar:
```bash
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

> **Nota:** La GPU solo acelera el evaluador quasi-1D. OpenFOAM RANS corre en CPU dentro del contenedor Docker — la GPU no aplica ahí.

---

## Diagnóstico de fallos OpenFOAM

Si un caso falla con `"status": "failed"`:

1. Revisar `out/.../openfoam_cases/<id>/log.shockFluid` — buscar `FATAL` o `divergence`
2. Revisar `out/.../openfoam_cases/<id>/log.checkMesh` — buscar celdas con skewness alta

Ajustes típicos para estabilizar:
- Reducir `mesh_nx`/`mesh_ny` (malla más gruesa = más estable)
- Reducir `end_time` (menos iteraciones si ya convergió antes)
- Estrechar `bounds` de optimización (geometrías menos extremas)
- Subir `fallback_on_failure: true` para exploración inicial

---

## Arquitectura del proyecto

```
main_pipeline.py              ← Pipeline clásico (Random/GA)
main_swarm_pipeline.py        ← Pipeline PSO
main_ensemble_pipeline.py     ← Pipeline ensemble CI (4 algoritmos)
│
├── geometry/
│   ├── MOCSolver.py          ← Genera geometría MOC
│   ├── NozzleGeometry.py     ← Representación y export de geometría
│   └── MinimumLengthNozzle.py ← Tobera de longitud mínima (MLN)
│
├── evaluators/
│   ├── CFDSimulation.py          ← Solver quasi-1D vectorizado (NumPy/CuPy)
│   ├── OpenFOAMRANSEvaluator.py  ← OpenFOAM nativo (shockFluid) + muestreo CFD
│   ├── EvaluationResult.py       ← Resultado de evaluación
│   └── FastEvaluator.py          ← Evaluador heurístico (screening)
│
├── optimization/
│   ├── Optimizer.py          ← Random / GA + multi-objetivo Pareto
│   ├── OptimizationRunner.py ← Runner para pipeline clásico
│   ├── SwarmOptimizer.py     ← PSO con inercia adaptativa
│   ├── MOPSO.py              ← MOPSO con Lévy flights (3 objetivos)
│   ├── FireflyOptimizer.py   ← Firefly Algorithm
│   ├── ABCOptimizer.py       ← Colonia Artificial de Abejas (3 objetivos)
│   └── EnsembleRunner.py     ← Meta-optimizador ensemble (4 algoritmos)
│
├── benchmarks/
│   └── BenchmarkSuite.py     ← MOC vs optimizada con el mismo evaluador
│
├── analysis/
│   ├── OptimizationPlots.py  ← Plots de convergencia
│   ├── ComparisonPlots.py    ← Plots MOC vs OPT
│   ├── SwarmPlots.py         ← Plots de enjambre (PSO/ABC/Firefly)
│   ├── EnsemblePlots.py      ← Plots comparativos ensemble
│   └── AnalysisNote.py       ← Generador de reporte markdown/JSON
│
├── scripts/
│   └── multiobjective_rank.py ← Análisis Pareto + rodilla + ranking
│
└── docs/
    └── *.json                 ← Configs de referencia
```

---

## Tiempos de referencia

| Config | Pipeline | Backend | Malla | Tiempo estimado |
|---|---|---|---|---|
| `local_quick_test.json` | Clásico | quasi-1D | — | ~5 s |
| `main_ensemble --quick` | Ensemble CI | quasi-1D | — | ~30 s |
| `rans_2d_quick.json` | Clásico | OpenFOAM 2D | 80×36 | ~10-20 min |
| `rans_2d_ensemble.json` | Ensemble CI | OpenFOAM 2D | 100×45 | ~20-40 min |
| `design_supersonic.json` | Clásico | OpenFOAM 2D | 120×54 | ~2-6 h |
| `overexpanded_sea_level.json` | Clásico | OpenFOAM 2D | 160×72 | ~2-6 h |

> Con quasi-1D y `n_workers > 1`, el tiempo escala casi linealmente (4 workers ≈ 4× más rápido).  
> Con OpenFOAM RANS dejar `n_workers: 1` — cada run ocupa todos los cores disponibles.  
> El ensemble CI corre los 4 algoritmos secuencialmente; cada uno usa swarm Q1D (rápido), luego solo los mejores se validan con RANS.

---

## Criterio de migración a SU2

La migración se recomienda solo si, tras estabilizar este pipeline:

1. La tasa de no-convergencia RANS supera 30%.
2. La variabilidad run-to-run en thrust supera 3% con misma geometría/config.
3. No se logra estabilizar o caracterizar el choque en campaña `overexpanded_sea_level`.

---

## Limpieza

```bash
scripts\clean_workspace.cmd
```

---

## Licencia

MIT
