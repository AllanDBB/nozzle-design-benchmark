# nozzle-design-benchmark

Framework computacional para comparar diseño de toberas supersónicas por tres enfoques:

| Componente | Descripción |
|---|---|
| **MOC** | Geometría base por Método de las Características (referencia clásica) |
| **Optimización** | Búsqueda paramétrica de geometría (random, GA) con evaluador quasi-1D o RANS |
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

### 2. RANS 2D con OpenFOAM — requiere Docker
```bash
# Una sola vez: construir imagen
docker compose build

# Correr pipeline completo con RANS 2D
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_2d_quick.json
```
Cada candidato tarda ~15-40 min dependiendo de la malla y convergencia.

---

## Resultados

Los resultados se guardan en `out/<nombre_del_caso>/`, configurado por `out_dir` en el JSON.

| Archivo | Contenido |
|---|---|
| `summary.json` | Estado, paths, comparación resumida |
| `comparison.json` | Delta de empuje y pérdida de presión MOC vs optimizada |
| `report.md` / `report.json` | Reporte completo con todos los métricas |
| `moc_geometry.csv` | Puntos de control de la geometría MOC |
| `optimized_geometry.csv` | Puntos de control de la geometría optimizada |
| `moc_characteristics.png` | Líneas características MOC |
| `optimization_thrust_history.png` | Convergencia del empuje durante optimización |
| `optimization_pressure_loss_history.png` | Convergencia de pérdida de presión |
| `optimization_best_so_far_*.png` | Evolución del mejor candidato (Mach, velocidad, geometría) |
| `compare_*.png` | Comparación MOC vs optimizada (geometría, Mach, velocidad, presión, temperatura, performance) |
| `multiobjective/pareto_*.png` | Frente de Pareto (si hay ≥2 candidatos) |

---

## Configuración (archivos JSON)

### Configs disponibles

| Archivo | Backend | Duración estimada | Uso |
|---|---|---|---|
| `docs/local_quick_test.json` | quasi-1D | ~5 s | Verificación local, sin Docker |
| `docs/rans_2d_quick.json` | OpenFOAM 2D | ~1-3 h | Test RANS mínimo con Docker |
| `docs/example_config.json` | OpenFOAM 2D | ~2-5 h | Config completa de referencia |

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
| `fallback_on_failure` | `true` / `false` | `true` = si OpenFOAM falla, usa quasi-1D como respaldo. `false` = estricto RANS, aborta si falla |
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
| `algorithm` | `"random"` (recomendado para empezar), `"ga"` (evolutivo) |
| `n_samples` | Número de candidatos a evaluar. ⚠️ Con RANS cada candidato cuesta tiempo real |
| `w_thrust` / `w_pressure_loss` | Pesos del objetivo multi-criterio (thrust maximize, loss minimize) |
| `use_pareto_knee` | `true` = usar geometría de la rodilla del frente de Pareto como ganadora |
| `n_workers` | Evaluaciones en paralelo. Con quasi-1D se puede subir (ej. 4). Con RANS dejar en 1 |

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
  "require_rans_converged": true
}
```
Con `require_rans_converged: true` el pipeline rechazará candidatos que no hayan convergido en OpenFOAM y abortará.

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

1. Revisar `out/.../openfoam_cases/<id>/log.rhoSimpleFoam` — buscar `FATAL` o `divergence`
2. Revisar `out/.../openfoam_cases/<id>/log.checkMesh` — buscar celdas con skewness alta

Ajustes típicos para estabilizar:
- Reducir `mesh_nx`/`mesh_ny` (malla más gruesa = más estable)
- Reducir `end_time` (menos iteraciones si ya convergió antes)
- Estrechar `bounds` de optimización (geometrías menos extremas)
- Subir `fallback_on_failure: true` para exploración inicial

---

## Arquitectura del proyecto

```
main_pipeline.py          ← único entrypoint
│
├── geometry/
│   ├── MOCSolver.py      ← Genera geometría MOC
│   └── NozzleGeometry.py ← Representación y export de geometría
│
├── evaluators/
│   ├── CFDSimulation.py          ← Solver quasi-1D vectorizado (NumPy/CuPy)
│   ├── OpenFOAMRANSEvaluator.py  ← Wrapper Docker + rhoSimpleFoam
│   ├── EvaluationResult.py       ← Resultado de evaluación
│   └── FastEvaluator.py          ← Evaluador heurístico (screening)
│
├── optimization/
│   ├── Optimizer.py        ← Random / GA + multi-objetivo Pareto
│   └── OptimizationRunner.py
│
├── benchmarks/
│   └── BenchmarkSuite.py   ← MOC vs optimizada con el mismo evaluador
│
├── analysis/
│   ├── OptimizationPlots.py  ← Plots de convergencia
│   ├── ComparisonPlots.py    ← Plots MOC vs OPT
│   └── AnalysisNote.py       ← Generador de reporte markdown/JSON
│
└── scripts/
    └── multiobjective_rank.py  ← Análisis Pareto post-optimización
```

---

## Tiempos de referencia

| Config | Backend | Malla | n_samples | Tiempo estimado |
|---|---|---|---|---|
| `local_quick_test.json` | quasi-1D | — | 5 | ~5 s |
| `rans_2d_quick.json` | OpenFOAM 2D | 80×36 | 3 | ~1-2 h |
| `example_config.json` | OpenFOAM 2D | 120×50 | 3 | ~2-5 h |

> Con `n_workers > 1` en quasi-1D el tiempo escala casi linealmente (4 workers ≈ 4× más rápido).  
> Con OpenFOAM RANS dejar `n_workers: 1` — cada run ocupa todos los cores disponibles.

---

## Limpieza

```bash
scripts\clean_workspace.cmd
```

---

## Licencia

MIT
