# nozzle-design-benchmark

Framework computacional para comparar diseno de toberas supersonicas por:
- Metodo de las Caracteristicas (MOC)
- Evaluacion CFD-like (referencia de mayor fidelidad)
- Optimizacion computacional (GA/CMA-ES-like/Bayesian-like)

## Objetivo
Cuantificar cuando MOC es suficiente como diseno preliminar y cuando la fidelidad CFD cambia de forma relevante el desempeno (empuje/perdidas).

## Estructura
- `geometry/`: generacion de geometria (MOC y parametrica)
- `evaluators/`: evaluadores (`CFDSimulation`, `FastEvaluator`)
- `optimization/`: optimizador y runner
- `benchmarks/`: comparacion MOC vs optimizado
- `analysis/`: reporte y metricas
- `main_pipeline.py`: pipeline completo en una ejecucion

## Instalacion
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion rapida
Pipeline completo:
```bash
python main_pipeline.py
```

Los defaults del pipeline representan un caso de tobera de alta expansion en condicion de gran altitud (`ambient_pressure = 10000 Pa`).

Con algoritmo especifico:
```bash
python main_pipeline.py --algorithm ga
python main_pipeline.py --algorithm bayesian
python main_pipeline.py --algorithm cma-es
python main_pipeline.py --algorithm grid
```

Salida personalizada:
```bash
python main_pipeline.py --out out/mi_estudio
```

Con archivo de configuracion JSON:
```bash
python main_pipeline.py --config docs/example_config.json
```

## Artefactos generados
En `out/pipeline/` (o el `--out` elegido):
- `moc_geometry.csv`, `optimized_geometry.csv`
- `moc_geometry.png`, `optimized_geometry.png`
- `moc_result.json`, `optimized_result.json`, `comparison.json`
- `optimization_history.json`, `best_result.json`
- `moc_fields.png`, `optimized_fields.png`
- `report.md`, `report.json`, `summary.json`

## Scripts individuales
- `python main_moc_pipeline.py`
- `python main_optimizer.py`
- `python main_benchmark.py`

## Notas metodologicas
- `CFDSimulation` ahora usa un modelo quasi-1D mas realista:
  - columna vertebral area-Mach compresible,
  - reduccion de area efectiva por capa limite desplazada,
  - perdidas de presion total por friccion y curvatura,
  - correccion por choque normal en salida sobreexpandida (opcional),
  - penalizacion por divergencia de flujo en salida (`eta_div`).
- Parametros fisicos ajustables en `evaluator`: `friction_scale`, `cf_multiplier`, `curvature_scale`, `bl_displacement_scale`, `discharge_coefficient`, `enable_shock_model`, `divergence_scale`.
- La optimizacion no reemplaza MOC; se usa para obtener una referencia de desempeno bajo el mismo evaluador.

## Licencia
MIT
