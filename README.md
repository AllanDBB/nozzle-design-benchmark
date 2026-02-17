# nozzle-design-benchmark

Framework computacional para comparar diseno de toberas supersonicas por:
- Metodo de las Caracteristicas (MOC)
- Evaluacion RANS 2D/3D con OpenFOAM (referencia de mayor fidelidad)
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

## Docker (OpenFOAM + Python)
Si prefieres no instalar solver en tu host, usa contenedor.

1. Inicia Docker Desktop.
2. Construye la imagen:
```bash
docker compose build
```
3. Ejecuta el pipeline:
```bash
docker compose run --rm nozzle
```

Comandos utiles dentro de Docker:
```bash
docker compose run --rm nozzle python3 main_optimizer.py
docker compose run --rm nozzle python3 main_benchmark.py
docker compose run --rm nozzle bash -lc "which foamRun && foamRun -help"
```

Los resultados quedan en tu carpeta local `out/` porque el proyecto se monta como volumen.

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

## Modo RANS (por defecto)
Ahora el pipeline esta configurado en modo RANS OpenFOAM por defecto.

- 2D RANS (planar): usar `docs/example_config.json`
- 3D RANS (canal extruido): usar `docs/rans_3d_config.json`

Ejemplos:
```bash
docker compose run --rm nozzle python3 main_pipeline.py --config docs/example_config.json
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_3d_config.json
```

Parametros clave en `evaluator`:
- `backend`: `openfoam`
- `dimension`: `2d_planar` o `3d_channel`
- `fallback_on_failure`: `false` para modo estricto RANS
- `mesh_nx`, `mesh_ny`, `mesh_nz`, `end_time`, `write_interval`
- condiciones: `stagnation_pressure`, `ambient_pressure`, `stagnation_temperature`

Si una corrida estricta diverge (caso frecuente en optimizacion automatica de CFD compresible),
puedes temporalmente usar:
- `fallback_on_failure: true` para no cortar el pipeline,
- y luego relanzar solo las mejores geometrías en modo estricto.

Adicionalmente, el optimizador penaliza candidatos que fallan en RANS (en lugar de abortar toda la corrida).

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
- `OpenFOAMRANSEvaluator` corre `rhoSimpleFoam` (RANS compresible + kOmegaSST) sobre una malla construida desde la geometria de tobera.
- Si activas `fallback_on_failure=true`, el sistema usa un respaldo quasi-1D cuando el caso RANS no converge.
- La optimizacion no reemplaza MOC; se usa para obtener una referencia de desempeno bajo el mismo evaluador.

## Licencia
MIT
