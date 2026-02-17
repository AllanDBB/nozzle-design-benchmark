# nozzle-design-benchmark

Framework computacional para comparar diseno de toberas supersónicas por tres enfoques:
- Geometría base por Metodo de las Caracteristicas (MOC)
- Evaluación de flujo por RANS (OpenFOAM, 2D/3D)
- Búsqueda de geometría por optimización

Este proyecto NO es una GUI. Se ejecuta por comandos, pero esta guia esta pensada para alguien que domina el tema de toberas y no necesariamente programación.

## 1) Que hace el proyecto
Objetivo principal:
- Generar una tobera MOC (referencia clásica)
- Buscar una tobera alternativa (optimizada)
- Compararlas con el mismo evaluador CFD

Salida principal:
- Empuje estimado
- Perdida de presión total
- Comparación MOC vs optimizada
- Geometrías y reportes guardados en `out/...`
- Graficas comparativas automáticas (incluye velocidad a lo largo del eje)

## 2) Lo minimo que necesitas
- Docker Desktop instalado y abierto
- Estar en la carpeta del proyecto en terminal

No necesitas instalar OpenFOAM en Windows host: el contenedor ya lo trae.

## 3) Primer arranque (solo una vez)
Construir imagen:
```bash
docker compose build
```

## 4) Como ejecutar (recomendado)
### Caso 2D rapido (estable para pruebas)
```bash
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_2d_quick.json
```

### Caso 3D rapido (estable para pruebas)
```bash
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_3d_quick.json
```

Estos son los perfiles recomendados para verificar flujo completo de trabajo.

## 5) Donde quedan los resultados
### 2D
- `out/test_rans_2d/summary.json`
- `out/test_rans_2d/comparison.json`
- `out/test_rans_2d/report.md`
- `out/test_rans_2d/moc_geometry.csv`
- `out/test_rans_2d/optimized_geometry.csv`
- `out/test_rans_2d/compare_velocity.png`
- `out/test_rans_2d/compare_mach.png`
- `out/test_rans_2d/compare_pressure.png`
- `out/test_rans_2d/compare_temperature.png`
- `out/test_rans_2d/compare_geometry.png`
- `out/test_rans_2d/compare_performance.png`

### 3D
- `out/test_rans_3d/summary.json`
- `out/test_rans_3d/comparison.json`
- `out/test_rans_3d/report.md`
- `out/test_rans_3d/moc_geometry.csv`
- `out/test_rans_3d/optimized_geometry.csv`
- `out/test_rans_3d/compare_velocity.png`
- `out/test_rans_3d/compare_mach.png`
- `out/test_rans_3d/compare_pressure.png`
- `out/test_rans_3d/compare_temperature.png`
- `out/test_rans_3d/compare_geometry.png`
- `out/test_rans_3d/compare_performance.png`

## 6) Como leer el resultado rapido
Abre `summary.json`.

Campos clave:
- `status`:
  - `ok`: corrida completa
  - `failed`: hubo fallo de solver en benchmark
- `comparison`:
  - `delta_thrust`: diferencia de empuje (optimizada - MOC)
  - `delta_pressure_loss`: diferencia de pérdida total (optimizada - MOC)
  - `moc_thrust`, `optimized_thrust`
  - `moc_pressure_loss`, `optimized_pressure_loss`

## 7) Configuracion: que puedes cambiar y para que
Todos los ajustes se hacen en archivos JSON (`docs/*.json`).

### Bloque `moc`
Define la geometría base MOC:
- `mach_exit`
- `geometry.throat_y`, `geometry.exit_y`, `geometry.length`
- `geometry.n_points`

### Bloque `evaluator` (CFD)
Define condiciones y tipo de simulación:
- `backend`: usar `openfoam`
- `dimension`: `2d_planar` o `3d_channel`
- `stagnation_pressure`, `ambient_pressure`, `stagnation_temperature`
- `mesh_nx`, `mesh_ny`, `mesh_nz` (solo 3D)
- `end_time`, `write_interval`
- `fallback_on_failure`:
  - `true`: robusto para exploración (si un caso falla, usa respaldo)
  - `false`: estricto RANS (si falla, aborta)

Nota sobre 3D:
- `3d_channel` usa paredes laterales en z (no simetría), es decir un canal 3D extruido real.
- Sigue siendo una geometría extruida (no revolución axisimétrica completa).

### Bloque `optimization`
Define cuanto explora la búsqueda:
- `bounds.exit_radius`, `bounds.length`, `bounds.shape`
- `n_samples` (cuantos candidatos se evalúan)
- `algorithm` (recomendado `random` para empezar)

## 8) Que deberias esperar (tiempos y comportamiento)
- 2D quick: normalmente minutos
- 3D quick: más lento que 2D
- Si subes malla o `n_samples`, sube tiempo de forma importante

En CFD compresible, es normal que algunos candidatos no converjan.
Por eso se recomienda:
1. empezar con `quick`
2. luego estrechar rangos de diseño
3. después probar modo estricto

## 9) Modo estricto RANS (solo si quieres forzar todo CFD)
Archivos:
- `docs/rans_2d_strict.json`
- `docs/rans_3d_strict_v2.json`

Ejecución:
```bash
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_2d_strict.json
docker compose run --rm nozzle python3 main_pipeline.py --config docs/rans_3d_strict_v2.json
```

Nota: hoy estos perfiles pueden divergir según geometría/condición. Son para validación dura, no para primera corrida.

## 10) Si falla: que mirar primero
1. `summary.json` -> `status`
2. Si `failed`, revisa `comparison.error`
3. Revisa logs del caso en:
   - `out/.../openfoam_cases/<id>/log.rhoSimpleFoam`
   - `out/.../openfoam_cases/<id>/log.checkMesh`

Ajustes típicos para estabilizar:
- bajar `mesh_nx/mesh_ny/mesh_nz`
- reducir `n_samples`
- estrechar `bounds`
- acercar condiciones (`stagnation_pressure` vs `ambient_pressure`)

## 11) Limpieza de resultados
Para borrar artefactos generados y empezar limpio:
```bash
scripts\clean_workspace.cmd
```

## 12) Comandos utiles
Pipeline con config personalizada:
```bash
docker compose run --rm nozzle python3 main_pipeline.py --config docs/mi_config.json
```

Solo optimización:
```bash
docker compose run --rm nozzle python3 main_optimizer.py
```

Solo benchmark:
```bash
docker compose run --rm nozzle python3 main_benchmark.py
```

## 13) Estado actual del proyecto
- Flujo completo 2D/3D funcional con perfiles `quick`
- Modo estricto RANS disponible, pero requiere más ajuste numérico para convergencia robusta en todos los casos

## Licencia
MIT
