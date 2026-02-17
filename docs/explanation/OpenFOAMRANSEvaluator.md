# OpenFOAMRANSEvaluator

Steady compressible RANS evaluator based on OpenFOAM (`rhoSimpleFoam`).

## Modes
- `2d_planar`: malla 2D equivalente (1 celda en espesor, `empty`).
- `3d_channel`: malla 3D extruida (`symmetryPlane` en front/back).

## Flow
1. Genera caso OpenFOAM por geometria (`blockMeshDict`, `0/`, `constant/`, `system/`).
2. Corre `blockMesh` y `rhoSimpleFoam`.
3. Extrae metricas en salida con `postProcess`:
   - `patchAverage(name=outlet,p)`
   - `patchAverage(name=outlet,mag(U))`
   - `patchAverage(name=outlet,T)`
   - `patchIntegrate(name=outlet,phi)`
4. Calcula empuje y perdidas.

## Notes
- Si OpenFOAM falla para una geometria (inestabilidad numerica), con `fallback_on_failure=true` usa `CFDSimulation` y guarda el error en `metadata.openfoam_error`.
- Para estudios 100% RANS, poner `fallback_on_failure=false` y ajustar BC/malla/solver hasta converger.
