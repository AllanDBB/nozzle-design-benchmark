# MOCSolver

Simplified Method of Characteristics generator for nozzle wall shapes.

## Purpose

Produces a physics-inspired wall profile using Prandtl-Meyer expansion. It is not a full MOC solver, but follows the same interface and conventions.

## Key Fields

- `machExit`: Target exit Mach number.
- `pressureRatio`: Reference pressure ratio (stored in metadata).
- `nCharacteristics`: Number of characteristic samples.

## Methods

### `computeCharacteristics()`

Returns a list of characteristic data with Mach, Prandtl-Meyer angle $\nu$, and wall angle $\theta$.

### `generateGeometry(params)`

Generates a smooth MOC-like geometry using a wall angle distribution. Parameters:

- `throat_y`, `exit_y`, `length`, `n_points`
- optional `gamma`
