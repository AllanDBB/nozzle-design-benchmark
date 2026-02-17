# CFDSimulation

Idealized CFD evaluator that returns isentropic estimates.

## Purpose

Provides a lightweight CFD-like interface that produces a Mach profile and thrust estimate using isentropic relations.

## Key Fields

- `geometry`: `NozzleGeometry` to evaluate.
- `solverConfig`: Configuration dictionary (supports `gamma`, `n_samples`).
- `resultPath`: Output path for future results.

## Methods

### `run()`

Executes the evaluation and stores the last result.

### `extractResults()`

Computes Mach exit, pressure ratio, and a linear Mach profile.

### `computeThrust()`

Returns thrust from the most recent evaluation.

### `plotFields()`

Plots the nozzle profile and the Mach profile.
