# Optimizer

Optimization driver over a set of candidate geometries.

## Fields

- `searchSpace`: Dictionary containing `candidates` and optional `n_samples`.
- `objectiveFunc`: Function returning `EvaluationResult`.
- `algorithm`: Strategy (`random` or full sweep).

## Methods

### `run()`

Evaluates candidates and returns the best result by thrust.

### `logBest()`

Prints the best result from the last run.
