# FastEvaluator

Surrogate evaluator for rapid scoring of geometries.

## Purpose

Provides fast predictions using a stored model or a simple isentropic heuristic.

## Key Fields

- `model`: Optional ML model with `predict`.
- `gamma`: Heat capacity ratio for isentropic fallback.

## Methods

### `predict(geometry)`

Returns an `EvaluationResult` based on the model or a heuristic.

### `train(dataset)`

Stores a dataset for future model fitting.
