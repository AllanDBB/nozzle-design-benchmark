# OptimizationRunner

Glue layer between `Optimizer` and evaluator.

## Fields

- `optimizer`: Optimizer instance.
- `evaluator`: CFDSimulation instance.
- `bestResult`: Best evaluation result.
- `historyPath`: Output path for history JSON.

## Methods

### `start()`

Runs optimization and stores `bestResult`.

### `saveHistory()`

Serializes evaluation history to JSON.
