# EvaluationResult

Container for evaluation outputs.

## Fields

- `machProfile`: List of Mach values along the nozzle.
- `pressureLoss`: Scalar pressure loss estimate.
- `thrust`: Scalar thrust estimate.
- `geometryId`: Identifier for the geometry.

## Methods

### `compareWith(other)`

Returns a dictionary of deltas between two results.

### `saveToJSON(filepath)`

Writes the result to a JSON file.
