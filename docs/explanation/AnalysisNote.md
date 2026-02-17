# AnalysisNote

Simple report object for analysis notes and metrics.

## Fields

- `title`: Report title.
- `notes`: Freeform text.
- `metrics`: Key-value store.

## Methods

### `addMetric(name, value)`

Adds or updates a metric.

### `toMarkdown()`

Returns a Markdown representation.

### `save(filepath)`

Writes the Markdown report to disk.
