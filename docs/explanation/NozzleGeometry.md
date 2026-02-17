# NozzleGeometry

Represents a 2D planar nozzle upper wall as an ordered list of $(x, y)$ points.

## Conventions

- Uses $y$ (not radius $r$): planar interpretation.
- `control_points` are sorted in strictly increasing $x$.
- Range is $x \in [0, L]$ and $y \in [y_t, y_e]$.

## Key Fields

- `control_points`: List of $(x, y)$ upper-wall points.
- `throat_radius`: Throat height $y_t$.
- `exit_radius`: Exit height $y_e$.
- `length`: Nozzle length $L$.
- `metadata`: Freeform metadata.

## Constructors

### `fromMOC(params)`

Builds geometry from a MOC-provided wall profile. Expects `params["wall_points"]`.

### `fromParams(params)`

Builds a smooth, parametric geometry with a cosine profile (non-MOC).

## Validations

- `x` is strictly increasing.
- `y > 0` for all points.
- First point at $(0, y_t)$ and last point at $(L, y_e)$.

## Output

- Ordered list of $(x, y)$ points on the upper wall.
