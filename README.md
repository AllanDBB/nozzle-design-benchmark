# nozzle-design-benchmark

## Overview

**nozzle-design-benchmark** is a fully computational research framework for benchmarking supersonic nozzle design methodologies.

The project focuses on evaluating the **Method of Characteristics (MOC)** as a preliminary design tool by comparing its results against higher-fidelity **CFD-based evaluations** and against geometries obtained through computational optimization. The goal is not to replace classical analytical methods, but to **quantify their accuracy, limitations, and applicability** under non-ideal flow conditions.

This repository is designed for academic and research use and supports reproducible, methodical comparison between classical theory and numerical optimization.

---

## Motivation

The Method of Characteristics is widely used in the preliminary design of supersonic nozzles due to its simplicity and low computational cost. However, MOC assumes ideal, isentropic flow and imposes specific geometric constraints that may not remain optimal once real-flow effects are considered.

Computational Fluid Dynamics (CFD) allows these non-ideal effects to be modeled but comes at a significantly higher computational cost. This raises fundamental design questions:

- When is MOC sufficiently accurate for preliminary nozzle design?
- How large is the performance penalty when relying solely on MOC?
- Under what conditions does CFD-based analysis become necessary?
- How close is a classical MOC design to a CFD-optimal geometry?

This project addresses these questions by establishing a **computational benchmark** where MOC-based designs are evaluated against CFD results and against an optimized CFD reference nozzle.

---

## Role of Computational Intelligence

Computational Intelligence (CI) methods are used **strictly as an evaluation and benchmarking tool**, not as a black-box replacement for analytical theory.

Optimization algorithms explore a parametrized nozzle geometry space and use CFD as a high-fidelity evaluator. The resulting optimized geometry represents a **performance reference** under the chosen modeling assumptions.

This reference allows the performance of MOC-derived geometries to be objectively measured and contextualized.

In this framework:
- CI provides a performance upper bound.
- CFD provides physical fidelity.
- MOC provides a classical baseline.

---

## Scope and Assumptions

- The project is **fully computational**.
- No experimental manufacturing or testing is considered.
- Axisymmetric nozzle geometries are assumed.
- Flow solvers may range from simplified evaluators (early stages) to full CFD simulations.
- The framework is modular and extensible by design.

---

## Repository Structure

```text
nozzle-design-benchmark/
│
├── geometry/        # Parametrized nozzle geometry generation and constraints
├── evaluators/      # Flow evaluators (isentropic, CFD interfaces, etc.)
├── optimization/    # Optimization algorithms and search strategies
├── benchmarks/      # Comparative studies (MOC vs CFD vs optimized designs)
├── analysis/        # Post-processing, plots, and paper-ready figures
├── docs/            # Technical notes, assumptions, and design decisions
└── README.md

-- 
## Intended Use

This repository is intended to support:

- Academic research and conference papers

- Methodological studies in propulsion and compressible flow

- Educational exploration of nozzle design tradeoffs

- Reproducible benchmarking of design methodologies

It is not intended as a production design tool for flight hardware.

--
## License MIT
