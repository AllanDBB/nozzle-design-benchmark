This project, nozzle-design-benchmark, is a fully computational research framework focused on the evaluation and comparison of supersonic nozzle design methodologies.

The core objective is to assess the Method of Characteristics (MOC) as a preliminary design tool for supersonic nozzles by comparing its results against higher-fidelity CFD-based evaluations and against geometries obtained through computational optimization. The intent is not to replace classical analytical methods with machine learning, but to quantify their accuracy, limitations, and domains of validity under increasingly realistic flow conditions.

The project treats CFD as a reference evaluator of nozzle performance, capturing non-ideal effects that are neglected in MOC (such as deviations from isentropic behavior, real pressure losses, and non-uniform exit flow). Computational optimization techniques are used solely to explore the geometric design space and to identify a CFD-optimal reference nozzle, which serves as a benchmark against which MOC-based designs can be objectively compared.

The role of Intelligence Computational methods in this project is therefore evaluative and diagnostic, not generative in a black-box sense. Optimization algorithms (e.g., evolutionary strategies or Bayesian optimization) operate on a parametrized nozzle geometry, proposing candidate shapes that are evaluated via CFD. The resulting optimal geometry represents a performance upper bound under the chosen modeling assumptions, enabling a direct measurement of how close—or how far—the MOC design lies from this bound.

Importantly, this framework is designed to be modular and extensible. Geometry generation, flow evaluation, optimization, and post-processing are decoupled, allowing the same pipeline to be used with simplified evaluators (e.g., isentropic relations) during early development and with CFD solvers during high-fidelity analysis. This also allows future extensions, such as sensitivity studies, surrogate modeling, or alternative nozzle design methodologies.

The scope of the project is intentionally limited to computational analysis only. No experimental manufacturing, testing, or propulsion system integration is considered. The goal is to provide a rigorous, reproducible computational benchmark that clarifies:

when the Method of Characteristics is sufficient for preliminary nozzle design,

when higher-fidelity CFD becomes necessary,

and how large the performance penalties are when relying solely on idealized analytical assumptions.

In summary, nozzle-design-benchmark is an engineering-focused computational study that uses optimization and CFD as tools to critically evaluate classical nozzle design theory, rather than to supplant it.