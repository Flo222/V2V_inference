# Final-clean refactor notes

## What was removed

- `opencood/comm/` compatibility package.
- `opencood/compression/` compatibility package.
- flat compatibility wrappers for the five active baselines.
- `opencood/models/comm_modules/` compatibility directory.
- `opencood/models/diffuser/` compatibility directory.
- baseline-specific compatibility wrappers previously mixed into `sub_modules/` and `fuse_modules/`.
- historical `archive/`, old `docs/`, `logreplay/`, `run_logs/`, and broken backup files.

## What was reorganized

- five active baselines -> `opencood/models/baselines/`.
- shared model blocks -> `opencood/models/common/`.
- other retained OpenCOOD models -> `opencood/models/upstream/`.
- shared communication -> `opencood/communication/`.
- ARCE/C2MAB -> `opencood/methods/arce/`.
- YAML model loading -> `opencood/models/registry.py`.

## Numerical-behavior policy

This refactor deliberately avoids rewriting the validated ARCE execution logic
or the baseline-specific payload semantics. Changes are directory/import/model-
resolution changes plus removal of compatibility wrappers. Existing YAML
`core_method` values continue to resolve to the same model classes.

## Verification performed in the packaging environment

Passed:

- full Python `compileall` for `opencood`, `scripts`, and `audit_tools`.
- `scripts/verify_refactor.py` structural smoke test.
- 25-arm Markov+C2MAB action-space unit test.
- FP32/FP16/INT8/INT4 compression audit smoke test.
- FEC recovery smoke test.
- RaptorQ planner tests (planner only, no external RaptorQ runtime backend needed).
- budget-retention mapping smoke test.

Environment-limited checks:

- CoSDH full import requires `pyquaternion` in the active environment.
- CoopDiff full import requires `spconv` in the active environment.
- exact RaptorQ transport execution requires `raptorq==1.6.3`.

The refactor source already had some budget-accounting audit unit tests whose
assertions fail in the pre-refactor package as well; those failures were
reproduced before and after the structural refactor and are therefore not caused
by moving modules.
