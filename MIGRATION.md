# Migration notes — final clean tree

1. Use the existing `opencood` conda environment.
2. Run from the `V2V_inference` project root and prepend it to `PYTHONPATH`.
3. Existing experiment YAML files may keep their original `model.core_method`
   values; `opencood.models.registry` resolves the five active baselines to
   their structured package locations.
4. Old source imports such as `opencood.comm.*`, `opencood.compression.*`,
   `opencood.models.comm_modules.*`, and moved baseline wrappers are no longer
   supported. All repository-owned code has been migrated to canonical paths.
5. Existing checkpoints are state-dict based and can be loaded from their old
   log directories; moving Python module files does not rename model parameter
   keys.

Verify after installation:

```bash
python scripts/verify_refactor.py
```
