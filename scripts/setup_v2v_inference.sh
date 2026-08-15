#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[V2V_inference] project: $PROJECT_ROOT"
echo "[V2V_inference] python : $(command -v python)"
python -V

# Keep the user's validated environment intact. Dependency installation is
# intentionally not forced here; the original project has CUDA/spconv-specific
# packages that should remain matched to the existing conda environment.
python -m pip install -e . --no-deps

# Build the OpenCOOD Cython IoU extension from the project root when no compiled
# extension is present. Running opencood/utils/setup.py from inside that folder
# is incorrect because its cythonize path is project-root relative.
if ! ls opencood/utils/box_overlaps*.so >/dev/null 2>&1; then
  echo "[V2V_inference] building opencood.utils.box_overlaps"
  python opencood/utils/setup.py build_ext --inplace
fi

python scripts/verify_refactor.py

echo
echo "Environment ready. In each shell, use:"
echo "  cd $PROJECT_ROOT"
echo '  export PYTHONPATH=$(pwd):${PYTHONPATH:-}'
