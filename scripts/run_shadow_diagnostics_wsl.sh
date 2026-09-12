#!/usr/bin/env bash
# Saved-product diagnosis and a portable bundle; no ISIS invocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 1 ]]; then
  echo 'Usage: bash scripts/run_shadow_diagnostics_wsl.sh path/to/registration-0.5' >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
RUN=$(date -u +%Y%m%dT%H%M%SZ)
git rev-parse HEAD | tee "logs/shadow-diagnostics-revision-$RUN.txt"
for test in shadow_likelihood landing_maps shadow_diagnostics; do
  python "tests/test_$test.py" 2>&1 | tee "logs/test-$test-$RUN.log"
done
python scripts/diagnose_landing_run.py --run-dir "$1" \
  2>&1 | tee "logs/shadow-diagnostics-$RUN.log"
