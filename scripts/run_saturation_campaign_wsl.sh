#!/usr/bin/env bash
# One sequential campaign. Uses cached arrays only; never invokes ISIS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
exec python scripts/run_saturation_campaign.py "$@"
