#!/usr/bin/env bash
# Postprocess the existing audited sweep. Does not run ISIS or ingestion.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
RUN=$(date -u +%Y%m%dT%H%M%SZ)
git rev-parse HEAD | tee "logs/landing-maps-revision-$RUN.txt"
python -m pip freeze > "logs/landing-maps-python-$RUN.txt"
for test in channels footprint coreg kinematics audit_regressions shadow_likelihood landing_maps; do
  python "tests/test_$test.py" 2>&1 | tee "logs/test-$test-$RUN.log"
done
# These are sensitivity assumptions, NOT measurements of local registration.
# HALF may be increased up to the smallest ingested half_px. Each run visits
# every interior analysis cell in the requested window; it is not the whole site.
for sigma in 0.25 0.5 1.0; do
  python scripts/landing_maps.py \
    --manifest data/sweep/manifest.json \
    --before 2025-03-06T00:00:00Z --half "${HALF:-256}" \
    --registration-sigma-px "$sigma" --noise-sigma 0.03 \
    --config configs/landing_illustrative.json \
    --output "output/athena/landing_maps_v253/$RUN/registration-$sigma" \
    2>&1 | tee "logs/landing-maps-registration-$sigma-$RUN.log"
done
python scripts/compare_landing_scenarios.py \
  "output/athena/landing_maps_v253/$RUN/registration-0.25" \
  "output/athena/landing_maps_v253/$RUN/registration-0.5" \
  "output/athena/landing_maps_v253/$RUN/registration-1.0" \
  --output "output/athena/landing_maps_v253/$RUN/comparison" \
  2>&1 | tee "logs/landing-maps-comparison-$RUN.log"
echo "Done. Three independent map files + fusion observability in each scenario:"
echo "output/athena/landing_maps_v253/$RUN"
