#!/usr/bin/env bash
# Execute on the Linux host holding the existing NOBILE03 reference data.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This runbook requires Linux or WSL.' >&2
  exit 1
fi
for tool in lronac2isis spiceinit campt getkey catlab lronaccal lronacecho cam2map; do
  command -v "$tool" >/dev/null || { echo "Missing ISIS program: $tool" >&2; exit 1; }
done
mkdir -p logs
RUN=$(date -u +%Y%m%dT%H%M%SZ)
git rev-parse HEAD | tee "logs/v25-revision-$RUN.txt"
python -m pip freeze > "logs/v25-python-$RUN.txt"
for test in channels footprint coreg kinematics audit_regressions shadow_likelihood; do
  python "tests/test_$test.py" 2>&1 | tee "logs/test-$test-$RUN.log"
done
python scripts/solar_sweep_query.py \
  --lat -84.7906 --lon 29.1957 --pt EDRNAC4 \
  --before 2025-03-06T00:00:00Z --min-margin-m 600 \
  2>&1 | tee "logs/query-$RUN.log"
python scripts/ingest_sweep.py \
  --catalog output/athena/solar_sweep_-84.791_+29.196_EDRNAC4.csv \
  --before 2025-03-06T00:00:00Z \
  --n 24 --min-elev 1.0 --max-elev 9.0 --target-elev 4.0 \
  --max-emission 40 --half 1200 --alternates 3 --execute \
  2>&1 | tee "logs/ingest-$RUN.log"
# Explicit sensitivity scenarios, NOT measured local registration errors.
for sigma in 0.25 0.5 1.0; do
  python scripts/shadow_roots_real.py \
    --before 2025-03-06T00:00:00Z --half 256 \
    --registration-sigma-px "$sigma" --noise-sigma 0.03 \
    --output "output/athena/shadow_roots_v25/$RUN/registration-$sigma" \
    2>&1 | tee "logs/roots-registration-$sigma-$RUN.log"
done
echo "Finished ingestion and v2.5 pilot rankings. Run ID: $RUN"
echo 'Registration/noise/flat-plane values are assumptions; inspect run.json before interpretation.'
