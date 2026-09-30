#!/usr/bin/env bash
# Post-landing validation for Athena, on frames acquired after the landing. They are
# held out from the counterfactual: ingested into their own folder, read only through
# the explicit post-landing window, never mixed with data/sweep.
#
#   1. ingest the post-landing frames (downloads and ISIS; needs `conda activate isis`)
#   2. build the maps exactly as for the pre-landing stack
#   3. export the diagnostic bundle
#   4. run the relief stages of the campaign, with HATI Watch
#   5. compare with the pre-landing campaign (configs/post_landing_check.json, or CHECK_CONFIG)
#
#   bash scripts/run_post_landing_wsl.sh
#
# Optional environment: PRE (pre-landing campaign folder), AFTER, BEFORE, SWEEP, OUT,
# STAGES, CONFIG, EXPORT_DIR, PORT, NO_BROWSER=1, SKIP_INGEST=1 (reuse ingested frames),
# INGEST_ARGS (extra ingest_sweep.py options, e.g. a declared frame list), CHECK_CONFIG
# (the declared plan the comparison applies). Unset, the run is the nine-frame check.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

AFTER="${AFTER:-2025-03-07T00:00:00Z}"
BEFORE="${BEFORE:-2025-03-22T00:00:00Z}"
SWEEP="${SWEEP:-data/sweep_post_landing}"
PRE="${PRE:-output/athena/saturation_campaign/full-03}"
RUN="${OUT:-output/athena/post_landing/$(date -u +%Y%m%dT%H%M%SZ)}"
EXPORT_DIR="${EXPORT_DIR:-/mnt/c/Users/Public/Downloads/HATI}"
CHECK_CONFIG="${CHECK_CONFIG:-configs/post_landing_check.json}"
read -r -a INGEST_EXTRA <<< "${INGEST_ARGS:-}"
export STAGES="${STAGES:-T1,T12,T13,T14}"
export CONFIG="${CONFIG:-configs/saturation_campaign_relief_workstation.json}"
export EXPORT_DIR PORT="${PORT:-8765}"

[[ -d "$PRE/stages/T14" ]] || { echo "Pre-landing campaign not found: $PRE (set PRE=...)" >&2; exit 1; }
[[ -f "$CHECK_CONFIG" ]] || { echo "Declared plan not found: $CHECK_CONFIG" >&2; exit 1; }
[[ -e "$RUN" ]] && { echo "Output already exists: $RUN (use a new OUT)" >&2; exit 1; }
mkdir -p logs "$RUN"
exec > >(tee -a "$RUN/post_landing.log") 2>&1
git rev-parse HEAD | tee "$RUN/revision.txt"

echo "== 1/5 Ingest frames acquired from $AFTER to before $BEFORE into $SWEEP"
if [[ "${SKIP_INGEST:-0}" != 1 ]]; then
  python scripts/ingest_sweep.py --after "$AFTER" --before "$BEFORE" --sweep-dir "$SWEEP" --n 72 \
    "${INGEST_EXTRA[@]}" --execute
fi
cp "$SWEEP/manifest.json" "$RUN/manifest.json"

echo "== 2/5 Maps on the post-landing stack"
python scripts/landing_maps.py --manifest "$SWEEP/manifest.json" --after "$AFTER" --before "$BEFORE" \
  --half 256 --registration-sigma-px 0.5 --noise-sigma 0.03 --config configs/landing_illustrative.json \
  --output "$RUN/maps"

echo "== 3/5 Diagnostic bundle"
python scripts/diagnose_landing_run.py --run-dir "$RUN/maps" --manifest "$SWEEP/manifest.json"

echo "== 4/5 Campaign stages $STAGES"
set +e
OUT="$RUN/campaign" bash scripts/run_noise_campaign_wsl.sh "$RUN/maps/diagnostics_v254/hati_diagnostic_bundle.zip" </dev/null
CODE=$?
set -e
if [[ ! -f "$RUN/campaign/stages/T14/result.json" ]]; then
  echo "The campaign did not finish T14 (exit $CODE); resume it with run_saturation_campaign_wsl.sh --resume." >&2
  exit "$CODE"
fi

echo "== 5/5 Compare with the pre-landing campaign $PRE"
python scripts/post_landing_check.py --pre "$PRE" --post "$RUN/campaign" --output "$RUN/check" --config "$CHECK_CONFIG"
NAME="post_landing_check_$(basename "$RUN")"
(cd "$RUN" && python -c "import shutil; shutil.make_archive('$NAME', 'zip', '.', 'check')")
mkdir -p "$EXPORT_DIR" && cp "$RUN/$NAME.zip" "$EXPORT_DIR/" || true
echo "Send these two files:"
echo "  ${RUN}/campaign_results.zip  (Windows copy in $EXPORT_DIR)"
echo "  ${RUN}/${NAME}.zip  (Windows copy in $EXPORT_DIR)"
