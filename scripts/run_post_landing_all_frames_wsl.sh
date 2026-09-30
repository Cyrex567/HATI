#!/usr/bin/env bash
# Post-landing validation on every usable frame that shows the landed Athena: the
# 24 frames declared in configs/post_landing_check_all_frames.json, nine of them the
# nine-frame check's. Same pipeline as run_post_landing_wsl.sh; the frame list, the
# window and the relaxed ingest gates come from the declared plan, and everything goes
# to its own folders so the nine-frame run is never overwritten.
#
#   bash scripts/run_post_landing_all_frames_wsl.sh
#
# Optional environment: as run_post_landing_wsl.sh (PRE, OUT, SWEEP, EXPORT_DIR, ...).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLAN=configs/post_landing_check_all_frames.json
plan() { python -c "import json, sys; p = json.load(open(sys.argv[1])); print($1)" "$PLAN"; }
# Plain assignments, so a failed read of the plan stops the script under set -e.
FRAMES=$(plan "','.join(f for group in p['frames'].values() for f in group)")
AFTER=$(plan "p['window']['after']")
BEFORE=$(plan "p['window']['before']")
CATALOG=$(plan "p['ingest']['catalog']")
EMISSION=$(plan "p['ingest']['max_emission_deg']")
MARGIN=$(plan "p['ingest']['min_margin_m']")
INGEST_ARGS="--frames $FRAMES --catalog $CATALOG --max-emission $EMISSION --min-margin-m $MARGIN"
CHECK_CONFIG="$PLAN"
export AFTER BEFORE INGEST_ARGS CHECK_CONFIG
export SWEEP="${SWEEP:-data/sweep_post_landing_all}"
export OUT="${OUT:-output/athena/post_landing/$(date -u +%Y%m%dT%H%M%SZ)-all-frames}"
export EXPORT_DIR="${EXPORT_DIR:-/mnt/c/Users/Public/Downloads/HATI/all-frames}"
echo "Declared plan $PLAN: $(tr ',' '\n' <<< "$FRAMES" | wc -l) frames, $AFTER to before $BEFORE"
exec bash scripts/run_post_landing_wsl.sh
