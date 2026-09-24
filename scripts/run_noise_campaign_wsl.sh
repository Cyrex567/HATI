#!/usr/bin/env bash
# Relief and noise campaign (T1 -> T12 -> T13 -> T14 -> T16) with HATI Watch open in
# the browser. Uses cached arrays only; never invokes ISIS. Software checks run
# first, as in every campaign. Prints the run folder and the Windows copy of the ZIP.
#
#   bash scripts/run_noise_campaign_wsl.sh [hati_diagnostic_bundle.zip]
#
# Optional environment: CONFIG, STAGES, OUT, EXPORT_DIR, PORT, NO_BROWSER=1.
# The first noise-only campaign used CONFIG=configs/saturation_campaign_noise_workstation.json STAGES=T1,T12,T16.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUNDLE="${1:-output/athena/landing_maps_v253/20260912T072640Z/registration-0.5/diagnostics_v254/hati_diagnostic_bundle.zip}"
CONFIG="${CONFIG:-configs/saturation_campaign_relief_workstation.json}"
STAGES="${STAGES:-T1,T12,T13,T14,T16}"
OUT="${OUT:-output/athena/saturation_campaign/relief-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPORT_DIR="${EXPORT_DIR:-/mnt/c/Users/Public/Downloads/HATI}"
PORT="${PORT:-8765}"

for f in "$BUNDLE" "$CONFIG"; do
  [[ -f "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }
done
if [[ -e "$OUT" ]]; then
  echo "Output already exists: $OUT (use a new OUT, or resume with run_saturation_campaign_wsl.sh --resume)" >&2
  exit 1
fi
# The runner requires an empty output folder, so the viewer log sits beside it.
mkdir -p "$(dirname "$OUT")"
WATCH_LOG="$OUT.watch.log"

# Read-only viewer. It may start before the run folder exists, and stopping it
# never touches the calculation.
python dashboard/hati_watch.py --run-dir "$OUT" --port "$PORT" >"$WATCH_LOG" 2>&1 &
WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT
sleep 2
URL="http://localhost:$PORT"
if kill -0 "$WATCH_PID" 2>/dev/null; then
  echo "HATI Watch: $URL"
  if [[ "${NO_BROWSER:-0}" != 1 ]]; then
    if command -v wslview >/dev/null 2>&1; then wslview "$URL" >/dev/null 2>&1 || true
    elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$URL" >/dev/null 2>&1 || true
    fi
  fi
else
  echo "HATI Watch did not start (port $PORT busy? rerun with PORT=8766). Log: $WATCH_LOG" >&2
  cat "$WATCH_LOG" >&2 || true
  echo "The campaign runs without it." >&2
fi

echo "Campaign stages $STAGES -> $OUT"
set +e
bash scripts/run_saturation_campaign_wsl.sh --bundle "$BUNDLE" --config "$CONFIG" --stages "$STAGES" \
  --output "$OUT" --export-dir "$EXPORT_DIR"
CODE=$?
set -e

case $CODE in
  0) echo "Execution finished (exit 0). Read VERDICT.md for the scientific result." ;;
  2) echo "Finished with a missing prerequisite or blocked stage (exit 2). See VERDICT.md." ;;
  *) echo "A stage failed or was interrupted (exit $CODE). See VERDICT.md and logs/; rerun with --resume to continue." ;;
esac
echo "Run folder: $OUT"
echo "Results ZIP: ${OUT}_results.zip (Windows copy in $EXPORT_DIR)"
if kill -0 "$WATCH_PID" 2>/dev/null && [[ -t 0 ]]; then
  read -r -p "HATI Watch still shows the finished run at $URL. Press Enter to close it. " || true
fi
exit "$CODE"
