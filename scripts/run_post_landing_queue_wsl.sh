#!/usr/bin/env bash
# The post-landing queue for the stationary machine: the nine-frame check
# (configs/post_landing_check.json), then the run on all 24 frames that show the
# lander (configs/post_landing_check_all_frames.json), one after the other on the
# same checkout, so both use the same code. A failure in the first does not stop the
# second, and each run keeps its own sweep, output and export folders.
#
#   conda activate isis
#   bash scripts/run_post_landing_queue_wsl.sh
#
# Leave the terminal open until it prints the summary. Optional environment:
# START_AT (for example 22:00) waits until that time today; PRE and PORT pass through.
# OUT, SWEEP and EXPORT_DIR are left to each run, so the second cannot overwrite the first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for tool in spiceinit campt cam2map; do
  command -v "$tool" >/dev/null || { echo "ISIS is not on PATH; run 'conda activate isis' first." >&2; exit 1; }
done
PRE="${PRE:-output/athena/saturation_campaign/full-03}"
[[ -d "$PRE/stages/T14" ]] || { echo "Pre-landing campaign not found: $PRE (set PRE=...)" >&2; exit 1; }
unset OUT SWEEP EXPORT_DIR
export PRE

mkdir -p logs
LOG="logs/post_landing_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "Queue log: $LOG"
echo "Revision: $(git rev-parse HEAD)"
echo "Free space: $(df -h . | awk 'NR == 2 {print $4}') (the 24-frame run downloads about 8.4 GB)"

if [[ -n "${START_AT:-}" ]]; then
  target=$(date -d "$START_AT" +%s) || exit 1
  now=$(date +%s)
  if (( target > now )); then
    echo "Waiting until $(date -d "@$target" '+%H:%M')"
    sleep $(( target - now ))
  fi
fi

summary=()
for job in run_post_landing_wsl.sh run_post_landing_all_frames_wsl.sh; do
  echo "== $(date -u +%FT%TZ) start $job"
  if [[ $job == run_post_landing_all_frames_wsl.sh ]]; then
    NO_BROWSER="${NO_BROWSER:-1}" bash "scripts/$job" </dev/null
  else
    bash "scripts/$job" </dev/null
  fi
  code=$?
  summary+=("$job: exit $code")
  echo "== $(date -u +%FT%TZ) $job finished (exit $code)"
done

echo
echo "Queue finished."
printf '  %s\n' "${summary[@]}"
echo "Nine frames: ZIPs in /mnt/c/Users/Public/Downloads/HATI"
echo "24 frames:   ZIPs in /mnt/c/Users/Public/Downloads/HATI/all-frames"
