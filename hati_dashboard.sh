#!/usr/bin/env bash
# HATI Mission Dashboard launcher (Linux/WSL GPU box):  chmod +x hati_dashboard.sh && ./hati_dashboard.sh
cd "$(dirname "$0")"
echo
echo "  H A T I   -  Hazard Assessment and Terrain Intelligence"
echo "  starting mission dashboard ..."
echo
exec python3 dashboard/hati_dashboard.py "$@"
