#!/bin/sh
# Refresh the code in the state volume on every start; keep the model state that al_cycle.py manages
# (models/, powerline_ai/thresholds.json, .al_work/).
set -e
mkdir -p /data/modal
cd /opt/modal-src
find . -type f ! -path './models/*' ! -path './powerline_ai/thresholds.json' | while read -r f; do
  mkdir -p "/data/modal/$(dirname "$f")"
  cp "$f" "/data/modal/$f"
done
mkdir -p /data/modal/models
[ -f /data/modal/powerline_ai/thresholds.json ] || cp powerline_ai/thresholds.json /data/modal/powerline_ai/thresholds.json
exec python /app/worker.py "$@"
