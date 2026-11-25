#!/usr/bin/env bash
# Wait until the GPU has room for the whole model, (re)start Ollama so the model loads 100% on GPU,
# then run the given command. Meant to run in a detached tmux session:
#
#   tmux new -d -s sgd_gpu "scripts/run_when_gpu_free.sh .venv/bin/python -m sgd.s06_run_agents --tag pilot --pilot"
#
# Env: MIN_FREE_MIB (default 20000), POLL_S (default 60), OLLAMA_BIN (default: ollama on PATH),
#      SGD_MODEL (default devstral-small-2), NUM_CTX (default 16384).
set -uo pipefail
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
POLL_S=${POLL_S:-60}
OLLAMA_BIN=${OLLAMA_BIN:-$(command -v ollama)}
MODEL=${SGD_MODEL:-devstral-small-2}
NUM_CTX=${NUM_CTX:-16384}
log() { echo "[$(date '+%F %T')] $*"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

stop_ollama() {
  pkill -u "$(id -u)" -f "ollama serve" 2>/dev/null
  pkill -u "$(id -u)" -f "ollama/lib/ollama/llama-server" 2>/dev/null
  sleep 5
}

start_ollama() {
  OLLAMA_CONTEXT_LENGTH=$NUM_CTX nohup "$OLLAMA_BIN" serve > "${OLLAMA_LOG:-/dev/null}" 2>&1 &
  for _ in $(seq 1 60); do curl -s localhost:11434/api/tags > /dev/null && return 0; sleep 1; done
  return 1
}

on_gpu() {  # load the model and check that it is fully on the GPU
  curl -s localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_ctx\":$NUM_CTX,\"num_predict\":1}}" > /dev/null
  "$OLLAMA_BIN" ps | tee /dev/stderr | grep -q "100% GPU"
}

while true; do
  stop_ollama  # our own server must not hold GPU memory while we measure
  ok=0
  while [ "$ok" -lt 2 ]; do
    f=$(free_mib)
    if [ "${f:-0}" -ge "$MIN_FREE_MIB" ]; then ok=$((ok + 1)); else ok=0; fi
    [ "$ok" -lt 2 ] && { log "GPU free ${f} MiB < ${MIN_FREE_MIB} (or not yet stable); waiting ${POLL_S}s"; sleep "$POLL_S"; }
  done
  log "GPU free ${f} MiB; starting Ollama"
  start_ollama || { log "Ollama did not start"; sleep "$POLL_S"; continue; }
  if on_gpu; then
    log "model is 100% on GPU; running: $*"
    "$@"
    rc=$?
    log "workload exited with $rc"
    exit $rc
  fi
  log "model not fully on GPU (someone took memory meanwhile); waiting again"
  sleep "$POLL_S"
done
