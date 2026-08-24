#!/usr/bin/env bash
# Start the multiplex OpenAI-compatible server + Open WebUI connected to it.
# Usage: ./serve-webui.sh          start both
#        ./serve-webui.sh stop     stop both
set -euo pipefail

DIR="$HOME/models/multiplex"
SERVE_PORT=8000
WEBUI_PORT=3000
MODEL="$HOME/.mtplx/models/Agents-A1-MTPLX"
OPEN_WEBUI="/opt/homebrew/Caskroom/miniconda/base/envs/open-webui/bin/open-webui"

if [ "${1:-}" = "stop" ]; then
  pkill -f "multiplex.server" 2>/dev/null || true
  pkill -f "open-webui serve" 2>/dev/null || true
  echo "Stopped multiplex server + Open WebUI"
  exit 0
fi

# 1. multiplex server (background), using the project-local custom MLX build.
cd "$DIR"
source "$DIR/scripts/common.sh"
PYTHON="$(multiplex_ensure_mlx "$DIR")"
echo "Starting multiplex server -> http://127.0.0.1:$SERVE_PORT/v1"
"$PYTHON" -m multiplex.server --model "$MODEL" --port "$SERVE_PORT" --debug &

# wait until /v1/models is up
for _ in $(seq 1 120); do
  curl -s -o /dev/null "http://127.0.0.1:$SERVE_PORT/v1/models" 2>/dev/null && { echo "server ready."; break; }
  sleep 2
done

# 2. Open WebUI (background), connected to the server. Disable WebUI's own
# background LLM tasks (title/tags/follow-up/autocomplete) so it doesn't fire
# extra requests behind each turn.
WEBUI_LOG="$DIR/open-webui.log"
echo "Starting Open WebUI -> http://127.0.0.1:$WEBUI_PORT  (log: $WEBUI_LOG)"
OPENAI_API_BASE_URL="http://127.0.0.1:$SERVE_PORT/v1" \
OPENAI_API_KEY="dummy" \
WEBUI_AUTH=False \
ENABLE_PERSISTENT_CONFIG=False \
ENABLE_TITLE_GENERATION=False \
ENABLE_TAGS_GENERATION=False \
ENABLE_FOLLOW_UP_GENERATION=False \
ENABLE_AUTOCOMPLETE_GENERATION=False \
  "$OPEN_WEBUI" serve --host 127.0.0.1 --port "$WEBUI_PORT" >"$WEBUI_LOG" 2>&1 &

echo ""
echo "  Web chat : http://127.0.0.1:$WEBUI_PORT"
echo "  API      : http://127.0.0.1:$SERVE_PORT/v1"
echo "  Stop     : $0 stop"
echo ""
echo "---- multiplex server output below ----"
wait
