#!/usr/bin/env bash
# Serve a model, and point the coding agents on this machine at it.
#
# Usage: ./serve.sh [any multiplex.server flag...]   # picks a model if none given
#
# Wiring up pi and Codex belongs here, not in the server: what else is installed
# on the machine is none of the server's business, and `trap EXIT` already does
# properly what a Python signal handler can only approximate. The served model's
# id comes back from GET /v1/models, so this script knows nothing about models.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MULTIPLEX_PYTHON:-python3}"

# --- pi ----------------------------------------------------------------------
# pi discovers models through its extension API only, so it gets an extension
# that calls /v1/models at pi's own startup. Always current, nothing to undo.

pi_install() {
  command -v pi >/dev/null 2>&1 || return 0
  local dir="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/extensions"
  local target="$dir/multiplex.ts"
  if [ -e "$target" ] && ! grep -q 'by multiplex' "$target"; then
    echo "[pi: $target is not ours, left alone]"
    return 0
  fi
  cmp -s "$HERE/agents/pi-extension.ts" "$target" 2>/dev/null && return 0
  mkdir -p "$dir"
  cp "$HERE/agents/pi-extension.ts" "$target"
  echo '[pi: extension installed]'
}

# --- Codex -------------------------------------------------------------------
# Codex has no dynamic discovery and its model picker does not reliably offer a
# custom provider, so its base config is the only dependable route. That is a
# global switch — an already-open session changes model too — hence the restore.

CODEX_BEGIN='# >>> multiplex begin — added while serving; removed on exit'
CODEX_END='# <<< multiplex end'
# Kept rather than deleted so restoring never has to reconstruct a value.
CODEX_DISPLACED='#multiplex-was# '
CODEX_BACKUP='.multiplex-restore'

codex_home() {
  local home="${CODEX_HOME:-$HOME/.codex}"
  # The directory outlives an uninstall; the config file is the real signal.
  [ -f "$home/config.toml" ] && echo "$home"
}

codex_switch() {  # model_id host port
  local home; home="$(codex_home)" || return 0
  [ -n "$home" ] || return 0
  local cfg="$home/config.toml" backup="$home/$CODEX_BACKUP"

  # A previous run was killed before it could restore. Its backup is the real
  # pre-switch file, so recover that instead of snapshotting our own block.
  if [ -f "$backup" ]; then
    cp "$backup" "$cfg"
    echo '[codex: recovered config.toml from an earlier run]'
  fi
  # Back up before switching: a crash in between has to stay recoverable.
  cp "$cfg" "$backup"
  {
    printf '%s\n' "$CODEX_BEGIN"
    printf 'model = "%s"\n' "$1"
    printf 'model_provider = "multiplex"\n'
    printf '%s\n\n' "$CODEX_END"
    # These keys must not appear twice — TOML rejects duplicates — and must sit
    # above the first [table], or they parse as members of it. Only top-level
    # copies count: the same name inside a table belongs to that table.
    awk -v mark="$CODEX_DISPLACED" '
      /^[[:space:]]*\[/ { t = 1 }
      !t && /^[[:space:]]*(model|model_provider)[[:space:]]*=/ {
        print mark $0; next
      }
      { print }' "$backup"
    printf '\n%s\n' "$CODEX_BEGIN"
    printf '[model_providers.multiplex]\n'
    printf 'name = "multiplex (local)"\n'
    printf 'base_url = "http://%s:%s/v1"\n' "$2" "$3"
    # Codex speaks the Responses API; chat completions is deprecated there.
    printf 'wire_api = "responses"\n'
    printf '%s\n' "$CODEX_END"
  } > "$cfg"
  echo '[codex: switched to this server; restored on exit]'
}

codex_restore() {
  local home; home="$(codex_home)" || return 0
  [ -n "$home" ] || return 0
  local cfg="$home/config.toml" backup="$home/$CODEX_BACKUP"
  [ -f "$backup" ] || return 0
  # cc-switch and friends replace config.toml wholesale. If our block is gone,
  # something else owns the file now and restoring would revert its write.
  if ! grep -qF "$CODEX_BEGIN" "$cfg" 2>/dev/null; then
    echo '[codex: config.toml was changed by something else; left alone]'
    return 0
  fi
  mv "$backup" "$cfg"
  echo '[codex: config.toml restored]'
}

# --- serve -------------------------------------------------------------------

main() {
  local host=127.0.0.1 port=8000 args=("$@") i
  for ((i = 0; i < $#; i++)); do
    case "${args[i]}" in
      --host) host="${args[i + 1]}" ;;
      --port) port="${args[i + 1]}" ;;
    esac
  done

  # The server runs in the FOREGROUND so it keeps the terminal: with no --model
  # it prompts for one, and bash gives a background job /dev/null for stdin,
  # which the prompt reads as "not interactive". So the waiting is what goes to
  # the background instead.
  attach "$host" "$port" &
  local attacher=$!
  trap 'kill "$attacher" 2>/dev/null; codex_restore' EXIT

  "$PYTHON" -m multiplex.server "$@"
}

attach() {  # host port
  # Codex needs the served model's id, which the server reports on /v1/models —
  # waiting for that doubles as waiting for it to come up (and for the user to
  # pick a model, and for the weights to load, so this cannot time out). The
  # EXIT trap kills this if the server never gets there.
  local card=''
  until card="$(curl -sf --max-time 2 "http://$1:$2/v1/models")" && [ -n "$card" ]; do
    sleep 1
  done
  local id
  id="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' <<<"$card" | head -1)"
  pi_install
  codex_switch "$id" "$1" "$2"
}

# Sourcing this file (the tests do) must define the functions without serving.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
