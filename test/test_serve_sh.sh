#!/usr/bin/env bash
# Tests for serve.sh's agent wiring. Run: ./test/test_serve_sh.sh
#
# Sourcing serve.sh defines its functions without serving, so each one is tested
# directly against a scratch $CODEX_HOME / $PI_CODING_AGENT_DIR.
set -uo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TEST_DIR/../serve.sh"   # also sets HERE, so read ours back afterwards
REPO="$(cd "$TEST_DIR/.." && pwd)"

pass=0 fail=0
check() {  # description expected actual
  if [ "$2" = "$3" ]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    printf 'FAIL %s\n  expected: %s\n  actual:   %s\n' "$1" "$2" "$3"
  fi
}
contains() {  # description haystack needle
  case "$2" in
    *"$3"*) pass=$((pass + 1)) ;;
    *) fail=$((fail + 1)); printf 'FAIL %s\n  %s\n  does not contain: %s\n' "$1" "$2" "$3" ;;
  esac
}

BASE_CONFIG='model = "gpt-5.6-terra"
model_reasoning_effort = "medium"
service_tier = "fast"

[tui]
model_availability_nux = 3

[model_providers.other]
name = "something else"
'

new_home() {  # -> a $CODEX_HOME holding BASE_CONFIG
  local home; home="$(mktemp -d)"
  printf '%s' "$BASE_CONFIG" > "$home/config.toml"
  echo "$home"
}

# $(cat file) drops the trailing newline, so byte-exactness needs a real compare.
same_as_base() {  # description path
  local want; want="$(mktemp)"; printf '%s' "$BASE_CONFIG" > "$want"
  if cmp -s "$want" "$2"; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    printf 'FAIL %s\n' "$1"; diff "$want" "$2" | head -5
  fi
  rm -f "$want"
}

# --- discovery ---------------------------------------------------------------
h="$(mktemp -d)"
check 'codex_home is empty without a config' '' "$(CODEX_HOME="$h" codex_home)"
h="$(new_home)"
check 'codex_home honours CODEX_HOME' "$h" "$(CODEX_HOME="$h" codex_home)"
check 'codex_home defaults to ~/.codex' "$h/.codex" \
  "$(mkdir -p "$h/.codex" && touch "$h/.codex/config.toml"
     HOME="$h" CODEX_HOME='' codex_home)"

# --- switching ---------------------------------------------------------------
h="$(new_home)"; out="$(CODEX_HOME="$h" codex_switch my-model 127.0.0.1 8123)"
cfg="$(cat "$h/config.toml")"
contains 'switch reports itself' "$out" 'switched to this server'
contains 'switch selects the model' "$cfg" 'model = "my-model"'
contains 'switch selects the provider' "$cfg" 'model_provider = "multiplex"'
contains 'switch points at the port' "$cfg" 'base_url = "http://127.0.0.1:8123/v1"'
contains 'switch uses the responses API' "$cfg" 'wire_api = "responses"'
contains 'switch displaces the old model' "$cfg" '#multiplex-was# model = "gpt-5.6-terra"'
contains 'switch leaves service_tier alone' "$cfg" 'service_tier = "fast"'
contains 'switch keeps unrelated base keys' "$cfg" 'model_reasoning_effort = "medium"'
contains 'switch leaves tables alone' "$cfg" 'model_availability_nux = 3'
contains 'switch does not touch a table key named name' "$cfg" 'name = "something else"'
same_as_base 'switch backs the config up first' "$h/.multiplex-restore"

# The base keys must precede the first table, or TOML reads them as its members.
check 'base keys sit above the first table' 'ok' \
  "$(awk '/^\[/ { table = 1 } /^model = / { if (table) { print "leaked into a table"; exit } } END { print "ok" }' "$h/config.toml")"

# Duplicate top-level keys are a hard TOML error, so the displaced copy matters.
check 'model is defined exactly once' '1' "$(grep -c '^model = ' "$h/config.toml")"

if command -v python3 >/dev/null 2>&1; then
  check 'switched config parses as TOML' 'ok' "$(python3 -c '
import sys, tomllib
try:
    tomllib.load(open(sys.argv[1], "rb")); print("ok")
except Exception as e:
    print(e)' "$h/config.toml")"
fi

# --- restoring ---------------------------------------------------------------
h="$(new_home)"; CODEX_HOME="$h" codex_switch m 127.0.0.1 8000 >/dev/null
out="$(CODEX_HOME="$h" codex_restore)"
contains 'restore reports itself' "$out" 'restored'
same_as_base 'restore returns the original bytes' "$h/config.toml"
check 'restore removes the backup' 'gone' "$([ -f "$h/.multiplex-restore" ] || echo gone)"

check 'restore is a no-op when nothing was switched' '' "$(CODEX_HOME="$(new_home)" codex_restore)"

h="$(new_home)"; CODEX_HOME="$h" codex_switch m 127.0.0.1 8000 >/dev/null
printf 'model = "cc-switch wrote this"\n' > "$h/config.toml"
out="$(CODEX_HOME="$h" codex_restore)"
contains 'restore refuses to revert another tool' "$out" 'changed by something else'
check 'restore leaves the other tool write intact' 'model = "cc-switch wrote this"' "$(cat "$h/config.toml")"

# A run killed with SIGKILL leaves a backup behind; the next start recovers it.
h="$(new_home)"; CODEX_HOME="$h" codex_switch model-a 127.0.0.1 8000 >/dev/null
out="$(CODEX_HOME="$h" codex_switch model-b 127.0.0.1 8123)"
contains 'a leaked switch is reported as recovered' "$out" 'recovered config.toml'
contains 'the second switch still applies' "$(cat "$h/config.toml")" 'model = "model-b"'
CODEX_HOME="$h" codex_restore >/dev/null
same_as_base 'restore after a recovery returns the original' "$h/config.toml"

# --- pi ----------------------------------------------------------------------
# pi_install is a no-op unless `pi` is on PATH, so fake one.
fakebin="$(mktemp -d)"; printf '#!/bin/sh\n' > "$fakebin/pi"; chmod +x "$fakebin/pi"
export PATH="$fakebin:$PATH"

d="$(mktemp -d)"; out="$(PI_CODING_AGENT_DIR="$d" pi_install)"
contains 'pi install reports itself' "$out" 'extension installed'
check 'pi install matches the source' "$(cat "$REPO/agents/pi-extension.ts")" "$(cat "$d/extensions/multiplex.ts")"
check 'pi install is quiet when current' '' "$(PI_CODING_AGENT_DIR="$d" pi_install)"

printf '// Installed by multiplex — an older copy\n' > "$d/extensions/multiplex.ts"
out="$(PI_CODING_AGENT_DIR="$d" pi_install)"
contains 'pi install refreshes an outdated copy' "$out" 'extension installed'
check 'pi install rewrites it fully' "$(cat "$REPO/agents/pi-extension.ts")" "$(cat "$d/extensions/multiplex.ts")"

d="$(mktemp -d)"; mkdir -p "$d/extensions"; printf '// mine\n' > "$d/extensions/multiplex.ts"
out="$(PI_CODING_AGENT_DIR="$d" pi_install)"
contains 'pi install spares a hand-written extension' "$out" 'not ours'
check 'pi install leaves it byte-identical' '// mine' "$(cat "$d/extensions/multiplex.ts")"

d="$(mktemp -d)"
check 'pi install is a no-op without pi' 'none' \
  "$(PATH=/usr/bin:/bin PI_CODING_AGENT_DIR="$d" pi_install >/dev/null; [ -e "$d/extensions" ] || echo none)"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
