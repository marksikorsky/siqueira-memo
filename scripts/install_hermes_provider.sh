#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_BIN="${HERMES_BIN:-$(command -v hermes || true)}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
CONFIG_PATH="${HERMES_CONFIG_PATH:-$HERMES_HOME/config.yaml}"
HERMES_ENV_PATH="${HERMES_ENV_PATH:-$HERMES_HOME/.env}"
PROJECT_ENV_PATH="${SIQUEIRA_PROJECT_ENV_PATH:-$ROOT_DIR/.env}"
PLUGIN_SRC="$ROOT_DIR/plugins/memory/siqueira-memo"
PLUGIN_DST="$HERMES_HOME/plugins/siqueira-memo"

log() { printf '[siqueira-hermes-install] %s\n' "$*"; }
fatal() { printf '[siqueira-hermes-install] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -n "$HERMES_BIN" ]] || fatal "hermes binary not found in PATH"
[[ -f "$CONFIG_PATH" ]] || fatal "Hermes config not found: $CONFIG_PATH"
[[ -d "$PLUGIN_SRC" ]] || fatal "plugin source not found: $PLUGIN_SRC"

# Resolve the Python interpreter used by the Hermes console script.
HERMES_PYTHON="${HERMES_PYTHON:-}"
if [[ -z "$HERMES_PYTHON" ]]; then
  shebang="$(head -n 1 "$HERMES_BIN" || true)"
  if [[ "$shebang" == '#!'*python* ]]; then
    HERMES_PYTHON="${shebang#\#!}"
  else
    HERMES_PYTHON="$(command -v python3 || command -v python)"
  fi
fi
[[ -x "$HERMES_PYTHON" ]] || fatal "Hermes Python is not executable: $HERMES_PYTHON"

if [[ ! -f "$PROJECT_ENV_PATH" ]]; then
  log "project .env missing; running bootstrap first"
  "$ROOT_DIR/scripts/bootstrap.sh"
fi
[[ -f "$PROJECT_ENV_PATH" ]] || fatal "project .env still missing after bootstrap: $PROJECT_ENV_PATH"

log "installing siqueira-memo package into Hermes Python: $HERMES_PYTHON"
if "$HERMES_PYTHON" -m pip --version >/dev/null 2>&1; then
  "$HERMES_PYTHON" -m pip install -e "$ROOT_DIR[postgres,queue,otel]" >/tmp/siqueira-hermes-pip.log 2>&1 || {
    tail -80 /tmp/siqueira-hermes-pip.log >&2 || true
    fatal "pip install failed"
  }
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "$HERMES_PYTHON" -e "$ROOT_DIR[postgres,queue,otel]" >/tmp/siqueira-hermes-pip.log 2>&1 || {
    tail -80 /tmp/siqueira-hermes-pip.log >&2 || true
    fatal "uv pip install failed"
  }
else
  fatal "Hermes Python has no pip, and uv is not installed. Install uv or bootstrap pip first."
fi

log "linking user memory provider plugin"
mkdir -p "$HERMES_HOME/plugins"
if [[ -L "$PLUGIN_DST" || -e "$PLUGIN_DST" ]]; then
  current="$(readlink "$PLUGIN_DST" 2>/dev/null || true)"
  if [[ "$current" != "$PLUGIN_SRC" ]]; then
    rm -rf "$PLUGIN_DST"
    ln -s "$PLUGIN_SRC" "$PLUGIN_DST"
  fi
else
  ln -s "$PLUGIN_SRC" "$PLUGIN_DST"
fi

log "writing Siqueira env vars into $HERMES_ENV_PATH"
"$HERMES_PYTHON" - "$PROJECT_ENV_PATH" "$HERMES_ENV_PATH" "$PLUGIN_SRC/system_prompt.md" "$HERMES_HOME" <<'PY'
from __future__ import annotations
import sys
from pathlib import Path

project_env = Path(sys.argv[1])
hermes_env = Path(sys.argv[2])
prompt_path = Path(sys.argv[3])
hermes_home = Path(sys.argv[4]).expanduser().resolve()

def parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        data[key.strip()] = val.strip().strip('"').strip("'")
    return data

project = parse_env(project_env)
existing = parse_env(hermes_env)

pg_user = project.get('POSTGRES_USER', 'siqueira')
pg_db = project.get('POSTGRES_DB', 'siqueira_memo')
pg_password = project.get('POSTGRES_PASSWORD', '')
pg_port = project.get('SIQUEIRA_POSTGRES_PORT', '5432')
redis_port = project.get('SIQUEIRA_REDIS_PORT', '6379')

database_url = project.get('SIQUEIRA_DATABASE_URL')
if not database_url:
    database_url = f'postgresql+asyncpg://{pg_user}:{pg_password}@127.0.0.1:{pg_port}/{pg_db}'

updates = {
    'SIQUEIRA_ENV': project.get('SIQUEIRA_ENV', 'development'),
    'SIQUEIRA_HOST': project.get('SIQUEIRA_HOST', '127.0.0.1'),
    'SIQUEIRA_PORT': project.get('SIQUEIRA_PUBLIC_PORT', project.get('SIQUEIRA_PORT', '8787')),
    'SIQUEIRA_API_TOKEN': project.get('SIQUEIRA_API_TOKEN', existing.get('SIQUEIRA_API_TOKEN', 'local-dev-token')),
    'SIQUEIRA_DATABASE_URL': database_url,
    'SIQUEIRA_REDIS_URL': project.get('SIQUEIRA_REDIS_URL', f'redis://127.0.0.1:{redis_port}/0'),
    'SIQUEIRA_QUEUE_BACKEND': project.get('SIQUEIRA_QUEUE_BACKEND', 'redis'),
    'SIQUEIRA_EMBEDDING_PROVIDER': project.get('SIQUEIRA_EMBEDDING_PROVIDER', 'mock'),
    'SIQUEIRA_EMBEDDING_MODEL': project.get('SIQUEIRA_EMBEDDING_MODEL', 'text-embedding-3-large'),
    'SIQUEIRA_EMBEDDING_DIMS': project.get('SIQUEIRA_EMBEDDING_DIMS', '3072'),
    'SIQUEIRA_RERANKER_PROVIDER': project.get('SIQUEIRA_RERANKER_PROVIDER', 'mock'),
    'SIQUEIRA_LOG_LEVEL': project.get('SIQUEIRA_LOG_LEVEL', 'INFO'),
    'SIQUEIRA_OTEL_ENABLED': project.get('SIQUEIRA_OTEL_ENABLED', 'false'),
    'SIQUEIRA_HERMES_HOME': str(hermes_home),
    'SIQUEIRA_HERMES_PLUGIN_PROMPT_PATH': str(prompt_path),
}

# Preserve comments/order as much as possible; replace only managed keys.
managed = set(updates)
lines = hermes_env.read_text().splitlines() if hermes_env.exists() else []
out: list[str] = []
seen: set[str] = set()
for raw in lines:
    stripped = raw.strip()
    if not stripped or stripped.startswith('#') or '=' not in stripped:
        out.append(raw)
        continue
    key = stripped.split('=', 1)[0].strip()
    if key in managed:
        out.append(f'{key}={updates[key]}')
        seen.add(key)
    else:
        out.append(raw)

missing = [k for k in updates if k not in seen]
if missing:
    if out and out[-1].strip():
        out.append('')
    out.append('# Siqueira Memo memory provider')
    for key in missing:
        out.append(f'{key}={updates[key]}')

hermes_env.parent.mkdir(parents=True, exist_ok=True)
hermes_env.write_text('\n'.join(out).rstrip() + '\n')
PY

log "setting memory.provider=siqueira-memo in Hermes config"
"$HERMES_PYTHON" - "$CONFIG_PATH" <<'PY'
from __future__ import annotations
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text()) or {}
config.setdefault('memory', {})['provider'] = 'siqueira-memo'
path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
PY

log "verifying Docker service readiness"
(
  cd "$ROOT_DIR"
  docker compose up -d postgres redis api worker >/tmp/siqueira-hermes-compose.log 2>&1 || {
    tail -80 /tmp/siqueira-hermes-compose.log >&2 || true
    fatal "docker compose up failed"
  }
)

# Load env for import/readiness checks without printing secrets.
set -a
# shellcheck disable=SC1090
source "$HERMES_ENV_PATH"
set +a

log "verifying provider discovery and import"
HERMES_SOURCE_DIR="$($HERMES_PYTHON - <<'PY'
from pathlib import Path
import hermes_cli
print(Path(hermes_cli.__file__).resolve().parents[1])
PY
)"
(
  cd "$HERMES_SOURCE_DIR"
  "$HERMES_PYTHON" - <<'PY'
from plugins.memory import discover_memory_providers, load_memory_provider
providers = {name: available for name, _desc, available in discover_memory_providers()}
if providers.get('siqueira-memo') is not True:
    raise SystemExit(f'siqueira-memo not available in Hermes providers: {providers!r}')
provider = load_memory_provider('siqueira-memo')
if provider is None or provider.name != 'siqueira-memo' or not provider.is_available():
    raise SystemExit('failed to load active siqueira-memo provider')
print('provider_ok')
PY
)

log "verifying Hermes memory status"
"$HERMES_BIN" memory status | sed -n '1,80p'

log "done. Restart Hermes gateway/CLI sessions for the new provider to take effect."
