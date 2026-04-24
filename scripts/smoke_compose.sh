#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

port="${SIQUEIRA_PUBLIC_PORT:-$(grep -E '^SIQUEIRA_PUBLIC_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2- || echo 8787)}"

docker compose config >/dev/null
curl -fsS "http://127.0.0.1:${port}/healthz" | python3 -m json.tool
curl -fsS "http://127.0.0.1:${port}/readyz" | python3 -m json.tool

echo "compose smoke ok"
