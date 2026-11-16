#!/usr/bin/env bash
set -euo pipefail
bipga_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$bipga_root/.venv/bin/bipga" ]; then
  echo 'Run bash install.sh --cpu or bash install.sh --gpu first.' >&2
  exit 1
fi
exec "$bipga_root/.venv/bin/bipga" "$@"
