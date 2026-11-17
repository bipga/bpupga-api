#!/usr/bin/env bash
set -euo pipefail
bipga_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bipga_python="${BIPGA_PYTHON:-python3}"
if [ -z "${BIPGA_PYTHON:-}" ] && command -v python3.10 >/dev/null 2>&1; then bipga_python=python3.10; fi
bipga_mode=cpu
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu) bipga_mode=cpu; shift ;;
    --gpu) bipga_mode=gpu; shift ;;
    --python) bipga_python="$2"; shift 2 ;;
    *) echo "Usage: bash install.sh [--cpu|--gpu] [--python PYTHON]" >&2; exit 2 ;;
  esac
done
"$bipga_python" -c 'import sys; assert sys.version_info[:2] == (3,10), "The complete runtime requires Python 3.10; select it with --python PATH"'
if [ ! -x "$bipga_root/.venv/bin/python" ]; then
  "$bipga_python" -m venv "$bipga_root/.venv"
fi
bipga_interpreter="$bipga_root/.venv/bin/python"
"$bipga_interpreter" -m pip install --upgrade pip
if [ "$bipga_mode" = gpu ]; then
  "$bipga_interpreter" -m pip install --no-cache-dir torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
else
  "$bipga_interpreter" -m pip install --no-cache-dir torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
fi
if [ -d "$bipga_root/wheels" ]; then
  "$bipga_interpreter" -m pip install --find-links "$bipga_root/wheels" -c "$bipga_root/requirements-lock.txt" 'bipga[all]==0.1.0'
else
  "$bipga_interpreter" -m pip install -c "$bipga_root/requirements-lock.txt" "$bipga_root[all]"
fi
if [ -f "$bipga_root/paper_artifact.tar.xz" ] && [ ! -e "$bipga_root/paper" ]; then
  "$bipga_interpreter" "$bipga_root/tools/extract_paper.py" "$bipga_root/paper_artifact.tar.xz" "$bipga_root/paper"
fi
"$bipga_root/.venv/bin/bipga" doctor --all
printf '\nInstalled. Run: bash bipga.sh methods\n'
