#!/usr/bin/env bash
# Tao va cai dat mot moi truong Python 3.12 tai thu muc goc.
set -euo pipefail
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Khong tim thay python3.12. Hay cai Python 3.12 va them vao PATH." >&2
  exit 1
fi
python3.12 --version
if [[ ! -d ".venv" ]]; then python3.12 -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cat <<'MSG'

Da cai dat xong trong .venv (Git da bo qua thu muc nay).
Kich hoat: source .venv/bin/activate
Kiem tra: python --version
MSG
