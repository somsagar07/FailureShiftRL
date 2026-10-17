#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -I check_environment.py --quiet; then
  printf 'Setting up the web interface environment…\n'
  bash setup.sh
fi
exec .venv/bin/python -E server.py --open "$@"
