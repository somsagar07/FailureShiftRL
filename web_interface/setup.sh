#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

printf 'Installing the web interface requirements into %s/.venv\n' "$PWD"
# Use CPU wheels for this CPU-only viewer instead of pulling CUDA runtimes.
if ! .venv/bin/python -I -c 'import torch; assert (2,6) <= tuple(map(int, torch.__version__.split(".")[:2])) < (3,0)' >/dev/null 2>&1; then
  if ! .venv/bin/python -I -m pip install --disable-pip-version-check --no-input --retries 1 --timeout 20 --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'; then
    printf '\nCould not install CPU PyTorch. Check internet access, then rerun: bash setup.sh\n' >&2
    exit 1
  fi
fi
if ! .venv/bin/python -I -m pip install --disable-pip-version-check --no-input --retries 1 --timeout 20 -r requirements.txt; then
  printf '\nSetup could not finish. Check the package-download error above and rerun: bash setup.sh\n' >&2
  exit 1
fi
.venv/bin/python -I -m pip check
.venv/bin/python -I check_environment.py
printf '\nEnvironment ready. Start the interface with: bash run.sh\n'
