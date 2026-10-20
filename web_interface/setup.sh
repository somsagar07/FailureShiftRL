#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=auto
mode_selected=false
dry_run=false
for argument in "$@"; do
  case "$argument" in
    --auto|--cpu|--cuda)
      if [[ "$mode_selected" == true && "$mode" != "${argument#--}" ]]; then
        printf 'Choose only one of --auto, --cpu, or --cuda.\n' >&2
        exit 2
      fi
      mode="${argument#--}"
      mode_selected=true
      ;;
    --dry-run) dry_run=true ;;
    --help|-h)
      printf 'Usage: bash setup.sh [--auto|--cuda|--cpu] [--dry-run]\n\n'
      printf 'Default: detect NVIDIA hardware and install a compatible CUDA build; otherwise use CPU.\n'
      printf '  --cuda     Require working CUDA, including a verified GPU tensor operation.\n'
      printf '  --cpu      Install/retain CPU-only PyTorch, including on NVIDIA machines.\n'
      printf '  --dry-run  Show the selected build without changing the environment.\n'
      exit 0
      ;;
    *) printf 'Unknown setup option: %s (see --help).\n' "$argument" >&2; exit 2 ;;
  esac
done

probe_python=.venv/bin/python
probe_args=()
if [[ ! -x "$probe_python" ]]; then
  probe_python=python3
  probe_args+=(--fresh)
fi
plan="$($probe_python -I check_environment.py --setup-plan "$mode" "${probe_args[@]}")"
IFS=$'\t' read -r backend action requirement index_url vision_action vision_requirement vision_index torch_pin <<< "$plan"
if [[ "$dry_run" == true ]]; then
  if [[ "$action" == install ]]; then
    printf 'Would install %s from %s into %s/.venv\n' "$requirement" "$index_url" "$PWD"
  fi
  if [[ "$vision_action" != keep ]]; then
    printf 'Would %s %s from %s, preserving %s.\n' "$vision_action" "$vision_requirement" "$vision_index" "$torch_pin"
  fi
  printf 'Would install app requirements, run pip check, install the isolated Vev runtime, and verify the %s runtime.\n' "$backend"
  exit 0
fi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
backup="$(mktemp "${TMPDIR:-/tmp}/failureshiftrl-pip-before-XXXXXX.txt")"
.venv/bin/python -I -m pip freeze --all > "$backup"
printf 'Package versions before setup: %s\n' "$backup"
printf 'Installing the web interface requirements into %s/.venv\n' "$PWD"
pip_options=(--disable-pip-version-check --no-input --no-cache-dir --retries 1 --timeout 60)
if [[ "$action" == install ]]; then
  if ! .venv/bin/python -I -m pip install "${pip_options[@]}" --index-url "$index_url" "$requirement"; then
    printf '\nCould not install %s PyTorch. Check the download error and rerun: bash setup.sh --%s\n' "$backend" "$mode" >&2
    exit 1
  fi
fi
vision_plan="$(.venv/bin/python -I check_environment.py --vision-plan)"
IFS=$'\t' read -r vision_action vision_requirement vision_index torch_pin <<< "$vision_plan"
if [[ "$vision_action" != keep ]]; then
  vision_options=("$torch_pin")
  if [[ "$vision_action" == reinstall ]]; then
    # Repair only the installed vision wheel; never force-reinstall working Torch.
    vision_options=(--force-reinstall --no-deps)
  fi
  if ! .venv/bin/python -I -m pip install "${pip_options[@]}" --index-url "$vision_index" \
      "$vision_requirement" "${vision_options[@]}"; then
    printf '\nCould not install torchvision for %s. The Torch version was pinned; check the download error above.\n' "$torch_pin" >&2
    exit 1
  fi
fi
constraints="$(mktemp "${TMPDIR:-/tmp}/failureshiftrl-torch-constraints-XXXXXX.txt")"
trap 'rm -f -- "$constraints"' EXIT
.venv/bin/python -I check_environment.py --installed-constraints > "$constraints"
if ! .venv/bin/python -I -m pip install "${pip_options[@]}" --constraint "$constraints" -r requirements.txt; then
  printf '\nSetup could not finish. Check the package-download error above and rerun: bash setup.sh\n' >&2
  exit 1
fi
.venv/bin/python -I -m pip check
.venv/bin/python -E setup_vev.py
check_args=()
if [[ "$backend" == cuda ]]; then
  check_args+=(--require-cuda)
fi
.venv/bin/python -I check_environment.py "${check_args[@]}"
printf '\nEnvironment ready. Start the interface with: bash run.sh\n'
