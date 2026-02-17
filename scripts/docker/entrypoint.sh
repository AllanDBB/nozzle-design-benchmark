#!/usr/bin/env bash
set -o pipefail

if [ -f /opt/openfoam13/etc/bashrc ]; then
  export ZSH_NAME="${ZSH_NAME:-}"
  # shellcheck disable=SC1091
  source /opt/openfoam13/etc/bashrc || true
fi

if [ "$#" -eq 0 ]; then
  set -- python3 main_pipeline.py
fi

exec "$@"
