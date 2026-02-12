#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$REPO_ROOT/GeomDiagnostics"
INSTALL_ROOT="${1:-$REPO_ROOT/.cli}"
CLI_NAME="${2:-geomdiagnostics}"

if ! command -v julia >/dev/null 2>&1; then
  echo "error: julia is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/Project.toml" ]]; then
  echo "error: expected Julia project at $PROJECT_DIR" >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT"

echo "[1/3] Instantiating Julia environment..."
julia --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate()'

echo "[2/3] Installing Comonicon entrypoint..."
julia --project="$PROJECT_DIR" -e "using GeomDiagnostics; GeomDiagnostics.comonicon_install(name=\"$CLI_NAME\", install_path=raw\"$INSTALL_ROOT\", install_completion=false)"

BINARY_PATH="$INSTALL_ROOT/bin/$CLI_NAME"

echo "[3/3] Done"
echo "CLI path: $BINARY_PATH"
echo "Try: $BINARY_PATH normal 100 --seed 999"
