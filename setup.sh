#!/usr/bin/env bash
# Bootstrap the pgoggles workspace.

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10.13}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_SDK_REQUIREMENTS="packages/python-sdk/requirements.txt"
IDENTITY_REQUIREMENTS="research/identity/requirements.txt"

echo ""
echo "========================================="
echo " pgoggles Workspace Setup"
echo "========================================="
echo ""

echo "[1/4] Python..."
if [[ -z "${CONDA_DEFAULT_ENV:-}" && ! -f /.dockerenv ]] && command -v pyenv >/dev/null 2>&1; then
    if [[ "$(uname)" == "Darwin" ]]; then
        SDKROOT="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
        if [[ -n "$SDKROOT" ]]; then
            export SDKROOT
            echo "      macOS SDK -> $SDKROOT"
        fi
    fi

    pyenv install "$PYTHON_VERSION" --skip-existing
    pyenv local "$PYTHON_VERSION"
    eval "$(pyenv init -)"
    echo "      Using Python $(python3 --version)"
else
    echo "      Using existing Python $(python3 --version)"
fi

echo "[2/4] Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/4] Installing base SDK dependencies..."
pip install --upgrade pip wheel
pip install -r "$PYTHON_SDK_REQUIREMENTS"
pip install -e packages/python-sdk

if [[ "${PGOGGLES_INSTALL_IDENTITY_RESEARCH:-0}" == "1" ]]; then
    echo "      Installing optional identity research dependencies..."
    pip install -r "$IDENTITY_REQUIREMENTS"
else
    echo "      Skipping identity research dependencies."
    echo "      Set PGOGGLES_INSTALL_IDENTITY_RESEARCH=1 to install them."
fi

echo "[4/4] Creating local environment file..."
if [[ ! -f .env ]]; then
    cat > .env <<'ENV'
# ---- pgoggles workspace ----
PGOGGLES_DEVICE=simulator

# ---- Optional identity research ----
# Explicitly opt in before running research/identity code.
PGOGGLES_IDENTITY_RESEARCH_ENABLED=0
PGOGGLES_DATABASE_URL=
INSIGHTFACE_HOME=.insightface
USE_GPU=0
ENV
    echo "      .env created."
else
    echo "      .env already exists; leaving it untouched."
fi

echo ""
echo "Setup complete"
echo "Activate with: source $VENV_DIR/bin/activate"
