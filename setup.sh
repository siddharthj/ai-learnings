#!/usr/bin/env bash
# One-time setup, safe to re-run. Creates .venv, installs requirements.txt,
# and registers a Jupyter kernel called "ai-learning".
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ (https://www.python.org/downloads/ or: brew install python) and re-run." >&2
  exit 1
fi

[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ai-learning --display-name "Python (ai-learning)"

# The graphviz pip package only draws if the Graphviz program is installed too.
if ! command -v dot >/dev/null 2>&1; then
  echo
  echo "Graphviz program not found. micrograd's draw_dot() needs it: brew install graphviz"
fi

echo
echo "Done. Python: $(python --version)"
python -c "import torch, numpy, matplotlib, graphviz; print('torch', torch.__version__)"
echo
echo "Run notebooks with:  source .venv/bin/activate && jupyter lab"
echo "In VS Code, pick the kernel 'Python (ai-learning)' or the .venv interpreter."
