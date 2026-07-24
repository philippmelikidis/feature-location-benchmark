#!/usr/bin/env bash
#
# setup_code_env.sh – isolierte conda-Env für trust_remote_code-Code-Embeddings.
#
# Die Basis-Env hat ein zu neues transformers (5.x dev), das jina/SFR brechen
# lässt. Diese Env pinnt ein stabiles transformers (4.46.3), mit dem jina-code
# (und SFR auf CUDA) laden. Native Modelle (qwen3, codesearch) brauchen das NICHT
# und laufen in deiner Basis-Env.
#
# Aufruf (aus dem Repo-Root benchmark-prostep/):
#   bash scripts/setup_code_env.sh            # Env-Name: benchmark-code
#   bash scripts/setup_code_env.sh meinname   # eigener Env-Name
#
# Danach:
#   conda activate benchmark-code
#   python scripts/run_code_embed_v11plus.py --model jina
#
set -euo pipefail

ENV_NAME="${1:-benchmark-code}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "FEHLER: conda nicht gefunden. (Alternativ: python3.11 -m venv .venv-code)"
  exit 1
fi

echo "==> Erstelle conda-Env '$ENV_NAME' (Python 3.11)…"
conda create -y -n "$ENV_NAME" python=3.11

# conda activate im Skript verfügbar machen
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installiere gepinnte Abhängigkeiten…"
pip install --upgrade pip
pip install -r requirements_code_embed.txt

echo ""
echo "==> Fertig. Sanity-Check transformers:"
python -c "import transformers, transformers.pytorch_utils as p; \
print('transformers', transformers.__version__, \
'| find_pruneable_heads_and_indices:', hasattr(p,'find_pruneable_heads_and_indices'))"

echo ""
echo "Nächste Schritte:"
echo "  conda activate $ENV_NAME"
echo "  python scripts/run_code_embed_v11plus.py --model jina   # starkes Code-Modell (ALiBi, MPS)"
echo "  # SFR (--model sfr) läuft hier nur auf CUDA-Hardware sinnvoll."
