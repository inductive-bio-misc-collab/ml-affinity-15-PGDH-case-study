#!/usr/bin/env bash
# Score docked poses with Vina (via gnina --minimize, CNN scoring disabled).
# Requires gnina installed locally (see https://github.com/gnina/gnina).
# Output SDF carries a `minimizedAffinity` property per ligand (kcal/mol).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

gnina \
  -r "${HERE}/../data/reference.pdb" \
  -l "${HERE}/../data/poses.sdf" \
  -o "${HERE}/vina_predictions.sdf" \
  --minimize \
  --seed 1 \
  --cnn_scoring none
