# Code for "Evaluating Machine Learning Affinity Prediction Methods on Novel Lead Optimization Data: A 15-PGDH Case Study"

This directory contains the code and data needed to reproduce the results and
figures in the paper. All analysis scripts (figures, statistics) run with the
`uv`-managed Python environment defined here. The AEV-PLIG scoring method
requires a separate setup described below.

## Directory layout

```
.
  data/
    poses.sdf          43 compounds used in the paper (3D poses, experimental pKi, FEP+ pKi)
    reference.pdb      15-PGDH receptor structure (PDB 9PFL)
    msa.a3m            Protein MSA used for Boltz-2 co-folding
    build_poses_sdf.py Script that built poses.sdf from the raw Nimbus data
  scoring/
    run_aevplig.py     Generate AEV-PLIG predictions (see AEV-PLIG section below)
    run_boltz.py       Generate Boltz-2 predictions
    run_vina.sh        Generate Vina/GNINA predictions
    aevplig_original.csv   Pre-computed AEV-PLIG (original weights) predictions
    aevplig_finetuned.csv  Pre-computed AEV-PLIG (finetuned weights) predictions
    boltz_predictions.csv  Pre-computed Boltz-2 predictions
    vina_predictions.sdf   Pre-computed Vina predictions
  analysis/
    make_figures.py    Reproduce all paper figures (Figure 2, 4, 5, pooled)
    make_stats.py      Reproduce Table 2 (Spearman ρ with bootstrap CIs)
    boltz_rmsd.py      Boltz-2 pose RMSD vs. reference (co-folding accuracy)
    export_aligned_poses.py  Export aligned protein-ligand complexes
  pyproject.toml       uv project, Python 3.11–3.12
  uv.lock              Locked dependencies
```

## Quick start: reproduce figures and statistics

Pre-computed prediction CSVs are included, so you can jump straight to the
analysis without re-running any scoring methods.

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync --locked
uv run python analysis/make_figures.py   # writes figures/ directory
uv run python analysis/make_stats.py     # prints Table 2, writes figures/correlations_table.csv
```

## Re-running the scoring methods

### Boltz-2

```bash
uv run python scoring/run_boltz.py --run      # run Boltz-2 inference
uv run python scoring/run_boltz.py --collect  # parse results into boltz_predictions.csv
```

Boltz-2 requires a GPU and separate installation (`pip install boltz`).
See `run_boltz.py` for full details.

### Vina / GNINA

```bash
bash scoring/run_vina.sh
```

Requires GNINA to be installed and on `PATH`.

### AEV-PLIG

AEV-PLIG uses Python 3.8 with a separate conda environment and cannot be
installed into the `uv` environment used by the rest of this repo. Follow the
steps below.

#### 1. Clone AEV-PLIG

```bash
git clone https://github.com/isakvals/AEV-PLIG /path/to/AEV-PLIG
```

The original AEV-PLIG paper is:
> Valsson et al., *Communications Chemistry* (2025).
> https://doi.org/10.1038/s42004-025-01428-y

#### 2. Set up the conda environment

```bash
# Linux:
conda env create -f /path/to/AEV-PLIG/aev-plig-linux.yml
# macOS:
conda env create -f /path/to/AEV-PLIG/aev-plig-mac.yml

conda activate aev-plig
```

#### 3. Obtain model weights

The **original** weights (`model_GATv2Net_ligsim90_fep_benchmark`) are
included in the AEV-PLIG GitHub repo under `output/trained_models/` and
require no extra download.

The **finetuned** weights used in this paper were trained on an expanded
dataset (PDBbind + BindingNet + BindingDB-DCS + additional data). These are
available from the paper data release [TODO: add DOI/URL once published].
Copy the 11 weight files into `/path/to/AEV-PLIG/output/trained_models/`:

```
20251101-200924_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_U_additional_ligsim90_fep_benchmark.pickle
20251101-200924_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_U_additional_ligsim90_fep_benchmark_0.model
...
20251101-200924_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_U_additional_ligsim90_fep_benchmark_9.model
```

#### 4. Run predictions

With the `aev-plig` conda environment active, run from the repository root:

```bash
conda activate aev-plig
python scoring/run_aevplig.py --aevplig-dir /path/to/AEV-PLIG
```

This writes `scoring/aevplig_original.csv` and (if finetuned weights are
present) `scoring/aevplig_finetuned.csv`. The analysis scripts pick up these
files automatically on the next run.

The script splits `scoring/vina_predictions.sdf` (Vina-minimized poses) into
per-compound SDF files, builds the required input CSV, and calls AEV-PLIG's
`process_and_predict.py` for each set of weights. Run time is a few minutes
on CPU.
