"""Run AEV-PLIG predictions for the 15-PGDH benchmark compounds.

Prerequisites
-------------
1.  Clone AEV-PLIG (published in *Communications Chemistry*):
        git clone https://github.com/isakvals/AEV-PLIG <aevplig-dir>
2.  Create and activate its conda environment (Python 3.8, separate from the
    uv environment used by the rest of this repo):
        # Linux:
        conda env create -f <aevplig-dir>/aev-plig-linux.yml
        conda activate aev-plig
        # macOS:
        conda env create -f <aevplig-dir>/aev-plig-mac.yml
        conda activate aev-plig
3.  (Finetuned model only) Copy the finetuned model weights into
        <aevplig-dir>/output/trained_models/
    Required files (11 total):
        20251101-200924_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_U_additional_ligsim90_fep_benchmark.pickle
        20251101-200924_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_U_additional_ligsim90_fep_benchmark_0.model
        ... (files _1.model through _9.model)
    These weights are available from the paper's data release [TODO: add DOI/URL].

Usage
-----
    # With the aev-plig conda env active, run from the repository root:
    python scoring/run_aevplig.py --aevplig-dir /path/to/AEV-PLIG

Outputs written to scoring/
----------------------------
    aevplig_original.csv   — original published AEV-PLIG weights
    aevplig_finetuned.csv  — finetuned weights (skipped if weights not present)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POSES_SDF = HERE / "vina_predictions.sdf"
REFERENCE_PDB = ROOT / "data" / "reference.pdb"

ORIGINAL_MODEL = "model_GATv2Net_ligsim90_fep_benchmark"
FINETUNED_MODEL = (
    "20251101-200924_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_U_additional"
    "_ligsim90_fep_benchmark"
)


def split_sdf(sdf_path: Path, out_dir: Path) -> list[tuple[str, Path]]:
    """Split a multi-mol SDF into one file per molecule. Returns [(id, path), ...]."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    entries = []
    for mol in suppl:
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        cid = props.get("ID") or mol.GetProp("_Name").strip()
        if not cid:
            raise ValueError(f"Molecule in {sdf_path} has no ID property or name")
        out_path = out_dir / f"{cid}.sdf"
        with SDWriter(str(out_path)) as w:
            w.write(Chem.AddHs(mol, addCoords=True))
        entries.append((cid, out_path))
    return entries


def build_input_csv(
    entries: list[tuple[str, Path]], pdb_path: Path, csv_path: Path
) -> None:
    rows = [
        {"unique_id": cid, "sdf_file": str(sdf), "pdb_file": str(pdb_path)}
        for cid, sdf in entries
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def model_weights_present(aevplig_dir: Path, model_name: str) -> bool:
    pickle_path = aevplig_dir / "output" / "trained_models" / f"{model_name}.pickle"
    return pickle_path.exists()


def run_model(
    aevplig_dir: Path, csv_path: Path, data_name: str, model_name: str
) -> Path:
    cmd = [
        sys.executable,
        "process_and_predict.py",
        f"--dataset_csv={csv_path}",
        f"--data_name={data_name}",
        f"--trained_model_name={model_name}",
    ]
    subprocess.run(cmd, cwd=aevplig_dir, check=True)
    return aevplig_dir / "output" / "predictions" / f"{data_name}_predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--aevplig-dir",
        required=True,
        type=Path,
        help="Path to a cloned AEV-PLIG repository",
    )
    parser.add_argument(
        "--original-model",
        default=ORIGINAL_MODEL,
        help="Trained-model name for the original weights (default: %(default)s)",
    )
    parser.add_argument(
        "--finetuned-model",
        default=FINETUNED_MODEL,
        help="Trained-model name for the finetuned weights (default: %(default)s)",
    )
    args = parser.parse_args()

    aevplig_dir = args.aevplig_dir.resolve()
    if not (aevplig_dir / "process_and_predict.py").exists():
        sys.exit(
            f"process_and_predict.py not found in {aevplig_dir}\n"
            "Check --aevplig-dir points to a cloned AEV-PLIG repo."
        )

    tmp = Path(tempfile.mkdtemp(prefix="aevplig_paper_"))
    try:
        entries = split_sdf(POSES_SDF, tmp)
        csv_path = tmp / "paper_input.csv"
        build_input_csv(entries, REFERENCE_PDB, csv_path)

        runs = [
            (args.original_model, HERE / "aevplig_original.csv", "pgdh_orig"),
            (args.finetuned_model, HERE / "aevplig_finetuned.csv", "pgdh_ft"),
        ]
        for model_name, output_csv, data_name in runs:
            if not model_weights_present(aevplig_dir, model_name):
                print(f"[skip] weights for {model_name!r} not found — skipping")
                continue
            preds_csv = run_model(aevplig_dir, csv_path, data_name, model_name)
            df = pd.read_csv(preds_csv)[["unique_id", "preds"]]
            df.to_csv(output_csv, index=False)
            print(f"Saved {output_csv.relative_to(ROOT)}")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
