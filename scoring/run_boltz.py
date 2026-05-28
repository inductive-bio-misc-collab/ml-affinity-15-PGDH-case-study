"""Generate Boltz 2 input YAMLs from poses.sdf, optionally invoke Boltz, and
collect the affinity + confidence results into boltz_predictions.csv.

Usage:
    python run_boltz.py            # generate YAMLs only, print Boltz command
    python run_boltz.py --run      # generate YAMLs and run Boltz
    python run_boltz.py --collect  # parse results into boltz_predictions.csv
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SDF = DATA / "poses.sdf"
MSA = DATA / "msa.a3m"
YAML_DIR = HERE / "boltz_yamls"
RESULTS_DIR = HERE / "boltz_results_boltz_yamls"
OUTPUT_CSV = HERE / "boltz_predictions.csv"

# 15-PGDH (HPGD) sequence used for the published Boltz runs.
PROTEIN_SEQUENCE = (
    "SHVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALDEQFEPQKTLFIQ"
    "CDVADQQQLRDTFRKVVDHFGRLDILVNNAGVNNEKNWEKTLQINLVSVISGTYLGLDYMSK"
    "QNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLMNSGVRLNAICPGFV"
    "NTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLITLIEDDALNGAIMKITTS"
    "KGIHFQDYDTTPFQ"
)

YAML_TEMPLATE = """version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
      msa: {msa_path}
  - ligand:
      id: B
      smiles: '{smiles}'
  - ligand:
      id: C
      ccd: NAD
properties:
  - affinity:
      binder: B
"""


def write_yamls() -> int:
    YAML_DIR.mkdir(exist_ok=True)
    df = PandasTools.LoadSDF(str(SDF))
    for _, row in df.iterrows():
        yaml = YAML_TEMPLATE.format(
            sequence=PROTEIN_SEQUENCE,
            msa_path=str(MSA),
            smiles=row["smiles"],
        )
        (YAML_DIR / f"{row['ID']}.yaml").write_text(yaml)
    return len(df)


def run_boltz() -> None:
    cmd = [
        "boltz", "predict", str(YAML_DIR),
        "--output_format", "pdb",
        "--override", "--seed", "1234",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=HERE, check=True)


def collect_results() -> None:
    pred_dir = RESULTS_DIR / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"{pred_dir} does not exist; run Boltz first")

    rows = []
    for compound_dir in sorted(pred_dir.iterdir()):
        if not compound_dir.is_dir():
            continue
        affinity_json = compound_dir / f"affinity_{compound_dir.name}.json"
        confidence_json = compound_dir / f"confidence_{compound_dir.name}_model_0.json"
        if not affinity_json.exists():
            continue
        result = json.loads(affinity_json.read_text())
        result["ID"] = compound_dir.name
        if confidence_json.exists():
            for k, v in json.loads(confidence_json.read_text()).items():
                if isinstance(v, float):
                    result[k] = v
        rows.append(result)

    df = pd.DataFrame(rows)
    df.columns = ["boltz_" + c if c != "ID" else c for c in df.columns]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="invoke `boltz predict` after writing YAMLs")
    parser.add_argument("--collect", action="store_true", help="parse predictions into boltz_predictions.csv")
    args = parser.parse_args()

    if args.collect:
        collect_results()
        return

    n = write_yamls()
    print(f"Wrote {n} YAMLs to {YAML_DIR}")
    if args.run:
        run_boltz()
        collect_results()
    else:
        print("Next: run `boltz predict boltz_yamls/ --output_format pdb --override` "
              "(or rerun with --run), then `python run_boltz.py --collect`.")


if __name__ == "__main__":
    main()
