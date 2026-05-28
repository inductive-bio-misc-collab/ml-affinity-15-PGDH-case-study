"""Compute symmetry-corrected ligand RMSD between Boltz-predicted poses and the
docked poses in poses.sdf, and plot a histogram + per-series boxplot.

Inputs:
  data/poses.sdf
  data/reference.pdb
  scoring/boltz_results_boltz_yamls/predictions/<id>/<id>_model_0.pdb

Output:
  figures/boltz_rmsd.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser, Superimposer
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, PandasTools

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SDF = ROOT / "data" / "poses.sdf"
REF_PDB = ROOT / "data" / "reference.pdb"
BOLTZ_DIR = ROOT / "scoring" / "boltz_results_boltz_yamls" / "predictions"
FIG = ROOT / "figures" / "boltz_rmsd.png"

parser = PDBParser(QUIET=True)


def reference_cas():
    ref = parser.get_structure("ref", str(REF_PDB))
    return [
        res["CA"]
        for res in ref[0]["A"].get_residues()
        if res.id[2] == " " and 1 <= res.id[1] <= 262 and "CA" in res
    ]


def compute_rmsd(compound_id: str, ref_cas, sdf_mol):
    pdb_path = BOLTZ_DIR / compound_id / f"{compound_id}_model_0.pdb"
    if not pdb_path.exists():
        return None

    boltz_struct = parser.get_structure("boltz", str(pdb_path))
    boltz_cas = [
        res["CA"]
        for res in boltz_struct[0]["A"].get_residues()
        if 1 <= res.id[1] <= 262 and "CA" in res
    ]
    if len(boltz_cas) != len(ref_cas):
        return None

    sup = Superimposer()
    sup.set_atoms(ref_cas, boltz_cas)
    sup.apply(list(boltz_struct.get_atoms()))

    boltz_lig_atoms = list(boltz_struct[0]["B"].get_atoms())
    boltz_lig_coords = np.array([a.get_vector().get_array() for a in boltz_lig_atoms])

    pdb_text = pdb_path.read_text().split("\n")
    chain_b = [l for l in pdb_text if l.startswith("HETATM") and len(l) > 21 and l[21] == "B"]
    serials = {int(l[6:11].strip()) for l in chain_b}
    conects = [
        l for l in pdb_text
        if l.startswith("CONECT") and int(l.split()[1]) in serials
    ]
    block = "\n".join(chain_b + conects + ["END"])
    boltz_rdkit = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True)
    if boltz_rdkit is None:
        return None

    sdf_noH = Chem.RemoveHs(sdf_mol)
    try:
        boltz_fixed = AllChem.AssignBondOrdersFromTemplate(sdf_noH, boltz_rdkit)
    except Exception:
        return None

    matches = sdf_noH.GetSubstructMatches(boltz_fixed, uniquify=False)
    if not matches:
        return None

    sdf_conf = sdf_noH.GetConformer()
    sdf_coords = np.array([list(sdf_conf.GetAtomPosition(i)) for i in range(sdf_noH.GetNumAtoms())])

    best = float("inf")
    for match in matches:
        diffs = np.array([
            boltz_lig_coords[i] - sdf_coords[match[i]]
            for i in range(boltz_rdkit.GetNumAtoms())
        ])
        best = min(best, float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))))

    return best


def main() -> None:
    ref_cas = reference_cas()
    df_sdf = PandasTools.LoadSDF(str(SDF))

    rows = []
    for _, row in df_sdf.iterrows():
        rmsd = compute_rmsd(row["ID"], ref_cas, row["ROMol"])
        if rmsd is not None:
            rows.append({"ID": row["ID"], "series": row["series"], "RMSD": rmsd})
    df_rmsd = pd.DataFrame(rows)

    print(f"n={len(df_rmsd)}  mean={df_rmsd['RMSD'].mean():.3f}  "
          f"median={df_rmsd['RMSD'].median():.3f}  std={df_rmsd['RMSD'].std():.3f}")
    print(df_rmsd.groupby("series")["RMSD"].agg(["count", "mean", "median", "std"]).round(3))

    sns.set_style("darkgrid")
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["figure.dpi"] = 200

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(df_rmsd["RMSD"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(df_rmsd["RMSD"].mean(), color="red", linestyle="--",
               label=f"Mean = {df_rmsd['RMSD'].mean():.2f} Å")
    ax.axvline(df_rmsd["RMSD"].median(), color="orange", linestyle="--",
               label=f"Median = {df_rmsd['RMSD'].median():.2f} Å")
    ax.set_xlabel("Ligand RMSD (Å)")
    ax.set_ylabel("Count")
    ax.set_title("Boltz 2 Ligand RMSD vs Docked Poses")
    ax.legend()

    ax = axes[1]
    order = sorted(df_rmsd["series"].unique())
    sns.boxplot(data=df_rmsd, x="series", y="RMSD", order=order, ax=ax, fliersize=0)
    sns.stripplot(data=df_rmsd, x="series", y="RMSD", order=order, ax=ax,
                  color="black", alpha=0.5, size=3)
    ax.set_xlabel("Series")
    ax.set_ylabel("Ligand RMSD (Å)")
    ax.set_title("Ligand RMSD by Series")

    FIG.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
