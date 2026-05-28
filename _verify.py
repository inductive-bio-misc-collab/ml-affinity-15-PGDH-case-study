"""Verification harness: remap existing scoring outputs (keyed by NDI/NHT IDs)
to compound_<N> IDs so the clean analysis scripts can run end-to-end against
real data, and compare overall correlations against the original notebook.

Not part of the released package — kept under leading-underscore filename.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, SDWriter

REPO = Path(__file__).resolve().parents[1]
ORIG = REPO
CLEAN = Path(__file__).resolve().parent

orig_sdf = PandasTools.LoadSDF(str(ORIG / "data" / "combined_set.sdf"))
orig_sdf = orig_sdf[orig_sdf["table"].isin(["1", "2", "3", "4/5"])]
orig_sdf = orig_sdf[orig_sdf["ID"] != "NHT-00994602"]
nimbus_to_compound = {
    row["ID"]: f"compound_{int(float(row['Compound']))}"
    for _, row in orig_sdf.iterrows()
}
print(f"Mapping {len(nimbus_to_compound)} Nimbus IDs to compound IDs")

# Boltz: remap and re-emit
boltz = pd.read_csv(ORIG / "scoring" / "boltz" / "boltz_predictions.csv")
boltz = boltz[boltz["ID"].isin(nimbus_to_compound)].copy()
boltz["ID"] = boltz["ID"].map(nimbus_to_compound)
boltz.to_csv(CLEAN / "scoring" / "boltz_predictions.csv", index=False)
print(f"Wrote scoring/boltz_predictions.csv ({len(boltz)} rows)")

# Vina: remap SDF
vina = PandasTools.LoadSDF(str(ORIG / "scoring" / "vina_gnina" / "combined_set_gnina_scores.sdf"))
vina = vina[vina["ID"].isin(nimbus_to_compound)].copy()
vina["minimizedAffinity"] = vina["minimizedAffinity"].astype(float)
with SDWriter(str(CLEAN / "scoring" / "vina_predictions.sdf")) as w:
    for _, row in vina.iterrows():
        mol = row["ROMol"]
        new_id = nimbus_to_compound[row["ID"]]
        for prop in list(mol.GetPropsAsDict().keys()):
            mol.ClearProp(prop)
        mol.SetProp("ID", new_id)
        mol.SetProp("_Name", new_id)
        mol.SetProp("minimizedAffinity", str(row["minimizedAffinity"]))
        w.write(mol)
print(f"Wrote scoring/vina_predictions.sdf ({len(vina)} rows)")

# AEV-PLIG
for src_name, dst_name in [
    ("aevplig_myoforte_orig_predictions.csv", "aevplig_original.csv"),
    ("aevplig_myoforte_predictions.csv", "aevplig_finetuned.csv"),
]:
    src = ORIG / "scoring" / "aevplig" / src_name
    a = pd.read_csv(src)
    a = a[a["unique_id"].isin(nimbus_to_compound)].copy()
    a["unique_id"] = a["unique_id"].map(nimbus_to_compound)
    a.to_csv(CLEAN / "scoring" / dst_name, index=False)
    print(f"Wrote scoring/{dst_name} ({len(a)} rows)")

# Boltz pose PDBs: copy and rename inner files for boltz_rmsd / export_aligned_poses
import shutil
predictions_dir = CLEAN / "scoring" / "boltz_results_boltz_yamls" / "predictions"
src_predictions_dir = ORIG / "scoring" / "boltz" / "boltz_results_boltz_yamls" / "predictions"
if predictions_dir.exists():
    shutil.rmtree(predictions_dir)
predictions_dir.mkdir(parents=True)
for nimbus, compound in nimbus_to_compound.items():
    src = src_predictions_dir / nimbus
    if not src.exists():
        continue
    dst = predictions_dir / compound
    dst.mkdir()
    for f in src.iterdir():
        new_name = f.name.replace(nimbus, compound)
        shutil.copy(f, dst / new_name)
print(f"Copied + renamed Boltz prediction dirs to {predictions_dir}")
