"""Export per-series SDF files for the docked poses and the Boltz poses
(aligned into the reference protein frame). These are the inputs used to
hand-render the binding-pocket figure in PyMOL.

Inputs:
  data/poses.sdf
  data/reference.pdb
  scoring/boltz_results_boltz_yamls/predictions/<id>/<id>_model_0.pdb

Outputs:
  figures/poses/docked_series_<S>.sdf
  figures/poses/boltz_series_<S>.sdf
"""

from pathlib import Path

from Bio.PDB import PDBParser, Superimposer
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, PandasTools, SDWriter

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SDF = ROOT / "data" / "poses.sdf"
REF_PDB = ROOT / "data" / "reference.pdb"
BOLTZ_DIR = ROOT / "scoring" / "boltz_results_boltz_yamls" / "predictions"
OUT_DIR = ROOT / "figures" / "poses"

parser = PDBParser(QUIET=True)


def reference_cas():
    ref = parser.get_structure("ref", str(REF_PDB))
    return [
        res["CA"]
        for res in ref[0]["A"].get_residues()
        if res.id[2] == " " and 1 <= res.id[1] <= 262 and "CA" in res
    ]


def aligned_boltz_ligand(compound_id: str, ref_cas, sdf_mol):
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
    boltz_lig_coords = [a.get_vector().get_array() for a in boltz_lig_atoms]

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

    conf = boltz_fixed.GetConformer()
    for i in range(boltz_fixed.GetNumAtoms()):
        x, y, z = boltz_lig_coords[i].tolist()
        conf.SetAtomPosition(i, (x, y, z))

    boltz_fixed.SetProp("_Name", compound_id)
    return boltz_fixed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_cas = reference_cas()
    df = PandasTools.LoadSDF(str(SDF))

    for series, sub in df.groupby("series"):
        docked_path = OUT_DIR / f"docked_series_{series}.sdf"
        with SDWriter(str(docked_path)) as w:
            for _, row in sub.iterrows():
                mol = Chem.RemoveHs(row["ROMol"])
                mol.SetProp("_Name", row["ID"])
                w.write(mol)

        boltz_path = OUT_DIR / f"boltz_series_{series}.sdf"
        n = 0
        with SDWriter(str(boltz_path)) as w:
            for _, row in sub.iterrows():
                mol = aligned_boltz_ligand(row["ID"], ref_cas, row["ROMol"])
                if mol is not None:
                    w.write(mol)
                    n += 1
        print(f"series {series}: {n}/{len(sub)} Boltz poses written")


if __name__ == "__main__":
    main()
