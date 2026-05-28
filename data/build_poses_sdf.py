"""Build data/poses.sdf from the original combined_set.sdf.

Run once. Keeps only the 44 ligands with paper Compound numbers (Series 1, 2,
3, 4/5), renames IDs to compound_<N>, converts FEP+ free energies to pKi,
and drops Nimbus-internal columns.
"""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import PandasTools, SDWriter

SOURCE = Path(__file__).resolve().parents[2] / "data" / "combined_set.sdf"
OUTPUT = Path(__file__).parent / "poses.sdf"

KCAL_TO_PKI = 1.364


def main() -> None:
    df = PandasTools.LoadSDF(str(SOURCE), removeHs=False)
    df = df[df["table"].isin(["1", "2", "3", "4/5"])]
    df = df[df["ID"] != "NHT-00994602"].reset_index(drop=True)
    assert len(df) == 43, f"expected 43 compounds, got {len(df)}"
    assert df["Compound"].is_unique, "Compound numbers must be unique"

    writer = SDWriter(str(OUTPUT))
    for _, row in df.iterrows():
        mol = row["ROMol"]
        for prop in list(mol.GetPropsAsDict().keys()):
            mol.ClearProp(prop)

        compound_id = f"compound_{int(float(row['Compound']))}"
        series = row["table"].replace("/", "_")
        pki = -float(row["r_fepplus_exp_dg"]) / KCAL_TO_PKI
        fep_pki = -float(row["r_fepplus_pred_dg"]) / KCAL_TO_PKI

        mol.SetProp("_Name", compound_id)
        mol.SetProp("ID", compound_id)
        mol.SetProp("series", series)
        mol.SetProp("pKi", f"{pki:.4f}")
        mol.SetProp("FEP_pKi", f"{fep_pki:.4f}")
        mol.SetProp("smiles", Chem.MolToSmiles(Chem.MolFromSmiles(row["smi"])))

        writer.write(mol)
    writer.close()
    print(f"Wrote {OUTPUT} with {len(df)} compounds")


if __name__ == "__main__":
    main()
