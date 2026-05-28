"""Generate per-series scatter figures, the pooled across-series figure, and
print the overall Spearman correlation matrix.

Inputs (relative to repo root):
  data/poses.sdf
  scoring/boltz_predictions.csv
  scoring/vina_predictions.sdf
  scoring/aevplig_original.csv     (optional)
  scoring/aevplig_finetuned.csv    (optional; shown as "AEVPLIG Retrained")

Outputs:
  figures/figure_2.png       (Series 1 + 2)
  figures/figure_4.png       (Series 3, highlight compound 4)
  figures/figure_5.png       (Series 4_5)
  figures/figure_pooled.png  (all series, colored by series)
  Prints overall Spearman correlation matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.Descriptors import HeavyAtomCount
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SDF = ROOT / "data" / "poses.sdf"
BOLTZ_CSV = ROOT / "scoring" / "boltz_predictions.csv"
VINA_SDF = ROOT / "scoring" / "vina_predictions.sdf"
AEVPLIG_ORIG = ROOT / "scoring" / "aevplig_original.csv"
AEVPLIG_FT = ROOT / "scoring" / "aevplig_finetuned.csv"
FIGURES_DIR = ROOT / "figures"

KCAL_TO_PKI = 1.364

SERIES_LABELS = {"1": "Series 1 & 2", "2": "Series 1 & 2", "3": "Series 3", "4_5": "Series 4"}
SERIES_ORDER = ["Series 1 & 2", "Series 3", "Series 4"]


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = PandasTools.LoadSDF(str(SDF))
    df["pKi"] = df["pKi"].astype(float)
    df["FEP_pKi"] = df["FEP_pKi"].astype(float)
    df["HAC"] = df["ROMol"].apply(HeavyAtomCount)
    df["Compound"] = df["ID"].str.removeprefix("compound_")
    df["Series"] = df["series"].map(SERIES_LABELS)

    boltz = pd.read_csv(BOLTZ_CSV)
    df = df.merge(boltz[["ID", "boltz_affinity_pred_value"]], on="ID")

    vina = PandasTools.LoadSDF(str(VINA_SDF))[["ID", "minimizedAffinity"]]
    vina["minimizedAffinity"] = vina["minimizedAffinity"].astype(float)
    df = df.merge(vina, on="ID")

    methods = ["FEP", "Boltz 2", "Vina"]
    df = df.rename(
        columns={
            "FEP_pKi": "FEP",
            "boltz_affinity_pred_value": "Boltz 2",
            "minimizedAffinity": "Vina",
        }
    )
    df["Vina"] = -df["Vina"].clip(upper=0) / KCAL_TO_PKI
    df["Boltz 2"] = -df["Boltz 2"] + 6

    if AEVPLIG_ORIG.exists():
        a = pd.read_csv(AEVPLIG_ORIG).rename(columns={"preds": "AEVPLIG", "unique_id": "ID"})
        df = df.merge(a[["ID", "AEVPLIG"]], on="ID")
        methods.append("AEVPLIG")
    else:
        print(f"[warn] {AEVPLIG_ORIG.name} not found; skipping AEVPLIG facet")

    if AEVPLIG_FT.exists():
        a = pd.read_csv(AEVPLIG_FT).rename(columns={"preds": "AEVPLIG Retrained", "unique_id": "ID"})
        df = df.merge(a[["ID", "AEVPLIG Retrained"]], on="ID")
        methods.append("AEVPLIG Retrained")
    else:
        print(f"[warn] {AEVPLIG_FT.name} not found; skipping AEVPLIG Retrained facet")

    methods.append("HAC")
    return df, methods


def plot_methods(
    df_plot: pd.DataFrame,
    methods: list[str],
    save_path: Path,
    highlight_compound: str | None = None,
    hue_col: str | None = None,
    hue_order: list[str] | None = None,
    palette: str = "Set1",
) -> None:
    sns.set_style("darkgrid")
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["figure.dpi"] = 200

    df_plot = df_plot.copy()
    df_plot["Experimental pKi"] = df_plot["pKi"]

    id_vars = ["Experimental pKi", "Compound"]
    if hue_col is not None and hue_col not in id_vars:
        id_vars.append(hue_col)
    df_melt = df_plot.melt(
        id_vars=id_vars,
        value_vars=methods,
        var_name="Method",
        value_name="Predicted value",
    )

    spearman = {
        m: spearmanr(
            df_melt[df_melt["Method"] == m]["Experimental pKi"],
            df_melt[df_melt["Method"] == m]["Predicted value"],
        )[0]
        for m in methods
    }

    g = sns.FacetGrid(
        df_melt, col="Method", col_wrap=3, sharex=False, sharey=False, height=3,
        hue=hue_col, hue_order=hue_order,
        palette=palette if hue_col else None,
    )
    g.map_dataframe(sns.scatterplot, x="Experimental pKi", y="Predicted value", alpha=0.7)
    g.set_titles("{col_name}")

    if highlight_compound is not None:
        highlight_label = f"Compound {highlight_compound}"
        sub = df_plot[df_plot["Compound"] == highlight_compound]
        for i, (ax, method) in enumerate(zip(g.axes.flat, g.col_names)):
            for _, row in sub.iterrows():
                ax.scatter(
                    [row["Experimental pKi"]], [row[method]],
                    color="C3", s=50, zorder=5, alpha=0.9,
                    label=highlight_label if i == 0 else None,
                )
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        if handles:
            g.fig.legend(handles, labels, loc="lower right", frameon=False, fontsize=9,
                         bbox_to_anchor=(1.0, -0.02))

    if hue_col is not None:
        g.add_legend()

    for ax, method in zip(g.axes.flat, g.col_names):
        ax.text(0.05, 0.95, f"ρ = {spearman[method]:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=10)

    FIGURES_DIR.mkdir(exist_ok=True)
    g.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(g.fig)
    print(f"Wrote {save_path}")


def main() -> None:
    df, methods = load_data()

    print("\nOverall Spearman correlations (n=%d):" % len(df))
    print(df[["pKi"] + methods].corr(method="spearman").round(3))

    plot_methods(
        df[df["series"].isin(["1", "2"])], methods,
        FIGURES_DIR / "figure_2.png",
    )
    plot_methods(
        df[df["series"] == "3"], methods,
        FIGURES_DIR / "figure_4.png",
        highlight_compound="4",
    )
    plot_methods(
        df[df["series"] == "4_5"], methods,
        FIGURES_DIR / "figure_5.png",
    )
    plot_methods(
        df, methods,
        FIGURES_DIR / "figure_pooled.png",
        hue_col="Series",
        hue_order=SERIES_ORDER,
    )


if __name__ == "__main__":
    main()
