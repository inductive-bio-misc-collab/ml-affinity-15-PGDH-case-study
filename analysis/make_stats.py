"""Spearman ρ with 95% bootstrap CIs and BH-FDR-adjusted significance vs FEP+,
both pooled and per series. Reproduces the content of paper Table 2.

Inputs: same as make_figures.py.

Output:
  Prints a Methods × {All series, Series 1 & 2, Series 3, Series 4} table.
  Writes figures/correlations_table.csv (same content, machine-readable).

Method
------
Nonparametric paired bootstrap (10,000 resamples, seed=0) over compounds
within each series group. 95% CIs from 2.5/97.5 percentiles. Two-sided
paired-bootstrap p-value for H0: ρ_method == ρ_FEP, then Benjamini–Hochberg
FDR correction across the 5 alternative methods per series.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from make_figures import load_data

N_BOOT = 10_000
RNG_SEED = 0
CI_LOW, CI_HIGH = 2.5, 97.5
OUT_CSV = Path(__file__).resolve().parents[1] / "figures" / "correlations_table.csv"


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho, _ = spearmanr(x, y)
    return rho


def paired_bootstrap(
    group_df: pd.DataFrame, methods: list[str], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    exp = group_df["pKi"].to_numpy()
    n = len(exp)
    preds = {m: group_df[m].to_numpy() for m in methods}
    out = {m: np.empty(N_BOOT) for m in methods}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        e = exp[idx]
        for m in methods:
            out[m][b] = _spearman_safe(e, preds[m][idx])
    return out


def bh_fdr(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def summarize_group(
    group_df: pd.DataFrame,
    methods: list[str],
    alt_methods: list[str],
    rng: np.random.Generator,
):
    boots = paired_bootstrap(group_df, methods, rng)
    exp = group_df["pKi"].to_numpy()
    point = {m: _spearman_safe(exp, group_df[m].to_numpy()) for m in methods}
    ci = {
        m: (np.nanpercentile(boots[m], CI_LOW), np.nanpercentile(boots[m], CI_HIGH))
        for m in methods
    }

    pvals_raw = {}
    for m in alt_methods:
        diff = boots[m] - boots["FEP"]
        diff = diff[np.isfinite(diff)]
        if len(diff) == 0:
            pvals_raw[m] = np.nan
            continue
        centered = diff - diff.mean()
        observed = point[m] - point["FEP"]
        pvals_raw[m] = float(np.mean(np.abs(centered) >= np.abs(observed)))

    pvals_adj_arr = bh_fdr([pvals_raw[m] for m in alt_methods])
    pvals_adj = dict(zip(alt_methods, pvals_adj_arr))
    return point, ci, pvals_raw, pvals_adj


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_cell(rho: float, lo: float, hi: float, p_adj: float) -> str:
    if not np.isfinite(rho):
        return "—"
    return f"{rho:.2f}{_stars(p_adj)} [{lo:+.2f}, {hi:+.2f}]"


def main() -> None:
    df, methods = load_data()
    alt_methods = [m for m in methods if m != "FEP"]

    groups = {
        "All series": df,
        "Series 1 & 2": df[df["series"].isin(["1", "2"])],
        "Series 3": df[df["series"] == "3"],
        "Series 4": df[df["series"] == "4_5"],
    }
    rng = np.random.default_rng(RNG_SEED)
    results = {
        name: summarize_group(g.reset_index(drop=True), methods, alt_methods, rng)
        for name, g in groups.items()
    }

    rows = []
    for m in methods:
        row = {"Method": m}
        for name in groups:
            point, ci, _, p_adj = results[name]
            row[f"{name} (n={len(groups[name])})"] = _fmt_cell(
                point[m], *ci[m], p_adj.get(m, np.nan)
            )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Method")
    print(table.to_string())

    OUT_CSV.parent.mkdir(exist_ok=True)
    table.to_csv(OUT_CSV)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
