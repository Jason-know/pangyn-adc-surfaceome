import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "inputs_v5"
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

KNOWN = {
    "FOLR1", "MUC16", "MSLN", "TACSTD2", "NECTIN4", "ERBB2", "CDH6",
    "CLDN6", "VTCN1", "F3", "SLC34A2", "CD276", "DLL3", "EPCAM",
    "ITGB6", "CDH3", "MET", "ROR1", "ROR2",
}
CANDIDATES = [
    "CLDN4", "MAL2", "TMPRSS4", "CLDN3", "MUC1",
    "CLDN7", "RAB25", "LSR",
]


def load_cspa():
    table = pd.read_excel(DATA / "table_S3_surfaceome.xlsx", sheet_name="SurfaceomeMasterTable", header=1)
    table = table.rename(columns={"UniProt gene": "gene"})
    table["cspa_high_confidence"] = table["CSPA category"].fillna("").str.contains("1 - high confidence")
    table["cspa_any"] = table["CSPA category"].notna()
    table["uniprot_surface"] = table["Surfaceome Label"].eq("surface")
    return table[
        ["gene", "cspa_high_confidence", "cspa_any", "uniprot_surface", "CSPA peptide count"]
    ].drop_duplicates("gene")


def load_hpa():
    with zipfile.ZipFile(DATA / "proteinatlas.tsv.zip") as archive:
        with archive.open("proteinatlas.tsv") as handle:
            columns = [
                "Gene",
                "Uniprot",
                "Subcellular location",
                "Subcellular main location",
                "Subcellular additional location",
                "Reliability (IF)",
            ]
            hpa = pd.read_csv(handle, sep="\t", usecols=columns)
    hpa = hpa.rename(columns={"Gene": "gene", "Uniprot": "uniprot"})
    locations = (
        hpa["Subcellular location"].fillna("")
        + ";"
        + hpa["Subcellular main location"].fillna("")
        + ";"
        + hpa["Subcellular additional location"].fillna("")
    )
    hpa["hpa_plasma_membrane"] = locations.str.contains(
        r"plasma membrane|cell membrane|cell surface",
        case=False,
        regex=True,
    )
    hpa["hpa_membrane_associated"] = locations.str.contains(
        r"plasma membrane|cell membrane|cell surface|cell junction",
        case=False,
        regex=True,
    )
    return hpa[
        [
            "gene",
            "uniprot",
            "hpa_plasma_membrane",
            "hpa_membrane_associated",
            "Reliability (IF)",
        ]
    ].drop_duplicates("gene")


def benchmark(frame, annotation):
    valid = frame[frame[annotation].notna()].copy()
    y = valid[annotation].astype(bool)
    auc = roc_auc_score(y, valid["surfaceome_score"])
    top = valid["surfaceome_decile"] == 0
    positive_top = int((top & y).sum())
    negative_top = int((top & ~y).sum())
    positive_rest = int((~top & y).sum())
    negative_rest = int((~top & ~y).sum())
    odds_ratio, p_value = fisher_exact(
        [[positive_top, negative_top], [positive_rest, negative_rest]],
        alternative="greater",
    )
    known = valid["gene"].isin(KNOWN)
    return {
        "annotation": annotation,
        "n_evaluable": len(valid),
        "n_annotated_positive": int(y.sum()),
        "annotation_prevalence": float(y.mean()),
        "auroc": auc,
        "top_decile_positive": positive_top,
        "top_decile_n": positive_top + negative_top,
        "top_decile_rate": positive_top / max(1, positive_top + negative_top),
        "rest_rate": positive_rest / max(1, positive_rest + negative_rest),
        "odds_ratio": odds_ratio,
        "fisher_p": p_value,
        "known_target_annotation_rate": float(y[known].mean()) if known.any() else np.nan,
        "other_gene_annotation_rate": float(y[~known].mean()) if (~known).any() else np.nan,
    }


def main():
    discovery = pd.read_csv(V5 / "surfaceome_discovery_v5.csv").rename(columns={"symbol": "gene"})
    cspa = load_cspa()
    hpa = load_hpa()
    data = discovery.merge(cspa, on="gene", how="left").merge(hpa, on="gene", how="left")
    boolean_columns = [
        "cspa_high_confidence",
        "cspa_any",
        "uniprot_surface",
        "hpa_plasma_membrane",
        "hpa_membrane_associated",
    ]
    data[boolean_columns] = data[boolean_columns].fillna(False).astype(bool)
    data["surfaceome_decile"] = pd.qcut(
        data["surfaceome_score"].rank(method="first", ascending=False),
        10,
        labels=False,
    )
    data["surfaceome_score_percentile"] = (
        data["surfaceome_score"].rank(pct=True, ascending=True)
    )
    data["known_adc_target"] = data["gene"].isin(KNOWN)
    data["non_panel_candidate"] = data["gene"].isin(CANDIDATES)
    data.to_csv(OUT / "surface_proteome_annotation_support_v6.csv", index=False)

    metrics = pd.DataFrame(
        [
            benchmark(data, "cspa_high_confidence"),
            benchmark(data, "cspa_any"),
            benchmark(data, "hpa_plasma_membrane"),
            benchmark(data, "hpa_membrane_associated"),
        ]
    )
    metrics["fdr_bh"] = np.nan
    from statsmodels.stats.multitest import multipletests

    metrics.loc[:, "fdr_bh"] = multipletests(metrics["fisher_p"], method="fdr_bh")[1]
    metrics.to_csv(OUT / "surface_proteome_benchmark_v6.csv", index=False)

    decile = (
        data.groupby("surfaceome_decile")
        .agg(
            n_genes=("gene", "size"),
            mean_score=("surfaceome_score", "mean"),
            cspa_high_confidence_rate=("cspa_high_confidence", "mean"),
            cspa_any_rate=("cspa_any", "mean"),
            hpa_plasma_membrane_rate=("hpa_plasma_membrane", "mean"),
            hpa_membrane_associated_rate=("hpa_membrane_associated", "mean"),
        )
        .reset_index()
    )
    decile["decile_label"] = decile["surfaceome_decile"].map(
        lambda value: f"D{int(value) + 1}"
    )
    decile.to_csv(OUT / "surface_proteome_decile_v6.csv", index=False)

    candidates = data[data["non_panel_candidate"]].copy()
    candidates.to_csv(OUT / "candidate_surface_proteome_support_v6.csv", index=False)

    print(metrics.round(4).to_string(index=False))
    print("\nCandidate support")
    print(
        candidates[
            [
                "gene",
                "surfaceome_rank",
                "cspa_high_confidence",
                "cspa_any",
                "CSPA peptide count",
                "hpa_plasma_membrane",
                "hpa_membrane_associated",
            ]
        ]
        .sort_values("surfaceome_rank")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
