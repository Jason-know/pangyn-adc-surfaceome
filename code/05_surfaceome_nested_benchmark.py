from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
V5 = ROOT / "inputs_v5"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261101

KNOWN = {
    "FOLR1", "MUC16", "MSLN", "TACSTD2", "NECTIN4", "ERBB2", "CDH6",
    "CLDN6", "VTCN1", "F3", "SLC34A2", "CD276", "DLL3", "EPCAM",
    "ITGB6", "CDH3", "MET", "ROR1", "ROR2",
}
CANDIDATES = [
    "CLDN4", "MAL2", "TMPRSS4", "CLDN3", "MUC1",
    "CLDN7", "RAB25", "LSR",
]


def zscore(values):
    values = pd.to_numeric(values, errors="coerce")
    return (values - values.mean()) / values.std(ddof=0)


def rank_desc(series):
    return series.rank(ascending=False, method="min").astype(int)


def recovery_table(frame, score_column, prefix):
    frame = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    n_total = len(frame)
    n_known = int(frame["known_adc_target"].sum())
    rows = []
    for fraction in [0.01, 0.02, 0.05, 0.10, 0.20]:
        cutoff = int(np.ceil(n_total * fraction))
        observed = int(frame.head(cutoff)["known_adc_target"].sum())
        expected = n_known * cutoff / n_total
        p_value = hypergeom.sf(observed - 1, n_total, n_known, cutoff)
        rows.append(
            {
                "benchmark": prefix,
                "fraction": fraction,
                "n_total": n_total,
                "n_known": n_known,
                "top_n": cutoff,
                "observed_known": observed,
                "expected_known": expected,
                "enrichment": observed / expected,
                "hypergeometric_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def matched_known_target_null(frame, score_column, top_fraction=0.10, n_perm=10000):
    work = frame.copy()
    if "topology_class" not in work:
        work["topology_class"] = "unknown"
    work["accessible"] = work.get("high_confidence_accessible_surface", False)
    work["accessible"] = work["accessible"].fillna(False).astype(bool)
    work["coverage_bin"] = pd.qcut(
        work["coverage_mean"].rank(method="first"),
        10,
        labels=False,
    )
    work["detect_bin"] = pd.qcut(
        work["detect"].rank(method="first"),
        10,
        labels=False,
    )
    top_n = int(np.ceil(len(work) * top_fraction))
    top_genes = set(work.nlargest(top_n, score_column).index)
    known = work[work["known_adc_target"]]
    observed = len(top_genes.intersection(known.index))

    pools = {}
    for key, group in work[~work["known_adc_target"]].groupby(
        ["coverage_bin", "detect_bin", "accessible"]
    ):
        pools[key] = group.index.to_numpy()
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(n_perm):
        sampled = []
        for _, row in known.iterrows():
            key = (row["coverage_bin"], row["detect_bin"], row["accessible"])
            candidates = pools.get(key, np.array([], dtype=object))
            if len(candidates):
                sampled.append(rng.choice(candidates))
        null.append(len(set(sampled).intersection(top_genes)))
    null = np.asarray(null)
    return {
        "top_fraction": top_fraction,
        "observed_known": observed,
        "known_total": len(known),
        "null_mean": float(null.mean()),
        "null_95th": float(np.quantile(null, 0.95)),
        "empirical_p": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
        "n_permutations": n_perm,
    }


def main():
    surfaceome = pd.read_csv(RES / "surfaceome_landscape_by_cancer.csv", index_col=0)
    topology = pd.read_csv(V5 / "surfaceome_topology_annotations_v5.csv")
    topology = topology.drop_duplicates("gene").set_index("gene")
    surfaceome = surfaceome.join(
        topology[["topology_class", "high_confidence_accessible_surface"]],
        how="left",
    )
    surfaceome["known_adc_target"] = surfaceome.index.isin(KNOWN)

    expression = pd.read_csv(
        ROOT / "data" / "tcga" / "surfaceome_tpm.csv.gz",
        index_col=0,
    )
    samples = pd.read_csv(ROOT / "data" / "tcga" / "surfaceome_samples.csv", index_col=0)
    samples["type"] = samples.index.to_series().str.split("-").str[3].str[:2]
    samples["patient"] = samples.index.to_series().str[:12]
    samples = samples[samples["type"] == "01"].drop_duplicates("patient")
    expression = expression.loc[
        expression.index.intersection(surfaceome.index),
        samples.index,
    ]
    cancer_coverage = {}
    for cancer in ["TCGA-OV", "TCGA-UCEC", "TCGA-CESC"]:
        ids = samples.index[samples["project"] == cancer]
        cancer_coverage[cancer] = (expression[ids] > 10).mean(axis=1)
    coverage = pd.DataFrame(cancer_coverage)
    surfaceome = surfaceome.join(
        coverage.rename(columns={column: f"coverage_{column}" for column in coverage}),
        how="left",
    )
    surfaceome["coverage_mean"] = coverage.mean(axis=1)
    surfaceome["detect"] = (expression > 0).mean(axis=1)

    surfaceome["expression_only_score"] = (
        zscore(surfaceome["focus_selectivity"])
        + zscore(surfaceome["all_selectivity"])
        + zscore(surfaceome["coverage_mean"])
    ) / 3.0
    surfaceome["expression_only_rank"] = rank_desc(surfaceome["expression_only_score"])

    for cancer in ["OV", "UCEC", "CESC"]:
        focus = zscore(
            surfaceome[f"TCGA-{cancer}"] - surfaceome["Normal gynecologic"]
        )
        all_normal = zscore(
            surfaceome[f"TCGA-{cancer}"] - surfaceome["Normal all"]
        )
        cancer_coverage_score = zscore(surfaceome[f"coverage_TCGA-{cancer}"])
        surfaceome[f"{cancer}_expression_score"] = (
            focus + all_normal + cancer_coverage_score
        ) / 3.0
        surfaceome[f"{cancer}_rank"] = rank_desc(surfaceome[f"{cancer}_expression_score"])

    full = pd.read_csv(V5 / "surfaceome_discovery_topology_v5.csv", index_col=0)
    full = full.rename(columns={"symbol": "gene"})
    candidates = pd.read_csv(V5 / "surfaceome_discovery_candidates_v5.csv", index_col=0)
    full = full.join(
        candidates[["rho_ov", "n_ov", "rho_ucec", "n_ucec"]],
        how="left",
    )
    full["known_adc_target"] = full.index.isin(KNOWN)
    full = full.join(
        surfaceome[["coverage_mean", "detect"]].rename(
            columns={
                "coverage_mean": "coverage_mean_landscape",
                "detect": "detect_landscape",
            }
        ),
        how="left",
    )
    full["coverage_mean"] = full["coverage_mean_landscape"]
    full["detect"] = full["detect_landscape"]
    full = full.drop(columns=["coverage_mean_landscape", "detect_landscape"])

    benchmark_sets = {
        "full_3131_with_cellular_specificity": (full, "surfaceome_score"),
        "expression_only_3640": (surfaceome, "expression_only_score"),
        "OV_expression_only": (surfaceome, "OV_expression_score"),
        "UCEC_expression_only": (surfaceome, "UCEC_expression_score"),
        "CESC_expression_only": (surfaceome, "CESC_expression_score"),
    }
    recovery = pd.concat(
        [
            recovery_table(frame, score, label)
            for label, (frame, score) in benchmark_sets.items()
        ],
        ignore_index=True,
    )
    recovery.to_csv(OUT / "surfaceome_nested_recovery_v6.csv", index=False)

    metrics = []
    for label, (frame, score) in benchmark_sets.items():
        valid = frame[score].notna()
        labels = frame.loc[valid, "known_adc_target"].astype(int)
        percentiles = frame.loc[valid, score].rank(pct=True, ascending=True)
        metrics.append(
            {
                "benchmark": label,
                "n_genes": int(valid.sum()),
                "n_known": int(labels.sum()),
                "auroc": roc_auc_score(labels, frame.loc[valid, score]),
                "average_precision": average_precision_score(labels, frame.loc[valid, score]),
                "known_median_percentile": float(
                    percentiles[labels.eq(1)].median()
                ),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "surfaceome_nested_metrics_v6.csv", index=False)

    null = matched_known_target_null(full, "surfaceome_score")
    pd.DataFrame([null]).to_csv(OUT / "surfaceome_known_target_matched_null_v6.csv", index=False)

    accessible = pd.read_csv(
        V5 / "surfaceome_discovery_extracellular_candidates_v5.csv",
        index_col=0,
    )
    stage_counts = pd.DataFrame(
        [
            {
                "stage": "1 membrane-associated surfaceome input",
                "genes": len(surfaceome),
                "known_targets": int(surfaceome["known_adc_target"].sum()),
            },
            {
                "stage": "2 complete RNA and ovarian cellular discovery",
                "genes": len(full),
                "known_targets": int(full["known_adc_target"].sum()),
            },
            {
                "stage": "3 measurable in both CPTAC cohorts",
                "genes": len(candidates),
                "known_targets": int(candidates["known_adc_target"].sum()),
            },
            {
                "stage": "4 high-confidence extracellular topology",
                "genes": len(accessible),
                "known_targets": int(accessible["known_adc_target"].sum()),
            },
        ]
    )
    stage_counts.to_csv(OUT / "surfaceome_nested_stage_counts_v6.csv", index=False)

    top50 = {
        cancer: set(
            surfaceome.nsmallest(50, f"{cancer}_rank").index
        )
        for cancer in ["OV", "UCEC", "CESC"]
    }
    overlap_rows = []
    for first, second in [("OV", "UCEC"), ("OV", "CESC"), ("UCEC", "CESC")]:
        overlap_rows.append(
            {
                "comparison": f"{first}_vs_{second}",
                "top50_overlap": len(top50[first] & top50[second]),
                "jaccard": len(top50[first] & top50[second]) / len(top50[first] | top50[second]),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(OUT / "surfaceome_cancer_specific_top50_v6.csv", index=False)

    surfaceome.reset_index().to_csv(OUT / "surfaceome_nested_scores_v6.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    print(pd.DataFrame([null]).round(4).to_string(index=False))
    print(pd.DataFrame(overlap_rows).round(3).to_string(index=False))
    print(stage_counts.to_string(index=False))


if __name__ == "__main__":
    main()
