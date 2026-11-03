from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t


ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = [
    "FOLR1",
    "TACSTD2",
    "CLDN6",
    "ERBB2",
    "VTCN1",
    "F3",
    "MSLN",
    "NECTIN4",
    "EPCAM",
    "SLC34A2",
]
PRIMARY_CLASS = {
    "FOLR1": "High expression",
    "TACSTD2": "High expression",
    "CLDN6": "High expression",
    "ERBB2": "High expression",
    "F3": "Any positivity",
    "VTCN1": None,
    "MSLN": "Any positivity",
}


def logit_prevalence(y, n):
    adjusted = (y + 0.5) / (n + 1.0)
    return float(np.log(adjusted / (1.0 - adjusted)))


def inverse_logit(value):
    return float(1.0 / (1.0 + np.exp(-value)))


def random_effects(frame):
    y = frame["logit_prev"].to_numpy()
    variance = frame["logit_var"].to_numpy()
    weights = 1.0 / variance
    fixed = np.sum(weights * y) / np.sum(weights)
    q_value = float(np.sum(weights * (y - fixed) ** 2))
    degrees = len(frame) - 1
    c_value = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0.0, (q_value - degrees) / c_value) if degrees > 0 else 0.0
    random_weights = 1.0 / (variance + tau2)
    pooled_logit = float(np.sum(random_weights * y) / np.sum(random_weights))
    standard_error = float(np.sqrt(1.0 / np.sum(random_weights)))
    z_critical = norm.ppf(0.975)
    ci_logit = (
        pooled_logit - z_critical * standard_error,
        pooled_logit + z_critical * standard_error,
    )
    p_heterogeneity = float(chi2.sf(q_value, degrees)) if degrees > 0 else np.nan
    i_squared = max(0.0, (q_value - degrees) / q_value) * 100 if q_value > 0 else 0.0
    if len(frame) >= 3:
        t_critical = t.ppf(0.975, len(frame) - 2)
        pi_logit = (
            pooled_logit - t_critical * np.sqrt(tau2 + standard_error**2),
            pooled_logit + t_critical * np.sqrt(tau2 + standard_error**2),
        )
    else:
        pi_logit = (np.nan, np.nan)
    return {
        "k": len(frame),
        "total_n": int(frame["n_total"].sum()),
        "total_positive": int(frame["n_positive"].sum()),
        "observed_prevalence": float(frame["n_positive"].sum() / frame["n_total"].sum()),
        "pooled_prevalence": inverse_logit(pooled_logit),
        "ci_low": inverse_logit(ci_logit[0]),
        "ci_high": inverse_logit(ci_logit[1]),
        "prediction_low": inverse_logit(pi_logit[0]) if np.isfinite(pi_logit[0]) else np.nan,
        "prediction_high": inverse_logit(pi_logit[1]) if np.isfinite(pi_logit[1]) else np.nan,
        "tau2": tau2,
        "q": q_value,
        "q_df": degrees,
        "q_p": p_heterogeneity,
        "i_squared": i_squared,
    }


def add_model_columns(frame):
    frame = frame.copy()
    frame["logit_prev"] = [
        logit_prevalence(y, n) for y, n in zip(frame["n_positive"], frame["n_total"])
    ]
    frame["logit_var"] = 1.0 / (frame["n_positive"] + 0.5) + 1.0 / (
        frame["n_total"] - frame["n_positive"] + 0.5
    )
    return frame


def main():
    studies = pd.read_csv(LIT / "ihc_study_characteristics.csv")
    estimates = pd.read_csv(LIT / "ihc_prevalence_estimates.csv")
    data = estimates.merge(
        studies[["study_id", "first_author", "year", "pmid", "doi"]],
        on="study_id",
        how="left",
    )
    if data["first_author"].isna().any():
        missing = data.loc[data["first_author"].isna(), "study_id"].unique().tolist()
        raise ValueError(f"Missing study metadata: {missing}")
    if (data["n_positive"] > data["n_total"]).any():
        raise ValueError("Positive events exceed denominator")
    data = add_model_columns(data)
    data["prevalence"] = data["n_positive"] / data["n_total"]

    primary_rows = []
    primary_details = {}
    for target in TARGET_ORDER:
        target_data = data[data["target"] == target].copy()
        threshold = PRIMARY_CLASS.get(target)
        selected = (
            target_data[target_data["threshold_class"] == threshold].copy()
            if threshold
            else target_data.copy()
        )
        if len(selected) < 3:
            continue
        result = random_effects(selected)
        result.update(
            {
                "target": target,
                "analysis_threshold": threshold or "Mixed thresholds",
                "model": "DerSimonian-Laird random effects on logit prevalence",
            }
        )
        primary_rows.append(result)
        primary_details[target] = selected.sort_values(["cancer", "year", "study_id"])
    primary = pd.DataFrame(primary_rows)
    primary = primary.sort_values(
        "target",
        key=lambda values: values.map({target: index for index, target in enumerate(TARGET_ORDER)}),
    )
    primary.to_csv(OUT / "ihc_meta_primary_v6.csv", index=False)

    detail = pd.concat(
        [frame.assign(target=target) for target, frame in primary_details.items()],
        ignore_index=True,
    )
    detail.to_csv(OUT / "ihc_meta_primary_inputs_v6.csv", index=False)

    subgroup_rows = []
    for target in TARGET_ORDER:
        target_data = data[data["target"] == target]
        all_data = target_data[["n_positive", "n_total", "logit_prev", "logit_var"]]
        if len(all_data) >= 3:
            result = random_effects(all_data)
            result.update({"target": target, "subgroup": "All thresholds"})
            subgroup_rows.append(result)
        for threshold, frame in target_data.groupby("threshold_class"):
            if len(frame) < 2:
                continue
            result = random_effects(frame)
            result.update({"target": target, "subgroup": threshold})
            subgroup_rows.append(result)
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(OUT / "ihc_meta_subgroups_v6.csv", index=False)

    loo_rows = []
    for target, frame in primary_details.items():
        if len(frame) < 4:
            continue
        for excluded in frame["study_id"].unique():
            subset = frame[frame["study_id"] != excluded]
            if len(subset) < 3:
                continue
            result = random_effects(subset)
            result.update({"target": target, "excluded_study": excluded})
            loo_rows.append(result)
    pd.DataFrame(loo_rows).to_csv(OUT / "ihc_meta_leave_one_study_out_v6.csv", index=False)

    non_meta = (
        data.groupby("target")
        .agg(
            cohorts=("study_id", "size"),
            studies=("study_id", "nunique"),
            total_n=("n_total", "sum"),
            observed_positive=("n_positive", "sum"),
        )
        .reset_index()
    )
    non_meta["observed_prevalence"] = non_meta["observed_positive"] / non_meta["total_n"]
    non_meta["meta_analysed"] = non_meta["target"].isin(primary["target"])
    non_meta.to_csv(OUT / "ihc_target_availability_v6.csv", index=False)

    print(primary.round(4).to_string(index=False))
    print("\nTargets without sufficient cohorts for the primary meta-analysis")
    print(non_meta[~non_meta["meta_analysed"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
