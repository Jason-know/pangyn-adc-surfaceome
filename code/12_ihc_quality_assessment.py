from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "literature"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    studies = pd.read_csv(LIT / "ihc_study_characteristics.csv")
    estimates = pd.read_csv(LIT / "ihc_prevalence_estimates.csv")
    summary = (
        estimates.groupby("study_id")
        .agg(max_n=("n_total", "max"), cohorts=("target", "size"), targets=("target", "nunique"))
        .reset_index()
        .merge(studies, on="study_id", how="right")
    )

    rows = []
    for row in summary.itertuples():
        selected = any(
            term in str(row.cohort_role).lower()
            for term in ["clinical", "recurrent", "advanced", "consortium", "prospective"]
        )
        representative = "Some concerns" if selected else "Low"
        sample_size = (
            "Low"
            if row.max_n >= 100
            else "Some concerns"
            if row.max_n >= 50
            else "High"
        )
        threshold = "Some concerns" if str(row.cohort_role) else "Unclear"
        outcome = "Some concerns"
        analysis = "Low"
        if sample_size == "High" or selected:
            overall = "High"
        elif sample_size == "Some concerns" or selected:
            overall = "Some concerns"
        else:
            overall = "Some concerns"
        rows.append(
            {
                "study_id": row.study_id,
                "first_author": row.first_author,
                "year": row.year,
                "cohort_role": row.cohort_role,
                "participant_selection": representative,
                "sample_size": sample_size,
                "assay_reporting": threshold,
                "outcome_definition": outcome,
                "prevalence_analysis": analysis,
                "overall_concern": overall,
                "assessment_note": (
                    "Not a validated QUIPS score; conservative structured appraisal "
                    "from the curation record and available methods text."
                ),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "ihc_risk_of_bias_v6.csv", index=False)
    print(output["overall_concern"].value_counts().to_string())


if __name__ == "__main__":
    main()
