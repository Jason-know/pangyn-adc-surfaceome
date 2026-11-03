from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
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
TARGETS = sorted(KNOWN | set(CANDIDATES))
LINEAGES = {
    "OV": "Ovary/Fallopian Tube",
    "UCEC": "Uterus",
    "CESC": "Cervix",
}
PAYLOAD_COMPOUNDS = {
    "TOP1_payload_proxy": ["SN-38", "exatecan-mesylate", "topotecan", "irinotecan"],
    "microtubule_payload_proxy": [
        "paclitaxel",
        "docetaxel",
        "vinblastine",
        "vincristine",
        "vinorelbine",
    ],
}


def gene_column(columns, gene):
    prefix = f"{gene} ("
    return next((column for column in columns if column.startswith(prefix)), None)


def read_selected_matrix(path, genes):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    selected = {"Unnamed: 0"}
    for gene in genes:
        column = gene_column(header, gene)
        if column:
            selected.add(column)
    frame = pd.read_csv(path, usecols=lambda column: column in selected)
    frame = frame.rename(columns={"Unnamed: 0": "ModelID"})
    rename = {}
    for gene in genes:
        column = gene_column(frame.columns, gene)
        if column:
            rename[column] = gene
    return frame.rename(columns=rename)


def depmap_analysis(model, expression, dependency):
    gynecologic = model[model["OncotreeLineage"].isin(LINEAGES.values())].copy()
    lineage_lookup = {lineage: cancer for cancer, lineage in LINEAGES.items()}
    gynecologic["cancer"] = gynecologic["OncotreeLineage"].map(lineage_lookup)
    merged = gynecologic[["ModelID", "CellLineName", "cancer"]].merge(
        expression,
        on="ModelID",
        how="inner",
    ).merge(dependency, on="ModelID", how="inner", suffixes=("_expr", "_dep"))
    availability = merged[["ModelID", "CellLineName", "cancer"]].copy()
    availability.to_csv(OUT / "depmap_gynecologic_models_v6.csv", index=False)

    rows = []
    for cancer, frame in merged.groupby("cancer"):
        for target in TARGETS:
            expression_column = f"{target}_expr"
            dependency_column = f"{target}_dep"
            if expression_column not in frame:
                continue
            expr = pd.to_numeric(frame[expression_column], errors="coerce")
            row = {
                "cancer": cancer,
                "target": target,
                "n_lines": int(expr.notna().sum()),
                "mean_expression": float(expr.mean()),
                "fraction_expressed": float((expr > 1).mean()),
                "known_adc_target": target in KNOWN,
                "non_panel_candidate": target in CANDIDATES,
            }
            if dependency_column in frame:
                dep = pd.to_numeric(frame[dependency_column], errors="coerce")
                valid = expr.notna() & dep.notna()
                row["median_gene_effect"] = float(dep[valid].median())
                row["fraction_dependent"] = float((dep[valid] <= -0.5).mean())
                if valid.sum() >= 10 and expr[valid].nunique() > 1 and dep[valid].nunique() > 1:
                    result = spearmanr(expr[valid], dep[valid])
                    row["expression_dependency_rho"] = float(result.statistic)
                    row["expression_dependency_p"] = float(result.pvalue)
            rows.append(row)
    output = pd.DataFrame(rows)
    for column in ["expression_dependency_p"]:
        if column in output:
            valid = output[column].notna()
            output.loc[valid, "expression_dependency_fdr"] = multipletests(
                output.loc[valid, column],
                method="fdr_bh",
            )[1]
    output.to_csv(OUT / "depmap_target_function_v6.csv", index=False)
    return merged, output


def prism_analysis(model, expression):
    prism_info = pd.read_csv(
        DATA
        / "prism_20q2"
        / "prism-repurposing-20q2-secondary-screen-cell-line-info.csv"
    )
    treatment = pd.read_csv(
        DATA
        / "prism_20q2"
        / "prism-repurposing-20q2-secondary-screen-replicate-collapsed-treatment-info.csv"
    )
    compound_names = {name for names in PAYLOAD_COMPOUNDS.values() for name in names}
    selected_treatment = treatment[
        treatment["name"].isin(compound_names)
        & treatment["moa"].fillna("").str.contains(
            "topoisomerase inhibitor|tubulin polymerization inhibitor|microtubule inhibitor",
            case=False,
            regex=True,
        )
    ].copy()
    selected_columns = set(selected_treatment["column_name"])
    prism_path = (
        DATA
        / "prism_20q2"
        / "prism-repurposing-20q2-secondary-screen-replicate-collapsed-logfold-change.csv"
    )
    prism = pd.read_csv(
        prism_path,
        usecols=lambda column: column == "Unnamed: 0" or column in selected_columns,
    ).rename(columns={"Unnamed: 0": "row_name"})
    prism["depmap_id"] = prism["row_name"].str.replace(r"^PR\d+_", "", regex=True)
    prism = prism.merge(prism_info[["depmap_id"]].drop_duplicates(), on="depmap_id", how="inner")

    model_lookup = model[["ModelID", "OncotreeLineage"]].copy()
    model_lookup["cancer"] = model_lookup["OncotreeLineage"].map(
        {lineage: cancer for cancer, lineage in LINEAGES.items()}
    )
    model_lookup = model_lookup.dropna(subset=["cancer"]).rename(columns={"ModelID": "depmap_id"})
    prism = prism.merge(model_lookup[["depmap_id", "cancer"]], on="depmap_id", how="inner")

    line_rows = prism[["depmap_id", "cancer"]].copy()
    for payload_class, compounds in PAYLOAD_COMPOUNDS.items():
        columns = selected_treatment.loc[
            selected_treatment["name"].isin(compounds),
            "column_name",
        ].tolist()
        columns = list(
            dict.fromkeys(column for column in columns if column in prism.columns)
        )
        if not columns:
            continue
        response = -prism[columns].apply(pd.to_numeric, errors="coerce")
        for column in columns:
            values = response[column]
            response[column] = (
                (values - values.mean()) / values.std(ddof=0)
                if values.notna().sum() >= 5 and values.std(ddof=0) > 0
                else np.nan
            )
        line_rows[payload_class] = response.mean(axis=1, skipna=True)

    expression = expression.rename(columns={"ModelID": "depmap_id"})
    line_rows = line_rows.merge(expression, on="depmap_id", how="inner")
    line_rows.to_csv(OUT / "prism_gynecologic_payload_scores_v6.csv", index=False)

    rows = []
    for cancer, frame in line_rows.groupby("cancer"):
        for payload_class in PAYLOAD_COMPOUNDS:
            if payload_class not in frame:
                continue
            response = pd.to_numeric(frame[payload_class], errors="coerce")
            for target in TARGETS:
                if target not in frame:
                    continue
                expr = pd.to_numeric(frame[target], errors="coerce")
                valid = expr.notna() & response.notna()
                if valid.sum() < 15 or expr[valid].nunique() < 2 or response[valid].nunique() < 2:
                    continue
                result = spearmanr(expr[valid], response[valid])
                rows.append(
                    {
                        "cancer": cancer,
                        "target": target,
                        "payload_class": payload_class,
                        "n_lines": int(valid.sum()),
                        "spearman_rho": float(result.statistic),
                        "p": float(result.pvalue),
                        "known_adc_target": target in KNOWN,
                        "non_panel_candidate": target in CANDIDATES,
                    }
                )
    output = pd.DataFrame(rows)
    output["fdr_bh"] = multipletests(output["p"], method="fdr_bh")[1]
    output.to_csv(OUT / "target_payload_functional_proxy_v6.csv", index=False)
    return output


def main():
    model = pd.read_csv(DATA / "depmap_24q4" / "Model.csv")
    expression = read_selected_matrix(
        DATA / "depmap_24q4" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        TARGETS,
    )
    dependency = read_selected_matrix(
        DATA / "depmap_24q4" / "CRISPRGeneEffect.csv",
        TARGETS,
    )
    depmap_merged, depmap_results = depmap_analysis(model, expression, dependency)
    payload_results = prism_analysis(model, expression)

    print(
        depmap_results[
            depmap_results["known_adc_target"]
            | depmap_results["non_panel_candidate"]
        ]
        .sort_values(["cancer", "known_adc_target", "mean_expression"], ascending=[True, False, False])
        .round(3)
        .to_string(index=False)
    )
    print("\nPayload-class proxy associations")
    print(payload_results.sort_values("p").head(30).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
