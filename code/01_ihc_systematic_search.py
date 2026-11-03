import html
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SEARCH_DATE = date(2026, 11, 2).isoformat()

TARGETS = (
    '("FOLR1" OR "folate receptor alpha" OR TACSTD2 OR TROP2 OR NECTIN4 '
    'OR ERBB2 OR HER2 OR MSLN OR mesothelin OR VTCN1 OR "B7-H4" '
    'OR CLDN6 OR "claudin-6" OR F3 OR "tissue factor" OR EPCAM '
    'OR MUC16 OR CA125 OR CDH6 OR "cadherin-6" OR SLC34A2 OR NaPi2b)'
)
CANCERS = (
    '("ovarian cancer" OR "ovarian carcinoma" OR "endometrial cancer" '
    'OR "uterine cancer" OR "cervical cancer" OR "gynecologic cancer" '
    'OR "gynaecologic cancer")'
)
METHOD = (
    '(immunohistochemistry OR immunohistochemical OR IHC '
    'OR "tissue microarray" OR immunostaining)'
)
QUERY = (
    f"TITLE_ABS:{TARGETS} AND TITLE_ABS:{CANCERS} AND TITLE_ABS:{METHOD} "
    "AND (FIRST_PDATE:[1990-01-01 TO 2026-11-02])"
)


def normalize(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def fetch():
    response = requests.get(
        BASE,
        params={
            "query": QUERY,
            "format": "json",
            "pageSize": 1000,
            "resultType": "core",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("resultList", {}).get("result", [])
    output = {
        "search_date": SEARCH_DATE,
        "query": QUERY,
        "reported_hits": payload.get("hitCount"),
        "records_retrieved": len(records),
        "records": records,
    }
    (OUT / "europepmc_search_20261102.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return output


def screen(payload):
    rows = []
    exclusion_pattern = re.compile(
        r"\b(review|systematic review|meta-analysis|case report|editorial|comment|"
        r"protocol|guideline|correction)\b",
        re.IGNORECASE,
    )
    for record in payload["records"]:
        title = normalize(record.get("title", ""))
        abstract = normalize(record.get("abstractText", ""))
        text = f"{title} {abstract}"
        has_cancer = bool(
            re.search(
                r"ovar|endometr|uterine|cervic|gynecolog|gynaecolog|fallopian",
                text,
                re.IGNORECASE,
            )
        )
        has_ihc = bool(
            re.search(
                r"immunohist|IHC|tissue microarray|immunostain",
                text,
                re.IGNORECASE,
            )
        )
        has_target = bool(
            re.search(
                r"FOLR1|folate receptor|TACSTD2|TROP2|NECTIN4|ERBB2|HER2|"
                r"MSLN|mesothelin|VTCN1|B7-H4|CLDN6|claudin-6|tissue factor|"
                r"\bF3\b|EPCAM|MUC16|CA125|CDH6|cadherin-6|SLC34A2|NaPi2b",
                text,
                re.IGNORECASE,
            )
        )
        has_quantitative_signal = bool(
            re.search(r"\d+(?:\.\d+)?\s*%|\b\d+\s*(?:/|of)\s*\d+\b", abstract)
        )
        review_like = bool(exclusion_pattern.search(title))
        basic_pass = (
            has_cancer
            and has_ihc
            and has_target
            and has_quantitative_signal
            and not review_like
        )
        rows.append(
            {
                "pmid": record.get("pmid"),
                "pmcid": record.get("pmcid"),
                "doi": record.get("doi"),
                "year": record.get("pubYear"),
                "title": title,
                "journal": (
                    (record.get("journalInfo", {}) or {}).get("journal", {}) or {}
                ).get("title"),
                "has_cancer": has_cancer,
                "has_ihc": has_ihc,
                "has_target": has_target,
                "has_quantitative_signal": has_quantitative_signal,
                "review_like": review_like,
                "basic_screen_pass": basic_pass,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "title_abstract_screening_20261102.csv", index=False)
    return frame


def main():
    payload = fetch()
    screened = screen(payload)
    included = screened[screened["basic_screen_pass"]]
    estimates_path = OUT / "ihc_prevalence_estimates.csv"
    studies_path = OUT / "ihc_study_characteristics.csv"
    if estimates_path.exists() and studies_path.exists():
        estimates = pd.read_csv(estimates_path)
        studies = pd.read_csv(studies_path)
        included_pmids = set(studies["pmid"].dropna().astype(int).astype(str))
        screened["manual_inclusion"] = screened["pmid"].fillna("").astype(str).isin(included_pmids)
        screened.to_csv(OUT / "title_abstract_screening_20261102.csv", index=False)
        included_reports = int(studies["study_id"].nunique())
        included_cohorts = len(estimates)
    else:
        included_reports = 0
        included_cohorts = 0
    summary = pd.DataFrame(
        [
            {
                "stage": "Records identified",
                "count": payload.get("reported_hits", len(screened)),
            },
            {"stage": "Records retrieved", "count": len(screened)},
            {
                "stage": "Records excluded at title and abstract screening",
                "count": len(screened) - len(included),
            },
            {
                "stage": "Records with target, cancer, IHC, and quantitative signal",
                "count": len(included),
            },
            {
                "stage": "Reports included after structured-data extraction",
                "count": included_reports,
            },
            {
                "stage": "Target-cancer cohorts included",
                "count": included_cohorts,
            },
            {
                "stage": "Records excluded because direct prevalence or denominator was unavailable",
                "count": len(included) - included_reports,
            },
        ]
    )
    summary.to_csv(OUT / "prisma_search_summary_20261102.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
