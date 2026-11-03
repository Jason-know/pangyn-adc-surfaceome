from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FILES = {
    DATA / "depmap_24q4" / "Model.csv": (
        "https://ndownloader.figshare.com/files/51065297",
        645696,
    ),
    DATA / "depmap_24q4" / "CRISPRGeneEffect.csv": (
        "https://ndownloader.figshare.com/files/51064667",
        428678699,
    ),
    DATA / "depmap_24q4" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv": (
        "https://ndownloader.figshare.com/files/51065489",
        506628654,
    ),
    DATA / "prism_20q2" / "prism-repurposing-20q2-secondary-screen-cell-line-info.csv": (
        "https://ndownloader.figshare.com/files/36794589",
        84730,
    ),
    DATA / "prism_20q2" / "prism-repurposing-20q2-secondary-screen-replicate-collapsed-treatment-info.csv": (
        "https://ndownloader.figshare.com/files/36794619",
        4284816,
    ),
    DATA / "prism_20q2" / "prism-repurposing-20q2-secondary-screen-replicate-collapsed-logfold-change.csv": (
        "https://ndownloader.figshare.com/files/36794616",
        114666971,
    ),
}


def download(url, destination, expected_size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        print("cached", destination.name, destination.stat().st_size)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    existing = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=180) as response:
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with temporary.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    if destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {destination.name}: "
            f"{destination.stat().st_size} vs {expected_size}"
        )
    print("downloaded", destination.name, destination.stat().st_size)


def main():
    for destination, (url, expected_size) in FILES.items():
        download(url, destination, expected_size)


if __name__ == "__main__":
    main()
