# Pan-gynecologic ADC target evidence framework

Code and processed results for: *A multilayer surfaceome framework for context-dependent antibody-drug conjugate target evidence in ovarian, endometrial, and cervical cancers: a computational study.*

All inputs are public, de-identified datasets. No patient-level controlled-access data are included.

## Layout
- `code/` analysis and figure scripts (run from the `code/` folder or repository root; paths are relative to the repository root)
- `results/` processed target-level tables reported in the manuscript
- `literature/` Europe PMC search output, screening decisions, extracted IHC counts
- `metadata/` data sources (Supplementary Table S3) and source URLs/access dates

## Environment
Python 3.12.
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data download
```bash
python code/00_download_external_data.py   # DepMap 24Q4 and PRISM 20Q2 (Figshare) -> data/
```
Download manually into `data/`: CSPA `table_S3_surfaceome.xlsx` (Bausch-Fluck et al. 2015), HPA v23 pathology and subcellular files, and TCGA/GTEx expression from UCSC Xena (`data/tcga/surfaceome_tpm.csv.gz`, `data/tcga/surfaceome_samples.csv`). Other sources and URLs are listed in `metadata/source_manifest_v6.tsv`.

Scripts 03 and 05 also read upstream surfaceome discovery tables (`inputs_v5/surfaceome_discovery_v5.csv`, `surfaceome_topology_annotations_v5.csv`, `surfaceome_discovery_topology_v5.csv`, `surfaceome_discovery_candidates_v5.csv`, `surfaceome_discovery_extracellular_candidates_v5.csv`) produced by an earlier pipeline stage; their outputs are provided in `results/`.

## Run order
```bash
python code/01_ihc_systematic_search.py
python code/02_ihc_meta_analysis.py
python code/12_ihc_quality_assessment.py
python code/03_surface_proteome_validation.py
python code/04_depmap_prism_functional_proxy.py
python code/05_surfaceome_nested_benchmark.py
cd code && python 15_weight_sensitivity.py && python 16_ihc_meta_reml_hksj_glmm.py && cd ..
python code/06_figure_ihc_meta.py
python code/07_figure_external_function.py
python code/08_figS7_ihc_forest.py
python code/09_figS8_nested_benchmark.py
python code/10_graphical_abstract_v6.py
```
Scripts 15 and 16 run from `results/` alone (weight sensitivity: seed 20261103; REML-HKSJ/GLMM meta-analysis).

## License
MIT (see LICENSE).
