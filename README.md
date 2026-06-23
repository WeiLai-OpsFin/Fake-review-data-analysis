# Fake Review Analytics: Detection, Trust, and Hotel Booking Decisions

This repository is a curated research portfolio on fake online reviews. It combines two complementary angles:

1. **NLP / machine-learning detection**: cleaning review text, building linguistic features, and preparing a fake-review classification workflow.
2. **Consumer-behavior analytics**: testing how fake reviews influence perceived authenticity, trust, perceived risk, and hotel booking intention using a questionnaire experiment, DID-style comparison, and PLS-SEM-style path analysis.

The repository emphasizes reproducibility, clear documentation, compact de-identified outputs, and interpretable English result tables.

## Research Questions

- Can fake reviews be identified through text preprocessing and classification features?
- Do fake reviews lower perceived authenticity and trust in hotel reviews?
- Which psychological mechanism matters most for booking intention: authenticity, trust, or perceived risk?
- How can business analytics combine text mining with causal-style and structural modeling?

## Headline Findings

- The final questionnaire sample contains **68 valid responses**: 32 in the real-review condition and 36 in the fake-review condition.
- Real reviews scored higher on perceived authenticity than fake reviews: mean difference = **0.746**, p = **0.0000**, Cohen's d = **1.08**.
- Real reviews also scored higher on trust: mean difference = **0.497**, p = **0.0021**.
- The DID-style estimate for booking intention is **-0.130** with p = **0.412**, so it is treated as exploratory rather than confirmatory evidence.
- The PLS-SEM-style model explains **66.7%** of the variance in booking intention.
- The strongest mechanism is authenticity -> trust -> booking intention: the indirect effect from review type through authenticity and trust is **0.166**, p = **0.0037**.

## Methods

- Text cleaning and normalization for online reviews
- NLP feature engineering and TF-IDF modeling
- Classification-ready fake-review detection pipeline
- Questionnaire data cleaning and screening
- DID-style comparison of pre/post booking intention
- PLS-SEM-style measurement and structural path analysis
- Bootstrap inference for structural paths and mediation effects

## Repository Structure

| Path | Purpose |
|---|---|
| `src/` | Reproducible Python scripts for text classification and survey modeling |
| `data/` | Compact, reviewable data artifacts without large raw archives |
| `results/` | Clean CSV outputs for DID, SEM, reliability, mediation, and IPMA results |
| `docs/` | Methodology notes and variable dictionary |
| `requirements.txt` | Python environment used for the analysis |

## Reproducibility

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the text pipeline on any review dataset with a text column and label column:

```bash
python src/fake_review_text_pipeline.py --input data/review_examples_20.csv --text-col review_text --label-col truth_label
```

Reproduce the SEM-style path model from the checked-in construct scores:

```bash
python src/sem_path_model_from_scores.py --scores-file data/pls_sem_construct_scores.csv --outdir results/reproduced
```

Run the full questionnaire analysis after placing the real-review and fake-review questionnaire exports in a local data folder:

```bash
python src/hotel_review_did_sem_analysis.py --real-file path/to/real_reviews.xlsx --fake-file path/to/fake_reviews.xlsx --outdir results
```

The checked-in `results/` files are the curated outputs from the final analysis.

## Curation Note

This portfolio intentionally excludes raw large archives, duplicated working folders, journal author guidelines, downloaded reference papers, chat logs, similarity / AI reports, and notebooks containing hard-coded external-service credentials. Those materials are useful during project work but not appropriate for a public admissions-facing repository.

## Skills Demonstrated

- Python data analysis with pandas, scipy, statsmodels, and scikit-learn
- Natural language preprocessing and fake-review classification design
- Survey experiment design and questionnaire data cleaning
- DID-style comparison and structural path modeling
- PLS-SEM interpretation: reliability, validity, mediation, R2, VIF, and IPMA
- Research communication for digital trust, platform governance, and consumer analytics

## Author

Lai Wei  
Research portfolio project on fake reviews, online trust, and hotel booking decisions
