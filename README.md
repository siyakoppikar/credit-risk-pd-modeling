# Credit Risk: 1-Year Probability of Default (PD) Modeling

A machine learning pipeline that estimates the **1-year probability of default (PD)** for corporate borrowers using historical financial-statement data, built for a simulated bank underwriting use case ("Banca Massiccia").

## Overview

Traditional credit scoring can miss non-linear relationships between a firm's financials, industry, and default risk. This project builds a data-driven PD model to support risk-based loan pricing, capital allocation, and underwriting decisions using firm-level financial statement data for a panel of Italian companies.

**Key results (2012 out-of-sample holdout):**
| Metric | Score |
|---|---|
| AUC | 0.917 |
| KS Statistic | 0.700 |
| Brier Score | 0.00196 |

## Approach

1. **Target construction** — Built a forward-looking `default_flag` using a realistic reporting lag (financials assumed available ~4 months after fiscal year-end), predicting default within the following 12 months.
2. **Data cleaning & imputation** — Filled key accounting fields (revenue, COGS, operating profit, cash, ROE) using identity relationships between related financial statement lines wherever possible.
3. **Feature engineering** — Constructed ratios across five categories:
   - **Liquidity**: current ratio, cash ratio
   - **Leverage**: debt-to-equity, debt ratio, equity ratio
   - **Profitability**: gross margin, operating margin, ROA, ROE
   - **Efficiency**: asset turnover
   - **Coverage**: asset coverage ratio
   - Missing-value indicator flags for each ratio, plus firm size (log total assets)
4. **Non-parametric transformation** — Fit quantile-based PD curves per ratio on the training set, converting raw ratios into monotonic, PD-scaled features to capture non-linear risk relationships.
5. **Modeling** — Logistic regression (L2-regularized), hyperparameters tuned via k-fold cross-validation.
6. **Validation** — Walk-forward (expanding window) evaluation across fiscal years 2008–2012, with a held-out final test year to confirm generalization.

## Repository Structure

```
├── code_base.ipynb       # Full exploratory notebook: EDA, feature engineering, model training & evaluation
├── estimation.py         # Production training pipeline (cleaning, ratios, transforms, model fit + save)
├── prediction.py         # Loads saved model artifacts and scores new data
├── harness.py            # CLI entry point: input CSV -> PD predictions CSV
├── sample_data.csv        # Small sample dataset (same schema as training data) for testing the pipeline
├── requirements.txt       # Pinned dependencies
└── docs/
    └── Team_Turquoise_Pitch_Deck.pdf
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train the model** (generates `pd_model.joblib`):
```bash
python3 estimation.py <path_to_training_csv> pd_model.joblib
```

**Score new data:**
```bash
python3 harness.py --input_csv sample_data.csv --output_csv pd_predictions.csv
```

The output is a single-column CSV of predicted probabilities of default for each firm-year record in the input.

## Data

Data consists of annual financial-statement records for a panel of non-financial, non-insurance Italian firms, each with more than €1.5M in total assets. Each row represents one firm-year observation with 44 fields covering balance sheet items, income statement items, and derived financial ratios. The full training dataset is proprietary and not included in this repository; `sample_data.csv` is provided to demonstrate the expected input schema and to test the pipeline end-to-end.

## My Contribution

This was completed as a 3-person graduate coursework project, built through close, hands-on collaboration — I was actively involved across the full pipeline: data cleaning, feature engineering, model coding, and evaluation. My primary area of ownership was **evaluation and the production harness**: I built `harness.py` to automate end-to-end scoring (cleaning → ratio construction → transforms → PD prediction), validated model performance (AUC, KS, Brier score) across the walk-forward evaluation windows, and compiled the results and references documented here. I also worked directly on data preparation, ratio construction, and model logic alongside my teammates.

## Notes on Academic Context

This project was completed as part of a graduate course under the guidance of Professor Roger M. Stein. It was submitted as part of a class competition, where our team placed **third**. Course materials, lecture notes, and *Active Credit Portfolio Management in Practice* by Roger M. Stein and Jeffrey R. Bohn informed the modeling approach. General programming references (Python docs, Stack Overflow, AI-assisted tools) were used for implementation support. No external datasets were used beyond what was provided for the course.
