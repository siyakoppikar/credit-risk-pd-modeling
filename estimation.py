# estimation.py
import os
import numpy as np
import pandas as pd
import joblib

from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def print_header(msg: str):
    print("\n")
    print(f"📌 {msg}")


# -------------------------------------------------------------------
# 1. CONFIG – ratios and flags (same as in your notebook)
# -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pd_model.joblib")

FINAL_RAW_RATIOS = [
    "current_ratio_raw",
    "cash_ratio_raw",
    "debt_ratio_raw",
    "operating_margin_raw",
    "roa_raw",
    "roe_raw",
    "asst_turnover_ratio_raw",
    "asst_coverage_ratio_raw",
]

FINAL_MISSING_FLAGS = [c + "_missing" for c in FINAL_RAW_RATIOS]


# -------------------------------------------------------------------
# 2. DATA PREP HELPERS – copy your inline code into these functions
# -------------------------------------------------------------------

def add_default_flag(df: pd.DataFrame) -> pd.DataFrame:
    # print_header("Creating default_flag...")
    # 1. Ensure dates are in datetime format
    # Make sure dates are proper datetimes
    if "stmt_date" in df.columns:
        df["stmt_date"] = pd.to_datetime(df["stmt_date"], errors="coerce")
    if "def_date" in df.columns:
        df["def_date"] = pd.to_datetime(df["def_date"], errors="coerce")   # will be NaT for non-defaulters

    # 2. Choose a reporting lag (here: 4 months ≈ "April/May", following the book)
    reporting_lag = pd.DateOffset(months=4)

    # 3. Compute the date when the financials are realistically available
    df['info_date'] = df['stmt_date'] + reporting_lag

    # 4. Given information as of info_year, will the firm default in the next 12 months?
    df['info_year'] = df['info_date'].dt.year

    # 5. Define the end of the 1-year prediction horizon
    df['horizon_end'] = df['info_date'] + pd.DateOffset(years=1)

    # 6. Create the default flag:
    #    1 if default occurs between info_date (inclusive) and horizon_end (exclusive), else 0.
    df['default_flag'] = np.where(
        df['def_date'].notna()
        & (df['def_date'] >= df['info_date'])
        & (df['def_date'] < df['horizon_end']),
        1,
        0
    ).astype(int)

    return df


def clean_and_impute_base_vars(df: pd.DataFrame) -> pd.DataFrame:
    # print_header("Cleaning & imputing core accounting variables...")
    # Drop col:eqty_corp_family_tot
    if 'eqty_corp_family_tot' in df.columns:
        df.drop(columns=['eqty_corp_family_tot'], inplace=True)


    cols_to_check = [
        "rev_operating",
        "COGS",
        "prof_operations",
        "cash_and_equiv",
        "roe"
    ]

    # Store missing counts BEFORE imputation
    missing_before = df[cols_to_check].isna().sum()

    # 1. Fill missing rev_operating
    mask_rev_missing = (
        df["rev_operating"].isna() &
        df["COGS"].notna() &
        df["prof_operations"].notna()
    )

    df.loc[mask_rev_missing, "rev_operating"] = (
        df.loc[mask_rev_missing, "COGS"] + df.loc[mask_rev_missing, "prof_operations"]
    )


    # 2. Fill missing COGS
    mask_cogs_missing = (
        df["COGS"].isna() &
        df["rev_operating"].notna() &
        df["prof_operations"].notna()
    )

    df.loc[mask_cogs_missing, "COGS"] = (
        df.loc[mask_cogs_missing, "rev_operating"] - df.loc[mask_cogs_missing, "prof_operations"]
    )


    # 3. Fill missing prof_operations
    mask_prof_missing = (
        df["prof_operations"].isna() &
        df["rev_operating"].notna() &
        df["COGS"].notna()
    )

    df.loc[mask_prof_missing, "prof_operations"] = (
        df.loc[mask_prof_missing, "rev_operating"] - df.loc[mask_prof_missing, "COGS"]
    )


    # Fill missing cash_and_equiv = asst_current - AR (only where both are available)
    mask = df["cash_and_equiv"].isna() & df["asst_current"].notna() & df["AR"].notna()

    df.loc[mask, "cash_and_equiv"] = df.loc[mask, "asst_current"] - df.loc[mask, "AR"]


    mask_roe = (
        df["roe"].isna() &                            # Only if ROE is missing
        df["profit"].notna() &                        # Profit available
        df["eqty_tot"].notna() &                      # Equity available
        ~((df["profit"] < 0) & (df["eqty_tot"] < 0))  # Exclude both negative
    )

    # Calculate ROE = (profit / eqty_tot) * 100 only for valid cases
    df.loc[mask_roe, "roe"] = (
        (df.loc[mask_roe, "profit"] / df.loc[mask_roe, "eqty_tot"]) * 100
    ).round(2)

    # Store missing counts AFTER imputation
    missing_after = df[cols_to_check].isna().sum()

    # Compare
    # print("\nMissing values before imputation:")
    # print(missing_before)

    # print("\nMissing values after imputation:")
    # print(missing_after)
    
    return df


def build_ratios_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    # print_header("Building raw ratios & missing-value flags...")
    # LEVERAGE RATIOS
    # Total Liabilities = Total Assets - Total Equity
    df["liab_tot"] = df["asst_tot"] - df["eqty_tot"]

    # Debt-to-Equity Ratio RAW
    df["debt_to_equity_raw"] = df["liab_tot"] / df["eqty_tot"]

    # Debt Ratio RAW
    df["debt_ratio_raw"] = df["liab_tot"] / df["asst_tot"]

    # Equity Ratio Raw
    df["equity_ratio_raw"] = df["eqty_tot"] / df["asst_tot"]

    ratio_cols_lev = [
        "debt_to_equity_raw",
        "debt_ratio_raw",
        "equity_ratio_raw"
    ]

    # print("\nMissing Values:")
    # print(df[ratio_cols_lev].isna().sum())
    # print("\n")


    # PROFITBILITY RATIOS
        # Gross Margin = (rev_operating - COGS) / rev_operating
    df["gross_margin_raw"] = (df["rev_operating"] - df["COGS"]) / df["rev_operating"]

    # Operating Margin = prof_operations / rev_operating
    df["operating_margin_raw"] = df["prof_operations"] / df["rev_operating"]

    # Return on Assets (ROA) = prof_operations / asst_tot
    df["roa_raw"] = df["prof_operations"] / df["asst_tot"]

    # Return on Equity (ROE)
    df.rename(columns={"roe": "roe_raw"}, inplace=True)

    # (Optional) Check NaNs in ratios
    ratio_cols_prf = [
        "gross_margin_raw",
        "operating_margin_raw",
        "roa_raw",
        "roe_raw"
    ]

    # print("Missing Values:")
    # print(df[ratio_cols_prf].isna().sum())
    # print("\n")


    # EFFICIENCY RATIOS
    # Compute Asset Turnover ratio
    df['asst_turnover_ratio_raw'] = df['rev_operating'] / df['asst_tot']

    # (Optional) Check NaNs in ratios
    ratio_cols_efy = [
        "asst_turnover_ratio_raw"
    ]

    # print("Missing Values:")
    # print(df[ratio_cols_efy].isna().sum())
    # print("\n")


    # LIQUIDITY RATIOS
    df["current_ratio_raw"] = df["asst_current"] / df["debt_st"]

    # Cash Ratio
    df["cash_ratio_raw"] = df["cash_and_equiv"] / df["debt_st"]

    # (Optional) Check NaNs in ratios
    ratio_cols_liq = [
        "current_ratio_raw",
        "cash_ratio_raw",
    ]

    # print("Missing Values:")
    # print(df[ratio_cols_liq].isna().sum())
    # print("\n")


    # COVERAGE RATIOS
    # Compute Asset Coverage Ratio
    df["asst_coverage_ratio_raw"] = (
        (df["asst_tot"] - df["asst_intang_fixed"] - df["debt_st"]) /
        (df["debt_st"] + df["debt_lt"])
    )

    # (Optional) Check NaNs in ratios
    ratio_cols = [
        "asst_coverage_ratio_raw",
    ]

    # print("Missing Values:")
    # print(df[ratio_cols].isna().sum())
    # print("\n")
    
    # Size = Total Assets
    df['log_asst_tot'] = np.log1p(df['asst_tot'].clip(lower=0))


    # MISSING FLAGS
    # Identify Ratio Columns
    raw_ratio_cols = [col for col in df.columns if col.endswith("_raw")]

    for col in raw_ratio_cols:
        df[col + "_missing"] = df[col].isna().astype(int)


    missing_flag_cols = [col for col in df.columns if col.endswith("_missing")]

    # print("Raw ratios:")
    # for col in raw_ratio_cols:
    #     print(" -", col)

    # print("\nMissing flags:")
    # for col in missing_flag_cols:
    #     print(" -", col)

    return df


def prepare_full_training_df(df: pd.DataFrame) -> pd.DataFrame:
    # Convenience wrapper for training: add default_flag + clean + ratios.
    # print_header("Preparing full training dataframe (default flag + cleaning + ratios)...")
    df = add_default_flag(df)
    df = clean_and_impute_base_vars(df)
    df = build_ratios_and_flags(df)
    return df


# -------------------------------------------------------------------
# 3. NONPARAMETRIC TRANSFORM FUNCTIONS – directly from your notebook
# -------------------------------------------------------------------

def fit_nonparametric_transform(df, x_col, y_col="default_flag", k=100):
    # Fit a nonparametric transform for one variable:
    # Returns a dict with: "edges", "quantiles" and "pd_per_bin"
        
    data = df[[x_col, y_col]].dropna().copy()
    x = data[x_col].values
    y = data[y_col].astype(float).values

    # Quantile positions (0..1)
    quantiles = np.linspace(0.0, 1.0, k + 1)

    # Quantile edges of x
    edges = np.quantile(x, quantiles)

    # Remove duplicate edges (can happen if many identical x values)
    edges, unique_idx = np.unique(edges, return_index=True)
    quantiles = quantiles[unique_idx]

    if len(edges) < 2:
        raise ValueError(f"Not enough unique values for transform of {x_col}")

    # Assign each observation to a bin based on edges
    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    data["bin"] = bin_idx

    # Default rate per bin; ensure we have one value for every bin
    grouped = data.groupby("bin")[y_col].mean()
    grouped = grouped.reindex(range(len(edges) - 1))
    # If any bin is empty, fill from neighbours
    grouped = grouped.fillna(method="ffill").fillna(method="bfill")
    pd_per_bin = grouped.values  # shape: (num_bins,)

    return {
        "edges": edges,          # bin boundaries in x
        "quantiles": quantiles,  # quantile positions for edges
        "pd_per_bin": pd_per_bin # default rate for each bin (between edges[i], edges[i+1])
    }


def apply_nonparametric_transform(x_series, transform):
    # Apply the fitted nonparametric transform with interpolation.

    edges = transform["edges"]
    quantiles = transform["quantiles"]
    pd_bins = transform["pd_per_bin"]

    x = x_series.values
    result = np.full_like(x, np.nan, dtype=float)

    # Mask for non-missing x
    non_missing = ~np.isnan(x)
    if not np.any(non_missing):
        # All missing: just return all-NaN series
        return pd.Series(result, index=x_series.index)

    x_nm = x[non_missing]

    # Find bins for non-missing x
    bin_idx = np.searchsorted(edges, x_nm, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    b_i = edges[bin_idx]
    b_ip1 = edges[bin_idx + 1]
    q_i = quantiles[bin_idx]
    q_ip1 = quantiles[bin_idx + 1]

    # Interpolation factor inside bin
    denom = (b_ip1 - b_i)
    t = np.zeros_like(x_nm, dtype=float)
    mask = denom > 0
    t[mask] = (x_nm[mask] - b_i[mask]) / denom[mask]
    t = np.clip(t, 0.0, 1.0)

    # Interpolated quantile position q*
    q_star = q_i + t * (q_ip1 - q_i)

    # PD is defined per bin; put them at mid-quantiles for interpolation
    q_mid = (quantiles[:-1] + quantiles[1:]) / 2.0  # one per bin, same length as pd_bins

    pd_vals_nm = np.interp(
        q_star,
        q_mid,
        pd_bins,
        left=pd_bins[0],
        right=pd_bins[-1],
    )

    # Fill result only for non-missing entries
    result[non_missing] = pd_vals_nm

    return pd.Series(result, index=x_series.index)
    # ...


def fit_all_transforms(train_df, raw_ratio_cols, y_col="default_flag", k=100):
    # Fit nonparametric transforms for all variables in raw_ratio_cols on the TRAIN data.
    # print_header("Fitting ALL nonparametric transforms for raw ratios...")
    transforms = {}
    for col in raw_ratio_cols:
        transforms[col] = fit_nonparametric_transform(
            train_df,
            x_col=col,
            y_col=y_col,
            k=k,
        )

    return transforms


def transform_train(train_df, raw_ratio_cols, transforms, suffix="_tr"):
    # Apply transforms to TRAIN data and compute mean T(x) per variable.
    # print_header("Applying transforms to TRAIN data...")
    df_tr = deepcopy(train_df)
    mean_tr = {}

    for col in raw_ratio_cols:
        tr_col = col.replace("_raw", "") + suffix
        # Transform (NaNs preserved)
        df_tr[tr_col] = apply_nonparametric_transform(df_tr[col], transforms[col])

        # Mean of non-missing transformed values T(x)
        mean_val = df_tr[tr_col].mean(skipna=True)
        mean_tr[col] = mean_val

        # Impute transformed NaNs with this mean
        df_tr[tr_col] = df_tr[tr_col].fillna(mean_val)

    return df_tr, mean_tr


def transform_test(test_df, raw_ratio_cols, transforms, mean_tr, suffix="_tr"):
    # Apply transforms to TEST (or validation) data.
    # print_header("Applying transforms to TEST data...")
    df_tr = deepcopy(test_df)

    for col in raw_ratio_cols:
        tr_col = col.replace("_raw", "") + suffix
        df_tr[tr_col] = apply_nonparametric_transform(df_tr[col], transforms[col])

        # Impute with TRAIN mean T(x)
        df_tr[tr_col] = df_tr[tr_col].fillna(mean_tr[col])

    return df_tr


# -------------------------------------------------------------------
# 4. MODELING HELPERS – also from notebook
# -------------------------------------------------------------------

def ks_statistic(y_true, y_pred):
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic


def perform_kfold_lr(train_df, feature_cols, target_col="default_flag"):
    # print_header("Running K-fold cross-validation...")
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    lr = LogisticRegression(max_iter=5000, solver="lbfgs")
    param_grid = {"C": [0.01, 0.1, 1, 10]}

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(lr, param_grid, cv=kfold, scoring="roc_auc")
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


def train_final_lr(train_df, best_params, feature_cols):
    # print_header("Training final logistic regression model with chosen hyperparameters...")
    X_train = train_df[feature_cols].values
    y_train = train_df["default_flag"].values

    lr = LogisticRegression(max_iter=5000, solver="lbfgs", **best_params)
    lr.fit(X_train, y_train)
    return lr


def evaluate_test_year(model, test_df, feature_cols):
    # print_header("Evaluating model on test-year slice...")
    X_test = test_df[feature_cols].values
    y_test = test_df["default_flag"].values

    preds = model.predict_proba(X_test)[:, 1]

    return {
        "preds": preds,
        "auc": roc_auc_score(y_test, preds),
        "ks": ks_statistic(y_test, preds),
        "brier": brier_score_loss(y_test, preds)
    }


# -------------------------------------------------------------------
# 5. WALK-FORWARD ANALYSIS (for *your* evaluation only)
# -------------------------------------------------------------------

def run_walk_forward(df: pd.DataFrame):
    # This is for analysis, not the harness.
    # print_header("Running Walk-Forward Analysis (WFA)...")
    
    # Defining windows for analysis
    walk_windows = [
        {"train_years": [2007], "test_year": 2008},
        {"train_years": [2007, 2008], "test_year": 2009},
        {"train_years": [2007, 2008, 2009], "test_year": 2010},
        {"train_years": [2007, 2008, 2009, 2010], "test_year": 2011},
        {"train_years": [2007, 2008, 2009, 2010, 2011], "test_year": 2012},
    ]

    results = []
    all_preds = []

    for window in walk_windows:

        train_df = df[df["fs_year"].isin(window["train_years"])].copy()
        test_df  = df[df["fs_year"] == window["test_year"]].copy()

        # print(f"\n===== Train {window['train_years']} → Test {window['test_year']} =====")

        # 1. Fit T(x) on TRAIN
        transforms = fit_all_transforms(train_df, FINAL_RAW_RATIOS)

        # 2. Apply transforms
        train_tr, mean_tr = transform_train(train_df, FINAL_RAW_RATIOS, transforms)
        test_tr = transform_test(test_df, FINAL_RAW_RATIOS, transforms, mean_tr)

        # 3. Features
        transformed_cols = [c.replace("_raw", "") + "_tr" for c in FINAL_RAW_RATIOS]
        feature_cols = transformed_cols + FINAL_MISSING_FLAGS + ["log_asst_tot"]

        # 4. K-fold hyperparameter tuning
        best_model, best_params = perform_kfold_lr(train_tr, feature_cols)
        # print("Best params:", best_params)

        # 5. Train final model
        final_model = train_final_lr(train_tr, best_params, feature_cols)

        # 6. Evaluate on test window
        eval_res = evaluate_test_year(final_model, test_tr, feature_cols)
        eval_res["test_year"] = window["test_year"]
        results.append(eval_res)

        # print(f"AUC: {eval_res['auc']:.3f}, KS: {eval_res['ks']:.3f}, Brier: {eval_res['brier']:.3f}")

        # Save test predictions
        tmp = test_tr[["id", "fs_year", "default_flag"]].copy()
        tmp["pd_pred"] = eval_res["preds"]
        all_preds.append(tmp)
    
    return results, all_preds


# -------------------------------------------------------------------
# 6. FINAL TRAINING + SAVE ARTIFACT – what the harness will use
# -------------------------------------------------------------------

def train_and_save_model(train_csv_path: str, model_out_path: str = "pd_model.joblib"):
    """
    Full training pipeline:
    1. load training csv
    2. prepare features
    3. fit transforms on fs_year <= 2011
    4. train final logistic model (C=0.1 or your tuned value)
    5. save everything needed for prediction
    """
    # print_header("START: Full training pipeline execution")
    df = pd.read_csv(train_csv_path, low_memory=False)
    df = prepare_full_training_df(df)

    # wf_results, wf_preds = run_walk_forward(df)
    # print("\n===== WALK-FORWARD METRICS =====")
    # for res in wf_results:
    #     print(res)
    # print("\n===== WALK-FORWARD PREDICTIONS (first few rows) =====")
    # print(pd.concat(wf_preds, ignore_index=True).head())

    # print_header("TRAINING FINAL PRODUCTION MODEL (fs_year <= 2011)...")
    final_train_df = df[df["fs_year"] <= 2012].copy()

    transforms_full = fit_all_transforms(final_train_df, FINAL_RAW_RATIOS)
    final_train_tr, final_mean_tr = transform_train(
        final_train_df, FINAL_RAW_RATIOS, transforms_full
    )

    transformed_cols = [c.replace("_raw", "") + "_tr" for c in FINAL_RAW_RATIOS]
    feature_cols_full = transformed_cols + FINAL_MISSING_FLAGS + ["log_asst_tot"]

    X_full = final_train_tr[feature_cols_full].values
    y_full = final_train_tr["default_flag"].values

    # You hard-coded C=0.1 in the notebook based on WFA.
    final_prod_model = LogisticRegression(max_iter=5000, solver="lbfgs", C=0.1)
    final_prod_model.fit(X_full, y_full)

    artifacts = {
        "model": final_prod_model,
        "transforms_full": transforms_full,
        "final_mean_tr": final_mean_tr,
        "feature_cols_full": feature_cols_full,
        "raw_ratio_cols": FINAL_RAW_RATIOS,
    }

    # print_header(f"Saving model artifacts to {model_out_path}...")
    joblib.dump(artifacts, model_out_path)
    # print(f"\nSaved model artifacts to {model_out_path}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # print("Usage: python3 estimation.py <train_csv_path> [output_model_path]")
        raise SystemExit(1)

    train_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH

    train_and_save_model(train_path, out_path)