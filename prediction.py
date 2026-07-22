# prediction.py
import pandas as pd
import joblib

from estimation import (
    clean_and_impute_base_vars,
    build_ratios_and_flags,
    transform_test,
    MODEL_PATH,
)


def load_artifacts(model_path: str = MODEL_PATH):
    """
    Load the artifacts saved by estimation.train_and_save_model.
    """
    artifacts = joblib.load(model_path)
    return artifacts


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    For NEW data:
    - we do NOT create default_flag (we don't know future defaults)
    - we DO run the same cleaning + ratios as in training
    """
    df = clean_and_impute_base_vars(df)
    df = build_ratios_and_flags(df)
    return df


def predict_pd(df_new: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    Main prediction function:
    - takes raw dataframe with accounting fields
    - returns dataframe with pd_pred column
    """
    artifacts = load_artifacts(model_path)
    model = artifacts["model"]
    transforms_full = artifacts["transforms_full"]
    final_mean_tr = artifacts["final_mean_tr"]
    feature_cols_full = artifacts["feature_cols_full"]
    raw_ratio_cols = artifacts["raw_ratio_cols"]

    df_proc = prepare_features_for_prediction(df_new.copy())

    df_tr = transform_test(df_proc, raw_ratio_cols, transforms_full, final_mean_tr)

    X_new = df_tr[feature_cols_full].values
    pd_hat = model.predict_proba(X_new)[:, 1]

    # Return a plain 1D array – harness will turn this into a 1-column CSV
    return pd_hat
