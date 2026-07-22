# harness.py
import argparse
import pandas as pd

from prediction import predict_pd


def run_harness(input_path: str, output_path: str):
    """
    Harness entry point.

    - input_path: CSV with raw financial statements (same structure as train-2025.csv)
    - output_path: path to save predictions as a 1-column CSV with no header
    """
    # 1. Read input CSV
    df_in = pd.read_csv(input_path, low_memory=False)

    # 2. Get PD estimates as a 1D numpy array
    pd_estimates = predict_pd(df_in)

    # 3. Save as a single-column CSV, no header, no index
    pd_series = pd.Series(pd_estimates)
    pd_series.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    # Run: python harness.py holdout.csv pd_predictions.csv
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV file with firm financials",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to output CSV file for PD predictions",
    )
    args = parser.parse_args()

    run_harness(args.input_csv, args.output_csv)