# src/preprocessing/features.py

import pandas as pd


def load_raw_weather(csv_path: str) -> pd.DataFrame:
    """
    Load the raw Kaggle/ECA&D weather prediction dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the weather_prediction_dataset.csv file.

    Returns
    -------
    pd.DataFrame
        Raw weather dataframe exactly as stored in the CSV.
    """
    df = pd.read_csv(csv_path)
    return df


def make_basel_features(raw_df: pd.DataFrame, city: str = "BASEL") -> pd.DataFrame:
    """
    Given the raw Kaggle/ECA&D weather dataframe, build the Basel-only
    feature table used in the project.

    Steps:
    - sort rows by DATE
    - create RainToday and RainTomorrow labels
    - add MONTH for seasonality
    - add 1-day lags of key weather variables
    - drop rows where lags are undefined

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw weather dataframe (all stations).
    city : str, optional
        Station name prefix to use (default is "BASEL").

    Returns
    -------
    pd.DataFrame
        Processed dataframe with DATE, labels, and engineered features.
    """
    df = raw_df.copy()

    # ensure DATE is datetime and sorted
    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), errors="coerce")
    df = df.sort_values("DATE").reset_index(drop=True)

    pref = f"{city}_"

    # labels
    df["RainToday"] = (df[f"{pref}precipitation"] > 0).astype(int)
    df["RainTomorrow"] = (df[f"{pref}precipitation"].shift(-1) > 0).astype(int)

    # drop last day with no "tomorrow"
    df = df.dropna(subset=["RainTomorrow"]).reset_index(drop=True)

    # simple seasonality feature
    df["MONTH"] = df["DATE"].dt.month

    # 1-day lags for main weather drivers
    for col in [f"{pref}pressure", f"{pref}humidity", f"{pref}temp_mean", f"{pref}sunshine"]:
        if col in df.columns:
            df[col + "_lag1"] = df[col].shift(1)

    # first row now has NaNs from lags → drop
    df = df.dropna().reset_index(drop=True)

    # final column order (only keep if they exist)
    feature_cols = [
        "DATE",
        "MONTH",
        "RainToday",
        "RainTomorrow",
        f"{pref}pressure",
        f"{pref}humidity",
        f"{pref}temp_mean",
        f"{pref}sunshine",
        f"{pref}pressure_lag1",
        f"{pref}humidity_lag1",
        f"{pref}temp_mean_lag1",
        f"{pref}sunshine_lag1",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df[feature_cols].copy()
