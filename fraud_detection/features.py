"""Feature engineering: UID construction, frequency encoding, aggregations."""

import gc

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import (AGG_COLS, AGG_STATS, DROP_COLS, FREQ_ENCODE_COLS,
                     SECONDS_PER_HOUR, TRAIN_SPLIT_RATIO)


def engineer_features(train, test):
    """Create features on combined train+test, then split back.

    Returns (X_train, y_train, X_test, submission_df).
    """
    print("Engineering features...")
    len_train = len(train)
    df = pd.concat([train, test], axis=0, sort=False)
    del train, test
    gc.collect()

    # UID: identifies clients by card + address + email domain
    df["uid"] = (df["card1"].astype(str) + "_"
                 + df["addr1"].astype(str) + "_"
                 + df["P_emaildomain"].astype(str))

    # Frequency encoding
    for col in FREQ_ENCODE_COLS:
        df[col + "_count"] = df[col].map(df[col].value_counts(dropna=False))

    # Group aggregations (cache the groupby object)
    grouped = df.groupby("uid")
    for col in AGG_COLS:
        if col in df.columns:
            for stat in AGG_STATS:
                df[f"{col}_to_uid_{stat}"] = grouped[col].transform(stat)

    # Time features
    df["hour"] = (df["TransactionDT"] / SECONDS_PER_HOUR) % 24

    # Decimal part of transaction amount
    df["TransactionAmt_decimal"] = ((df["TransactionAmt"] % 1) * 1000).astype(int)

    # Label-encode categorical columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Split back
    X_train = df.iloc[:len_train].drop(DROP_COLS, axis=1)
    y_train = df.iloc[:len_train]["isFraud"]
    X_test = df.iloc[len_train:].drop(DROP_COLS, axis=1)
    submission = pd.DataFrame({"TransactionID": df.iloc[len_train:]["TransactionID"]})
    del df
    gc.collect()

    return X_train, y_train, X_test, submission


def time_based_split(X, y, ratio=TRAIN_SPLIT_RATIO):
    """Split chronologically — train on first `ratio`, validate on the rest."""
    idx = int(len(X) * ratio)
    X_tr, X_val = X.iloc[:idx].copy(), X.iloc[idx:].copy()
    y_tr, y_val = y.iloc[:idx].copy(), y.iloc[idx:].copy()
    del X
    gc.collect()
    return X_tr, y_tr, X_val, y_val
