"""Feature engineering: UID construction, frequency encoding, aggregations,
NA handling, and train/test split — all leak-free."""

import gc

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from .config import (AGG_COLS, AGG_STATS, DROP_COLS, FREQ_ENCODE_COLS,
                     IDENTITY_MARKER_COL, RANDOM_SEED, SECONDS_PER_HOUR,
                     TEST_SIZE)
from .data_loader import detect_high_na_cols


# ── Public API ──────────────────────────────────────────────────────────


def prepare_data(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """Full leak-free pipeline: clean → split → engineer → impute.

    1. Add ``has_identity`` indicator.
    2. Drop columns with >50 % NA.
    3. Separate target, drop meta columns.
    4. Stratified random split.
    5. Engineer features on train (fit), then apply same mappings to test.
    6. Median-impute NaN for sklearn models.

    Returns (X_train, X_test, y_train, y_test).
    ``X_train`` / ``X_test`` contain an extra ``_imp`` suffix set of columns?
    No — returns two versions: raw (for LGB/XGB) and imputed (for RF) via
    a second call to ``impute_for_rf``.

    Actually returns the raw (NaN-preserving) frames.  Call ``impute_for_rf``
    afterwards for the RF-ready copies.
    """
    print("Preparing data...")

    # ── has_identity (before dropping id columns) ───────────────────────
    if IDENTITY_MARKER_COL in df.columns:
        df["has_identity"] = df[IDENTITY_MARKER_COL].notna().astype(np.int8)

    # ── Drop high-NA columns ────────────────────────────────────────────
    high_na_cols = detect_high_na_cols(df)
    cols_to_drop = [c for c in high_na_cols if c not in DROP_COLS]
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} high-NA columns.")
        df.drop(columns=cols_to_drop, inplace=True)

    # ── Separate target; keep TransactionDT for hour extraction later ──
    y = df["isFraud"].copy()
    pre_split_drop = [c for c in ("isFraud", "TransactionID") if c in df.columns]
    X = df.drop(columns=pre_split_drop)
    del df
    gc.collect()

    # ── Stratified random split BEFORE any feature engineering ──────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    del X, y
    gc.collect()
    print(f"Split → Train: {X_train.shape}, Test: {X_test.shape}  "
          f"(fraud rate train={y_train.mean():.4f}, test={y_test.mean():.4f})")

    # ── Feature engineering (train-fit, test-transform) ─────────────────
    X_train, X_test = _engineer_features(X_train, X_test)

    return X_train, X_test, y_train, y_test


def impute_for_rf(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute NaN values for sklearn models (fit on train only).

    Returns copies — originals are left untouched for LGB/XGB.
    """
    imputer = SimpleImputer(strategy="median")
    cols = X_train.columns

    train_nans = int(X_train.isnull().sum().sum())
    test_nans = int(X_test.isnull().sum().sum())

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=cols, index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=cols, index=X_test.index,
    )
    print(f"Imputed {train_nans:,} NaNs in train, "
          f"{test_nans:,} in test (median strategy).")
    return X_train_imp, X_test_imp


# ── Internal helpers ────────────────────────────────────────────────────


def _build_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Create pseudo-user ID from card1 + addr1 + email domain."""
    df["uid"] = (df["card1"].astype(str) + "_"
                 + df["addr1"].astype(str) + "_"
                 + df["P_emaildomain"].astype(str))
    return df


def _engineer_features(
    train: pd.DataFrame, test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply feature engineering: fit statistics on *train*, map onto *test*.

    Guarantees zero data leakage.
    """
    print("Engineering features (leak-free)...")

    # ── UID (deterministic per row — no leakage) ────────────────────────
    train = _build_uid(train)
    test = _build_uid(test)

    # ── Frequency encoding — train-only counts ──────────────────────────
    for col in FREQ_ENCODE_COLS:
        if col not in train.columns:
            continue
        freq_map = train[col].value_counts(dropna=False)
        train[col + "_count"] = train[col].map(freq_map).fillna(0).astype(np.int32)
        test[col + "_count"] = test[col].map(freq_map).fillna(0).astype(np.int32)

    # ── Group aggregations — train-only statistics ──────────────────────
    train_grouped = train.groupby("uid")
    for col in AGG_COLS:
        if col not in train.columns:
            continue
        stats = train_grouped[col].agg(AGG_STATS)
        stats.columns = [f"{col}_to_uid_{s}" for s in AGG_STATS]
        train = train.merge(stats, on="uid", how="left")
        test = test.merge(stats, on="uid", how="left")

    # ── Time features (row-level, no leakage) ───────────────────────────
    for df in (train, test):
        df["hour"] = (df["TransactionDT"] / SECONDS_PER_HOUR) % 24
        df["TransactionAmt_decimal"] = np.floor(
            (df["TransactionAmt"] % 1) * 1000
        ).astype(np.int16)

    # ── Label-encode categoricals — train-derived mapping ───────────────
    for col in train.columns:
        if not pd.api.types.is_string_dtype(train[col]):
            continue
        codes, uniques = pd.factorize(train[col].astype(str), sort=True)
        train[col] = codes.astype(np.int32)
        mapping = {val: idx for idx, val in enumerate(uniques)}
        test[col] = test[col].astype(str).map(mapping).fillna(-1).astype(np.int32)

    # ── Drop TransactionDT (used only for hour) ────────────────────────
    for df in (train, test):
        if "TransactionDT" in df.columns:
            df.drop(columns=["TransactionDT"], inplace=True)

    print(f"Final features: {train.shape[1]} columns.")
    return train, test
