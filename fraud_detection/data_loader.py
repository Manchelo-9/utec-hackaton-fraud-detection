"""Load, merge, and reduce memory usage of the IEEE fraud dataset."""

import gc

import numpy as np
import pandas as pd

from .config import DATA_DIR, NA_THRESHOLD


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest viable dtype."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype = df[col].dtype
        if not hasattr(dtype, "kind"):
            continue
        if dtype.kind in ("i", "u"):
            c_min, c_max = df[col].min(), df[col].max()
            for candidate in [np.int8, np.int16, np.int32]:
                info = np.iinfo(candidate)
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(candidate)
                    break
        elif dtype.kind == "f":
            c_min, c_max = df[col].min(), df[col].max()
            finfo = np.finfo(np.float32)
            if c_min >= finfo.min and c_max <= finfo.max:
                df[col] = df[col].astype(np.float32)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem else 0
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({reduction:.1f}% reduction)")
    return df


def detect_high_na_cols(df: pd.DataFrame, threshold: float = NA_THRESHOLD) -> list[str]:
    """Return column names where the fraction of NAs exceeds *threshold*."""
    na_frac = df.isnull().mean()
    high_na = na_frac[na_frac > threshold].sort_values(ascending=False)
    if len(high_na):
        print(f"Detected {len(high_na)} columns with >{threshold*100:.0f}% NA:")
        for col, frac in high_na.items():
            print(f"  {col:30s} {frac*100:6.2f}%")
    return list(high_na.index)


def load_data(data_dir=DATA_DIR) -> pd.DataFrame:
    """Load and merge train transaction + identity CSVs (train only)."""
    print("Loading data...")
    train_txn = pd.read_csv(data_dir / "train_transaction.csv")
    train_id = pd.read_csv(data_dir / "train_identity.csv")

    train = pd.merge(train_txn, train_id, on="TransactionID", how="left")
    del train_txn, train_id
    gc.collect()

    print(f"Train shape: {train.shape}")
    train = reduce_mem_usage(train)
    return train
