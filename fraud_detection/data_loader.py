"""Load, merge, and reduce memory usage of the IEEE fraud dataset."""

import gc

import numpy as np
import pandas as pd

from .config import DATA_DIR


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest viable dtype."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype = df[col].dtype
        if not hasattr(dtype, "kind"):
            continue
        if dtype.kind == "i" or dtype.kind == "u":
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


def load_data(data_dir=DATA_DIR):
    """Load and merge transaction + identity CSVs."""
    print("Loading data...")
    train_txn = pd.read_csv(data_dir / "train_transaction.csv")
    train_id = pd.read_csv(data_dir / "train_identity.csv")
    test_txn = pd.read_csv(data_dir / "test_transaction.csv")
    test_id = pd.read_csv(data_dir / "test_identity.csv")

    train = pd.merge(train_txn, train_id, on="TransactionID", how="left")
    test = pd.merge(test_txn, test_id, on="TransactionID", how="left")
    del train_txn, train_id, test_txn, test_id
    gc.collect()

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    return train, test
