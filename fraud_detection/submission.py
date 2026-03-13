"""Save individual and ensemble submission CSVs."""

from .config import OUTPUT_DIR


def save_submissions(submission_df, test_preds: dict, ensemble_test, output_dir=OUTPUT_DIR):
    """Write one CSV per model plus the ensemble submission."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Saving submissions...")

    for name, preds in test_preds.items():
        path = output_dir / f"submission_{name}.csv"
        submission_df["isFraud"] = preds
        submission_df.to_csv(path, index=False)

    path = output_dir / "submission_ensemble.csv"
    submission_df["isFraud"] = ensemble_test
    submission_df.to_csv(path, index=False)

    print(f"  Saved {len(test_preds) + 1} submission files to {output_dir}/")
