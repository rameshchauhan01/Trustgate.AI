"""
model_validation.py

Generic validation framework for your ML models.

- Loads a trained sklearn Pipeline (preprocessor + classifier) from .joblib
- Loads a CSV with features + target
- Validates:
    * schema (required columns present)
    * metrics (accuracy, precision, recall, F1, AUC)
    * threshold checks (e.g., F1 >= 0.75)

Usage (basic):
    python model_validation.py \
        --model-path loan_approval_best_model.joblib \
        --data-path Loan_Approval_Validation.csv \
        --target-col Loan_Status
"""

import argparse
import numpy as np
import pandas as pd
import joblib

from typing import Tuple, Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ====== DEFAULT CONFIG (override via CLI) ======
DEFAULT_MODEL_PATH = "loan_approval_best_model.joblib"
DEFAULT_DATA_PATH = "./Customer_Financial_Info.csv"            # or Loan_Approval_Validation.csv
DEFAULT_TARGET_COL = "Has_Consumer_Loan"

F1_THRESHOLD = 0.75
AUC_THRESHOLD = 0.75
# ================================================


def load_model(model_path: str) -> Any:
    """Load trained sklearn pipeline from .joblib."""
    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model


def load_data(data_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load CSV and split into X, y."""
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[INFO] Data shape: {df.shape} (X: {X.shape}, y: {y.shape})")
    return X, y, df


def get_expected_columns_from_model(model: Any):
    """
    Extract the expected feature column names from the fitted ColumnTransformer
    inside the 'preprocessor' step of the pipeline.
    """
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
        print("[WARN] No 'preprocessor' step found in pipeline. Skipping schema check.")
        return None

    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "transformers_"):
        print("[WARN] Preprocessor not fitted or has no transformers_. Skipping schema check.")
        return None

    expected_cols = []
    for name, transformer, cols in preprocessor.transformers_:
        # cols can be list or slice or 'drop'
        if cols is None or cols == "drop":
            continue
        if isinstance(cols, list):
            expected_cols.extend(cols)

    return expected_cols


def check_schema(model: Any, X: pd.DataFrame) -> bool:
    """
    Basic schema validation:
    - All expected columns must be present in X.
    - Extra columns in X are allowed but reported.
    """
    expected_cols = get_expected_columns_from_model(model)
    if expected_cols is None:
        # can't check schema -> assume OK
        print("[INFO] Schema check skipped (no expected columns found).")
        return True

    missing = set(expected_cols) - set(X.columns)
    extra = set(X.columns) - set(expected_cols)

    if missing:
        print("[ERROR] Missing columns in input data:", missing)
    else:
        print("[INFO] No required columns missing.")

    if extra:
        print("[INFO] Extra columns in data (ignored by model):", extra)

    return len(missing) == 0


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Run predictions and compute evaluation metrics.
    """
    print("[INFO] Running predictions...")
    y_pred = model.predict(X)

    print("\n[INFO] Classification report:")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    print("\n[INFO] Confusion matrix:")
    print(cm)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary"
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        try:
            metrics["auc"] = roc_auc_score(y, y_proba)
        except ValueError:
            # e.g. only one class present in y
            metrics["auc"] = np.nan
    else:
        metrics["auc"] = np.nan

    print("\n[INFO] Metrics:")
    for k, v in metrics.items():
        if np.isnan(v):
            print(f"  {k}: NaN")
        else:
            print(f"  {k}: {v:.4f}")

    return metrics


def check_thresholds(
    metrics: Dict[str, float],
    f1_threshold: float = F1_THRESHOLD,
    auc_threshold: float = AUC_THRESHOLD,
) -> bool:
    """
    Check if the model metrics meet predefined thresholds.
    Returns True if all checks pass, else False.
    """
    ok = True

    f1 = metrics.get("f1", np.nan)
    auc = metrics.get("auc", np.nan)

    if not np.isnan(f1) and f1 < f1_threshold:
        print(f"[FAIL] F1-score {f1:.4f} < threshold {f1_threshold}")
        ok = False
    else:
        print(f"[PASS] F1-score check (threshold {f1_threshold})")

    if not np.isnan(auc):
        if auc < auc_threshold:
            print(f"[FAIL] AUC {auc:.4f} < threshold {auc_threshold}")
            ok = False
        else:
            print(f"[PASS] AUC check (threshold {auc_threshold})")
    else:
        print("[WARN] AUC is NaN (maybe no predict_proba or only one class). Skipping AUC check.")

    return ok


def main(
    model_path: str,
    data_path: str,
    target_col: str,
    f1_threshold: float = F1_THRESHOLD,
    auc_threshold: float = AUC_THRESHOLD,
) -> None:
    """
    Full validation workflow.
    """
    model = load_model(model_path)
    X, y, _ = load_data(data_path, target_col)

    print("\n[STEP] Schema validation")
    schema_ok = check_schema(model, X)
    if not schema_ok:
        print("[RESULT] ❌ Validation FAILED due to schema mismatch.")
        return

    print("\n[STEP] Metric evaluation")
    metrics = evaluate_model(model, X, y)

    print("\n[STEP] Threshold checks")
    ok = check_thresholds(metrics, f1_threshold, auc_threshold)

    if ok:
        print("\n[RESULT] ✅ Model PASSED validation.")
    else:
        print("\n[RESULT] ❌ Model FAILED validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained ML model on a dataset.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model .joblib file.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to the CSV file for validation.")
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL,
                        help="Name of the target column in the CSV.")
    parser.add_argument("--f1-threshold", type=float, default=F1_THRESHOLD,
                        help="Minimum acceptable F1-score.")
    parser.add_argument("--auc-threshold", type=float, default=AUC_THRESHOLD,
                        help="Minimum acceptable AUC-ROC (if available).")

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        data_path=args.data_path,
        target_col=args.target_col,
        f1_threshold=args.f1_threshold,
        auc_threshold=args.auc_threshold,
    )
