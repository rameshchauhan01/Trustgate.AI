import pandas as pd
from sklearn.model_selection import train_test_split

from model_validation import (
    load_model,
    load_data,
    check_schema,
    evaluate_model,
    check_thresholds,
)

MODEL_PATH = "loan_approval_best_model.joblib"
TARGET_COL = "Has_Consumer_Loan"

# 1) Split once and save
df = pd.read_csv("./Customer_Financial_Info.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET_COL],
)

train_df.to_csv("Loan_Approval_train.csv", index=False)
test_df.to_csv("Loan_Approval_test.csv", index=False)

# 2) Point DATASETS to those two CSVs
DATASETS = {
    "train_80": "Loan_Approval_train.csv",
    "test_20": "Loan_Approval_test.csv",
}

def validate_dataset(model, data_path: str, target_col: str, label: str):
    print("\n" + "=" * 80)
    print(f"[DATASET] {label} -> {data_path}")
    print("=" * 80)

    # Load X, y
    X, y, df = load_data(data_path, target_col)

    # Schema check
    print("[STEP] Schema check")
    assert check_schema(model, X), f"Schema mismatch for dataset: {label}"

    # Metrics
    print("[STEP] Metrics")
    metrics = evaluate_model(model, X, y)

    # Thresholds (tune as needed)
    print("[STEP] Threshold checks")
    ok = check_thresholds(metrics, f1_threshold=0.96, auc_threshold=0.75)

    print(f"[RESULT] Dataset '{label}' -> OK? {ok}")
    return ok, metrics


def main():
    # Load trained pipeline (preprocessor + model)
    model = load_model(MODEL_PATH)

    results = {}

    for label, path in DATASETS.items():
        ok, metrics = validate_dataset(model, path, TARGET_COL, label)
        results[label] = {
            "ok": ok,
            **metrics
        }

    print("\n" + "#" * 80)
    print("SUMMARY ACROSS ALL DATASETS")
    print("#" * 80)
    for label, res in results.items():
        status = "PASS ✅" if res["ok"] else "FAIL ❌"
        print(f"\n[{label}] -> {status}")
        for k in ["accuracy", "precision", "recall", "f1", "auc"]:
            v = res.get(k)
            if v is None or pd.isna(v):
                print(f"  {k}: NaN")
            else:
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
