"""
Loan Approval Prediction - Production Style Script

- Loads data from CSV
- Preprocesses numeric & categorical features
- Trains multiple models with GridSearchCV
- Evaluates on a test set
- Selects the best model by F1-score
- Saves the best model and comparison summary

Usage:
    python loan_approval_training.py
"""

import pandas as pd
import numpy as np

from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

import joblib


# ======== CONFIG =========
DATA_PATH = "./Customer_Financial_Info.csv"
TARGET_COL = "Has_Consumer_Loan"
MODEL_OUTPUT_PATH = "loan_approval_best_model.joblib"
SUMMARY_OUTPUT_PATH = "model_comparison_summary.csv"
# =========================


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Split dataframe into features (X) and target (y),
    and detect numeric & categorical columns.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No features found after splitting.")

    return X, y, numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list, categorical_cols: list
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical features.
    - Numeric: median imputation + standard scaling
    - Categorical: most_frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def get_models_and_param_grid() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Define candidate models and their parameter grids for GridSearchCV.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    param_grid = {
        "LogisticRegression": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        "DecisionTree": {
            "clf__max_depth": [3, 5, 7, None],
            "clf__min_samples_split": [2, 5, 10],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
        },
        "RandomForest": {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        },
    }

    return models, param_grid


def train_and_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    models: Dict[str, Any],
    param_grid: Dict[str, Dict[str, Any]],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data, run GridSearchCV for each model, evaluate on test set,
    and return best model (by F1-score) and comparison summary.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_estimators = {}
    summary_rows = []

    for name, model in models.items():
        print(f"\n=== Training {name} ===")

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", model),
            ]
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid[name],
            cv=cv,
            scoring="f1",      # change here if you prefer 'roc_auc'
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_estimator = grid.best_estimator_
        best_estimators[name] = best_estimator

        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} best CV F1: {grid.best_score_:.4f}")

        # Evaluate on test set
        y_pred = best_estimator.predict(X_test)
        print(f"\n{name} classification report on test set:\n")
        print(classification_report(y_test, y_pred))

        if hasattr(best_estimator, "predict_proba"):
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = np.nan

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        summary_rows.append(
            {
                "Model": name,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "AUC-ROC": auc,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nModel comparison on test set:\n", summary_df)

    # Select best model by F1-score
    best_idx = summary_df["F1-Score"].idxmax()
    best_model_name = summary_df.loc[best_idx, "Model"]
    best_model = best_estimators[best_model_name]

    print(f"\nBest model (by F1-score): {best_model_name}")
    print(summary_df.loc[summary_df["Model"] == best_model_name])

    return best_model, summary_df


def save_artifacts(
    model: Any,
    summary_df: pd.DataFrame,
    model_path: str = MODEL_OUTPUT_PATH,
    summary_path: str = SUMMARY_OUTPUT_PATH,
):
    """
    Save trained model and summary dataframe to disk.
    """
    joblib.dump(model, model_path)
    print(f"Saved best model to: {model_path}")

    summary_df.to_csv(summary_path, index=False)
    print(f"Saved model comparison summary to: {summary_path}")


def predict_single(model: Any, input_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a prediction for a single applicant using the trained pipeline model.
    `model` is the full pipeline (preprocessor + classifier).

    Example:
        loaded_model = joblib.load(MODEL_OUTPUT_PATH)
        sample = {
            "Gender": "Male",
            "Married": "Yes",
            "ApplicantIncome": 5000,
            ...
        }
        y_pred, y_proba = predict_single(loaded_model, sample)
    """
    df_single = pd.DataFrame([input_dict])
    y_pred = model.predict(df_single)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(df_single)
    else:
        y_proba = np.array([[np.nan] * len(model.classes_)])
    return y_pred, y_proba


def main():
    """
    End-to-end training pipeline:
    - Load data
    - Split features/target
    - Build preprocessor
    - Train multiple models with GridSearchCV
    - Select best model
    - Save artifacts
    """
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")

    print("Preparing features and target...")
    X, y, numeric_cols, categorical_cols = split_features_target(df, TARGET_COL)
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    print("Building preprocessor...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    print("Setting up models and hyperparameters...")
    models, param_grid = get_models_and_param_grid()

    print("Training models and selecting the best one...")
    best_model, summary_df = train_and_select_model(
        X, y, preprocessor, models, param_grid
    )

    print("Saving artifacts...")
    save_artifacts(best_model, summary_df)

    print("\nDone. You can now load the model with joblib.load "
          "and use predict_single() for inference.")


if __name__ == "__main__":
    main()
