import json
import warnings
import joblib
import mlflow
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLS,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_COLS,
)
from src.features import split_features_target_groups

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names.*",
)


def _sanitize_params(params: dict) -> dict[str, str]:
    return {str(k): str(v)[:250] for k, v in params.items()}


# 1. Preprocessing
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_COLS if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


# 2. Train and evaluate models
def train_and_evaluate(data_path=DATA_PATH) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X, y_raw, groups = split_features_target_groups(df)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Customer-level train/test split to prevent data leakage across months
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    assert set(groups.iloc[train_idx]).isdisjoint(
        set(groups.iloc[test_idx])
    ), "Customer leakage detected!"

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    mlflow.set_experiment("paisabazaar_credit_risk")

    results: dict[str, dict[str, float]] = {}
    best_f1 = -1.0
    best_model = None
    best_name = None
    best_predictions = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            accuracy = float(accuracy_score(y_test, predictions))
            macro_f1 = float(f1_score(y_test, predictions, average="macro"))
            weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))

            mlflow.log_params(_sanitize_params(model.get_params()))
            mlflow.log_metrics({
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            })

            results[name] = {
                "accuracy": round(accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
            }

            print(f"{name}: Accuracy={accuracy:.4f}, Macro F1={macro_f1:.4f}")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_model = pipeline
                best_name = name
                best_predictions = predictions

    class_names = [str(c) for c in encoder.classes_]

    print("\nBest Model:", best_name)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            best_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            best_predictions,
            labels=list(range(len(class_names))),
        )
    )

    joblib.dump(
        {
            "pipeline": best_model,
            "label_encoder": encoder,
            "classes": class_names,
            "feature_columns": list(X.columns),
            "best_model_name": best_name,
        },
        MODEL_PATH,
    )

    summary = {
        "best_model": best_name,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nModel saved:", MODEL_PATH)
    print("Metrics saved:", METRICS_PATH)

    return summary


if __name__ == "__main__":
    train_and_evaluate()