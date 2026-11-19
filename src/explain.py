from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.features import engineer_features

try:
    import shap
except ImportError:
    shap = None


def get_feature_names(pipeline, fallback_cols: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            raw_names = list(preprocessor.get_feature_names_out())
            return [name.split("__", 1)[-1] for name in raw_names]
        except Exception:
            pass
    return fallback_cols


def get_global_feature_importance(model_path=MODEL_PATH, top_n: int = 10) -> pd.DataFrame:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    feature_cols = payload["feature_columns"]

    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    names = get_feature_names(pipeline, feature_cols)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        importances = np.ones(len(names), dtype=float)

    # Align lengths safely if needed
    min_len = min(len(names), len(importances))
    df_imp = pd.DataFrame({
        "feature": names[:min_len],
        "importance": importances[:min_len],
    }).sort_values("importance", ascending=False)

    return df_imp.head(top_n).reset_index(drop=True)


def explain_single_record(
    record: dict[str, Any],
    model_path=MODEL_PATH,
    background_df: pd.DataFrame | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    feature_cols = payload["feature_columns"]

    raw_df = pd.DataFrame([record])
    engineered = engineer_features(raw_df)
    for col in feature_cols:
        if col not in engineered.columns:
            engineered[col] = None
    X_row = engineered[feature_cols]

    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")
    names = get_feature_names(pipeline, feature_cols)

    X_trans = preprocessor.transform(X_row) if preprocessor is not None else X_row.to_numpy()
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Try SHAP TreeExplainer for tree ensembles
    if shap is not None and hasattr(model, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_trans)

            pred_idx = int(pipeline.predict(X_row)[0])

            # Handle binary, multiclass list, or 3D array SHAP outputs
            if isinstance(shap_vals, list):
                class_shap = np.asarray(shap_vals[pred_idx])[0]
            else:
                arr = np.asarray(shap_vals)
                if arr.ndim == 3:
                    # shape: (n_samples, n_features, n_classes)
                    class_shap = arr[0, :, pred_idx]
                elif arr.ndim == 2:
                    class_shap = arr[0]
                else:
                    class_shap = np.asarray(model.feature_importances_, dtype=float)

            min_len = min(len(names), len(class_shap))
            df_exp = pd.DataFrame({
                "feature": names[:min_len],
                "shap_impact": np.round(class_shap[:min_len].astype(float), 4),
                "abs_impact": np.abs(class_shap[:min_len].astype(float)),
            }).sort_values("abs_impact", ascending=False)

            return df_exp.head(top_n)[["feature", "shap_impact"]].reset_index(drop=True)
        except Exception:
            pass

    # Fallback: top global feature importances
    return get_global_feature_importance(model_path=model_path, top_n=top_n)