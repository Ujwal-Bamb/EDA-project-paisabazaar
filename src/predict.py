import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.features import engineer_features


class CreditRiskPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")

        saved_model = joblib.load(model_path)

        self.model = saved_model["pipeline"]
        self.encoder = saved_model["label_encoder"]
        self.classes = saved_model["classes"]
        self.features = saved_model["feature_columns"]
        self.model_name = saved_model["best_model_name"]

    def predict_records(self, records):
        if not records:
            return []

        df = pd.DataFrame(records)
        df = engineer_features(df)
        df = df.reindex(columns=self.features)

        pred_indices = self.model.predict(df)
        predictions = self.encoder.inverse_transform(pred_indices)

        has_proba = hasattr(self.model, "predict_proba")
        probabilities = self.model.predict_proba(df) if has_proba else None

        results = []
        for i, prediction in enumerate(predictions):
            item = {
                "predicted_credit_score": str(prediction),
                "model_name": self.model_name,
            }
            if probabilities is not None:
                item["probabilities"] = {
                    str(cls): round(float(probabilities[i][j]), 4)
                    for j, cls in enumerate(self.classes)
                }
            results.append(item)

        return results