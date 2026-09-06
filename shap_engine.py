"""
shap_engine.py
--------------
Generates patient-specific and global SHAP explanations for both
the CKD (Random Forest) and Diabetes (XGBoost) models.
Uses TreeExplainer, which is fast and EXACT for tree-based models
(no approximation), matching Section 9 of the proposal.
"""

import shap
import numpy as np
import pandas as pd


class SHAPEngine:
    def __init__(self, ckd_model, diabetes_model,
                 ckd_background: pd.DataFrame = None,
                 diabetes_background: pd.DataFrame = None,
                 top_n: int = 5):
        self.top_n = top_n

        self.ckd_explainer = (
            shap.TreeExplainer(ckd_model, data=ckd_background)
            if ckd_background is not None else shap.TreeExplainer(ckd_model)
        )
        self.diabetes_explainer = (
            shap.TreeExplainer(diabetes_model, data=diabetes_background)
            if diabetes_background is not None else shap.TreeExplainer(diabetes_model)
        )

    @staticmethod
    def _extract_positive_class(shap_values, expected_value):
        """Handles both list-of-arrays and single-array SHAP output formats
        across different shap/sklearn/xgboost versions."""
        if isinstance(shap_values, list):
            return shap_values[1][0], (
                expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            )
        # Single array case (newer shap versions for binary classifiers)
        arr = np.array(shap_values)
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            return arr[0, :, 1], (
                expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            )
        return arr[0], expected_value

    def _format(self, shap_row, feature_names, feature_values):
        contributions = []
        for fname, fval, sval in zip(feature_names, feature_values, shap_row):
            contributions.append({
                "feature": fname,
                "value": float(fval) if isinstance(fval, (int, float, np.integer, np.floating)) else str(fval),
                "contribution": round(float(sval), 4),
                "direction": "increases_risk" if sval > 0 else "decreases_risk"
            })
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:self.top_n]

    def explain_ckd(self, X_row: pd.DataFrame) -> dict:
        raw = self.ckd_explainer.shap_values(X_row)
        row_values, base_value = self._extract_positive_class(raw, self.ckd_explainer.expected_value)
        return {
            "base_value": round(float(base_value), 4),
            "top_features": self._format(row_values, X_row.columns, X_row.iloc[0].values)
        }

    def explain_diabetes(self, X_row: pd.DataFrame) -> dict:
        raw = self.diabetes_explainer.shap_values(X_row)
        row_values, base_value = self._extract_positive_class(raw, self.diabetes_explainer.expected_value)
        return {
            "base_value": round(float(base_value), 4),
            "top_features": self._format(row_values, X_row.columns, X_row.iloc[0].values)
        }

    def global_importance(self, model_name: str, X_sample: pd.DataFrame) -> list:
        explainer = self.ckd_explainer if model_name == "ckd" else self.diabetes_explainer
        raw = explainer.shap_values(X_sample)

        if isinstance(raw, list):
            values = raw[1]
        else:
            arr = np.array(raw)
            values = arr[:, :, 1] if arr.ndim == 3 else arr

        mean_abs = np.abs(values).mean(axis=0)
        importance = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False)
        return importance.to_dict(orient="records")
