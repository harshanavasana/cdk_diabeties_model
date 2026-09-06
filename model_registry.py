"""
model_registry.py
------------------
Loads every trained artifact ONCE at process startup so the
orchestrator never re-loads a model per request (matters for
latency -- Section 18 lists end-to-end latency as an evaluation
metric).

Expected files in models/ (copy your saved .pkl files here):
    ckd_random_forest_model.pkl
    ckd_num_imputer.pkl
    ckd_cat_imputer.pkl
    ckd_label_encoders.pkl
    ckd_feature_order.pkl
    diabetes_xgboost_model.pkl
    diabetes_feature_order.pkl
"""

import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class ModelRegistry:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir

        def load(name):
            return joblib.load(os.path.join(models_dir, name))

        # CKD artifacts
        self.ckd_model = load("ckd_random_forest_model.pkl")
        self.ckd_num_imputer = load("ckd_num_imputer.pkl")
        self.ckd_cat_imputer = load("ckd_cat_imputer.pkl")
        self.ckd_encoders = load("ckd_label_encoders.pkl")
        self.ckd_feature_order = load("ckd_feature_order.pkl")
        self.ckd_scaler = load("ckd_scaler.pkl")

        # Diabetes artifacts
        self.diabetes_model = load("diabetes_xgboost_model.pkl")
        self.diabetes_feature_order = load("diabetes_feature_order.pkl")
        self.diabetes_scaler = load("diabetes_scaler.pkl")

        self.ckd_required_fields = self.ckd_feature_order
        self.diabetes_required_fields = self.diabetes_feature_order
