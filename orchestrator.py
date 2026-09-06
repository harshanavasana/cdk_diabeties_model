"""
orchestrator.py
----------------
The central coordination layer (Section 12 of proposal).

Responsibilities (and nothing more -- it never makes clinical
judgments itself):
  1. Validate inputs
  2. Route to CKD and/or diabetes pipeline
  3. Preprocess using each model's saved artifacts
  4. Invoke the model(s)
  5. Request SHAP explanations
  6. Aggregate into one structured result
  7. Save to memory
  8. Hand off structured evidence to the LLM for a natural-language
     explanation
"""

import pandas as pd
from datetime import datetime, timezone

from model_registry import ModelRegistry
from shap_engine import SHAPEngine
from memory_store import MemoryStore
from llm_engine import LLMEngine


class InputValidator:
    @staticmethod
    def validate(patient_data: dict, required_fields: list) -> dict:
        missing = [f for f in required_fields if f not in patient_data or patient_data[f] is None]
        return {"valid": len(missing) == 0, "missing_fields": missing}


class Preprocessor:
    @staticmethod
    def _encode_category(values, encoder):
        categories = {str(category): str(category) for category in encoder.classes_}
        normalized = []
        for value in values.astype(str):
            if value in categories:
                normalized.append(categories[value])
                continue
            try:
                numeric_value = float(value)
                numeric_match = next(
                    (category for category in categories if float(category) == numeric_value),
                    None,
                )
            except (TypeError, ValueError):
                numeric_match = None
            if numeric_match is None:
                normalized.append(value)
            else:
                normalized.append(numeric_match)
        return encoder.transform(normalized)

    @staticmethod
    def preprocess_ckd(patient_data: dict, registry: ModelRegistry) -> pd.DataFrame:
        df = pd.DataFrame([patient_data])[registry.ckd_feature_order]

        numeric_cols = list(registry.ckd_num_imputer.feature_names_in_)
        categorical_cols = list(registry.ckd_cat_imputer.feature_names_in_)

        if numeric_cols:
            df[numeric_cols] = registry.ckd_num_imputer.transform(df[numeric_cols])
        if categorical_cols:
            df[categorical_cols] = registry.ckd_cat_imputer.transform(df[categorical_cols])
            for col in categorical_cols:
                le = registry.ckd_encoders[col]
                df[col] = Preprocessor._encode_category(df[col], le)

        scaled = registry.ckd_scaler.transform(df[registry.ckd_feature_order])
        return pd.DataFrame(scaled, columns=registry.ckd_feature_order, index=df.index)

    @staticmethod
    def preprocess_diabetes(patient_data: dict, registry: ModelRegistry) -> pd.DataFrame:
        df = pd.DataFrame([patient_data])[registry.diabetes_feature_order]
        scaled = registry.diabetes_scaler.transform(df)
        return pd.DataFrame(scaled, columns=registry.diabetes_feature_order, index=df.index)


class Orchestrator:
    def __init__(self,
                 registry: ModelRegistry,
                 shap_engine: SHAPEngine = None,
                 memory_store: MemoryStore = None,
                 llm_engine: LLMEngine = None):
        self.registry = registry
        self.shap_engine = shap_engine
        self.memory = memory_store
        self.llm = llm_engine

    def run(self, patient_data: dict, run_ckd: bool = None, run_diabetes: bool = None,
            session_id: str = "default", generate_explanation: bool = True) -> dict:

        if run_ckd is None:
            run_ckd = all(
                patient_data.get(field) is not None
                for field in self.registry.ckd_required_fields
            )
        if run_diabetes is None:
            run_diabetes = all(
                patient_data.get(field) is not None
                for field in self.registry.diabetes_required_fields
            )

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ckd": None,
            "diabetes": None,
            "errors": []
        }

        if run_ckd is False and run_diabetes is False:
            supplied_fields = set(patient_data)
            ckd_supplied = supplied_fields.intersection(self.registry.ckd_required_fields)
            diabetes_supplied = supplied_fields.intersection(self.registry.diabetes_required_fields)
            if ckd_supplied or diabetes_supplied:
                result["errors"].append({
                    "pipeline": "routing",
                    "reason": "incomplete_patient_profile",
                    "details": {
                        "ckd_missing_fields": [
                            field for field in self.registry.ckd_required_fields
                            if ckd_supplied and field not in supplied_fields
                        ],
                        "diabetes_missing_fields": [
                            field for field in self.registry.diabetes_required_fields
                            if diabetes_supplied and field not in supplied_fields
                        ],
                        "message": (
                            "Complete one health profile before analysis. "
                            "The router needs all fields for at least one trained model."
                        ),
                    },
                })

        # ---------------- CKD PIPELINE ----------------
        if run_ckd:
            validation = InputValidator.validate(patient_data, self.registry.ckd_required_fields)
            if not validation["valid"]:
                result["errors"].append({
                    "pipeline": "ckd", "reason": "missing_fields",
                    "details": validation["missing_fields"]
                })
            else:
                try:
                    X_ckd = Preprocessor.preprocess_ckd(patient_data, self.registry)
                    pred = self.registry.ckd_model.predict(X_ckd)[0]
                    proba = self.registry.ckd_model.predict_proba(X_ckd)[0][1]

                    shap_result = self.shap_engine.explain_ckd(X_ckd) if self.shap_engine else None

                    result["ckd"] = {
                        "prediction": "ckd" if pred == 1 else "notckd",
                        "probability": float(proba),
                        "shap": shap_result
                    }
                except Exception as e:
                    result["errors"].append({"pipeline": "ckd", "reason": "execution_error", "details": str(e)})

        # ---------------- DIABETES PIPELINE ----------------
        if run_diabetes:
            validation = InputValidator.validate(patient_data, self.registry.diabetes_required_fields)
            if not validation["valid"]:
                result["errors"].append({
                    "pipeline": "diabetes", "reason": "missing_fields",
                    "details": validation["missing_fields"]
                })
            else:
                try:
                    X_dia = Preprocessor.preprocess_diabetes(patient_data, self.registry)
                    pred = self.registry.diabetes_model.predict(X_dia)[0]
                    proba = self.registry.diabetes_model.predict_proba(X_dia)[0][1]

                    shap_result = self.shap_engine.explain_diabetes(X_dia) if self.shap_engine else None

                    result["diabetes"] = {
                        "prediction": "diabetic" if pred == 1 else "not_diabetic",
                        "probability": float(proba),
                        "shap": shap_result
                    }
                except Exception as e:
                    result["errors"].append({"pipeline": "diabetes", "reason": "execution_error", "details": str(e)})

        # ---------------- MEMORY ----------------
        if self.memory:
            record_id = self.memory.save(result, session_id=session_id)
            result["memory_record_id"] = record_id

        # ---------------- LLM EXPLANATION ----------------
        if generate_explanation and self.llm and (result["ckd"] or result["diabetes"]):
            result["explanation"] = self.llm.generate_explanation(result)

        return result
