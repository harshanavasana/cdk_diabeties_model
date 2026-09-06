"""
llm_engine.py
-------------
LLM interpretation layer (Section 10 of proposal).

IMPORTANT DESIGN NOTE: the LLM here is an INTERPRETER, not a
diagnostician. It is only given structured predictions + SHAP
evidence and asked to phrase it in plain language. It is NOT
asked to reason about medicine on its own, which is why a small
general-purpose instruct model is appropriate here -- a heavier
"medical" model would risk injecting clinical claims that go
beyond your SHAP evidence, which is exactly what Section 10 and
Section 23 (Limitations) warn against.

Model choice: Qwen2.5-1.5B-Instruct (free, runs on CPU, small
enough for Hugging Face Spaces' free tier). Swap MODEL_NAME for
any other free instruct model if you want to experiment.

A template-based fallback is included so the app still produces
a correct, grounded explanation even if the model fails to load
(e.g. out of memory on a free Space) -- this keeps the whole
pipeline demo-safe.
"""

import json

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class LLMEngine:
    def __init__(self, model_name: str = MODEL_NAME, use_model: bool = True):
        self.model_name = model_name
        self.pipe = None

        if use_model:
            try:
                from transformers import pipeline
                self.pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device_map="auto",
                    max_new_tokens=300,
                )
            except Exception as e:
                print(f"[LLMEngine] Could not load model '{model_name}': {e}")
                print("[LLMEngine] Falling back to template-based explanation.")
                self.pipe = None

    # ------------------------------------------------------------------
    # PROMPT CONSTRUCTION — only grounded evidence goes into the prompt
    # ------------------------------------------------------------------
    def _build_prompt(self, evidence: dict) -> str:
        lines = [
            "You are a medical explanation assistant. You must ONLY use the "
            "structured evidence given below. Do not invent facts, do not add "
            "medical knowledge not present in the evidence, and do not provide "
            "a diagnosis. State clearly that this is a model prediction, not a "
            "medical diagnosis, and that a clinician should be consulted.",
            "",
            "EVIDENCE:",
        ]

        ckd = evidence.get("ckd")
        if ckd:
            lines.append(f"- CKD model prediction: {ckd['prediction']} "
                          f"(probability {ckd['probability']:.2f})")
            if ckd.get("shap"):
                top = ckd["shap"]["top_features"]
                feat_str = "; ".join(
                    f"{f['feature']}={f['value']} ({f['direction']}, "
                    f"contribution {f['contribution']:+.3f})" for f in top
                )
                lines.append(f"  Top contributing CKD features: {feat_str}")

        diabetes = evidence.get("diabetes")
        if diabetes:
            lines.append(f"- Diabetes model prediction: {diabetes['prediction']} "
                          f"(probability {diabetes['probability']:.2f})")
            if diabetes.get("shap"):
                top = diabetes["shap"]["top_features"]
                feat_str = "; ".join(
                    f"{f['feature']}={f['value']} ({f['direction']}, "
                    f"contribution {f['contribution']:+.3f})" for f in top
                )
                lines.append(f"  Top contributing diabetes features: {feat_str}")

        lines.append("")
        lines.append("Write a short (4-6 sentence) plain-language explanation "
                      "of these results for a patient, grounded only in the "
                      "evidence above.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # TEMPLATE FALLBACK — deterministic, always available, no model needed
    # ------------------------------------------------------------------
    def _template_explanation(self, evidence: dict) -> str:
        parts = []
        ckd = evidence.get("ckd")
        diabetes = evidence.get("diabetes")

        if ckd:
            risk_word = "elevated" if ckd["prediction"] == "ckd" else "low"
            parts.append(
                f"The CKD model estimates a {risk_word} likelihood of chronic "
                f"kidney disease ({ckd['probability']*100:.1f}% probability)."
            )
            if ckd.get("shap"):
                top = ckd["shap"]["top_features"][:3]
                feat_list = ", ".join(f"{f['feature']} ({f['direction'].replace('_', ' ')})" for f in top)
                parts.append(f"The strongest contributing factors were: {feat_list}.")

        if diabetes:
            risk_word = "elevated" if diabetes["prediction"] == "diabetic" else "low"
            parts.append(
                f"The diabetes model estimates a {risk_word} risk "
                f"({diabetes['probability']*100:.1f}% probability)."
            )
            if diabetes.get("shap"):
                top = diabetes["shap"]["top_features"][:3]
                feat_list = ", ".join(f"{f['feature']} ({f['direction'].replace('_', ' ')})" for f in top)
                parts.append(f"The strongest contributing factors were: {feat_list}.")

        parts.append(
            "These are statistical model predictions based on the supplied "
            "data, not a medical diagnosis. Please consult a qualified "
            "clinician for interpretation and next steps."
        )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------------------
    def generate_explanation(self, orchestrator_result: dict) -> str:
        evidence = {
            "ckd": orchestrator_result.get("ckd"),
            "diabetes": orchestrator_result.get("diabetes"),
        }

        if not evidence["ckd"] and not evidence["diabetes"]:
            return "No predictions were generated, so no explanation is available."

        if self.pipe is None:
            return self._template_explanation(evidence)

        try:
            prompt = self._build_prompt(evidence)
            messages = [{"role": "user", "content": prompt}]
            output = self.pipe(messages)
            generated = output[0]["generated_text"]
            # transformers chat pipelines return the full message list;
            # take the last assistant turn
            if isinstance(generated, list):
                text = generated[-1]["content"]
            else:
                text = str(generated)
            return text.strip()
        except Exception as e:
            print(f"[LLMEngine] Generation failed, using template fallback: {e}")
            return self._template_explanation(evidence)
