"""
app.py
------
Gradio front-end + wiring for Hugging Face Spaces deployment.

Hugging Face Spaces auto-detects `app.py` and runs it if you choose
the "Gradio" SDK when creating the Space. Just push this whole
folder (including models/) to your Space's repo.
"""

import json
import joblib
import os
import gradio as gr

from model_registry import ModelRegistry
from shap_engine import SHAPEngine
from memory_store import MemoryStore
from llm_engine import LLMEngine
from orchestrator import Orchestrator

# ------------------------------------------------------------------
# STARTUP: load everything ONCE when the Space boots
# ------------------------------------------------------------------
print("Loading model registry...")
registry = ModelRegistry()

print("Loading SHAP background samples...")
# Background samples improve SHAP baseline quality; falling back to
# None still works since TreeExplainer can run without one.
try:
    ckd_background = joblib.load("models/ckd_background_sample.pkl")
except FileNotFoundError:
    ckd_background = None
try:
    diabetes_background = joblib.load("models/diabetes_background_sample.pkl")
except FileNotFoundError:
    diabetes_background = None

shap_engine = SHAPEngine(
    ckd_model=registry.ckd_model,
    diabetes_model=registry.diabetes_model,
    ckd_background=ckd_background,
    diabetes_background=diabetes_background,
)

print("Initializing memory store (SQLite)...")
memory = MemoryStore(db_path="memory.db")

print("Loading LLM engine (this may take a minute on first boot)...")
use_llm_model = os.getenv("CKD_USE_LLM_MODEL", "false").lower() in {"1", "true", "yes"}
llm = LLMEngine(use_model=use_llm_model)

orchestrator = Orchestrator(registry, shap_engine=shap_engine, memory_store=memory, llm_engine=llm)

CKD_FIELDS = list(registry.ckd_feature_order)
DIABETES_FIELDS = list(registry.diabetes_feature_order)

CSS = """
:root {
    --ink: #173042;
    --muted: #607584;
    --teal: #087f8c;
    --teal-dark: #075c68;
    --aqua: #e8f6f3;
    --line: #d8e5e4;
    --paper: #fbfdfc;
}
body { background: #edf5f3; }
.gradio-container { max-width: 1180px !important; margin: 0 auto; }
.hero { background: linear-gradient(120deg, #0b5965, #169c96); color: white; border-radius: 18px; padding: 30px 34px; margin-bottom: 18px; box-shadow: 0 12px 30px rgba(8, 91, 101, .16); }
.hero h1 { font-size: 34px; margin: 0 0 8px; color: white; }
.hero p { margin: 0; max-width: 720px; color: #e8fbf7; font-size: 16px; }
.section { background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 18px; }
.section-title { color: var(--ink); font-size: 18px; font-weight: 700; margin: 0 0 4px; }
.section-copy { color: var(--muted); margin: 0 0 14px; }
.status { background: var(--aqua); border: 1px solid #b9dfd8; border-radius: 12px; padding: 12px 16px; color: var(--teal-dark); }
.primary-btn { min-height: 52px; font-size: 17px !important; }
footer { display: none !important; }
"""

def _display_name(field):
    return field.replace("_", " ").replace("GenHlth", "General health").title()


def _build_feature_inputs(fields, defaults=None, encoders=None):
    """Build form controls from the exact feature order used by each model."""
    defaults = defaults or {}
    encoders = encoders or {}
    components = []

    for start in range(0, len(fields), 2):
        with gr.Row():
            for field in fields[start:start + 2]:
                label = _display_name(field)
                if field in encoders:
                    choices = [str(value) for value in encoders[field].classes_]
                    value = defaults.get(field)
                    if value is not None:
                        value = str(value)
                    component = gr.Dropdown(
                        choices=choices,
                        value=value if value in choices else None,
                        label=label,
                        allow_custom_value=False,
                    )
                else:
                    component = gr.Number(
                        label=label,
                        value=defaults.get(field),
                    )
                components.append(component)

    return components


# ------------------------------------------------------------------
# GRADIO CALLBACK
# ------------------------------------------------------------------
def predict(session_id, *feature_values):
    patient_data = {
        field: value
        for field, value in zip(CKD_FIELDS + DIABETES_FIELDS, feature_values)
        if value is not None and value != ""
    }

    result = orchestrator.run(
        patient_data,
        session_id=session_id or "default",
    )

    explanation = result.get("explanation", "(no explanation generated)")
    completed = []
    if result.get("ckd"):
        completed.append("kidney health")
    if result.get("diabetes"):
        completed.append("diabetes and general health")
    if completed:
        status = "Analysis completed for: " + " and ".join(completed) + "."
    elif result.get("errors"):
        status = "Please complete all fields for at least one health profile before analyzing."
    else:
        status = "Add patient information to begin the analysis."
    return json.dumps(result, indent=2), explanation, status


def clear_form():
    return [None] * (len(CKD_FIELDS) + len(DIABETES_FIELDS)) + ["", "", "Add patient information to begin the analysis."]


with gr.Blocks(title="Patient Health Insights", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        '<div class="hero">'
        '<h1>Patient Health Insights</h1>'
        '<p>Enter the health information you have. Our analysis router identifies which complete health profile can be evaluated and explains the result in plain language.</p>'
        '</div>'
    )

    gr.Markdown(
        "**One patient profile, one analysis button.** You do not need to choose a model or enter JSON. "
        "Complete either health section, or complete both for a combined view. This tool is for research and education, not diagnosis."
    )
    session_id = gr.Textbox(label="Session name (optional)", value="demo-session", visible=False)

    with gr.Group(elem_classes="section"):
        gr.Markdown("### Kidney health", elem_classes="section-title")
        gr.Markdown("Lab results and symptoms commonly used for kidney health analysis.", elem_classes="section-copy")
        ckd_inputs = _build_feature_inputs(CKD_FIELDS, encoders=registry.ckd_encoders)

    with gr.Group(elem_classes="section"):
        gr.Markdown("### General health and lifestyle", elem_classes="section-title")
        gr.Markdown("Everyday health information used for diabetes risk analysis.", elem_classes="section-copy")
        diabetes_inputs = _build_feature_inputs(DIABETES_FIELDS)

    with gr.Row():
        submit_btn = gr.Button("Analyze patient profile", variant="primary", elem_classes="primary-btn")
        clear_btn = gr.Button("Clear form")

    status_output = gr.Markdown("Add patient information to begin the analysis.", elem_classes="status")
    explanation_output = gr.Textbox(label="Your results", lines=10)
    with gr.Accordion("Technical details", open=False):
        result_output = gr.Code(label="Structured result", language="json", lines=10)

    submit_btn.click(
        fn=predict,
        inputs=[session_id, *ckd_inputs, *diabetes_inputs],
        outputs=[result_output, explanation_output, status_output]
    )
    clear_btn.click(
        fn=clear_form,
        inputs=[],
        outputs=[*ckd_inputs, *diabetes_inputs, result_output, explanation_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()
