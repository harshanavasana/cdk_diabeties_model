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

try:
    import spaces
except ImportError:
    class _LocalSpaces:
        @staticmethod
        def GPU(function):
            return function

    spaces = _LocalSpaces()

from model_registry import ModelRegistry
from shap_engine import SHAPEngine
from memory_store import MemoryStore
from auth_store import AuthStore
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
auth = AuthStore(db_path="memory.db")

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
.auth-box { max-width: 480px; margin: 42px auto; }
.step-note { color: var(--muted); font-size: 14px; }
.result-box textarea { font-size: 16px !important; line-height: 1.55 !important; }
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


def _updates(visible):
    return gr.update(visible=visible)


def create_account(email, password):
    _, message = auth.create_user(email, password)
    return message


def login(email, password):
    valid, message = auth.authenticate(email, password)
    if not valid:
        return message, _updates(True), _updates(False), "", ""
    return f"Signed in as {message}", _updates(False), _updates(True), message, f"Signed in as `{message}`"


def logout():
    return "You are signed out.", _updates(True), _updates(False), "", ""


def _patient_data(fields, values):
    return {field: value for field, value in zip(fields, values) if value is not None and value != ""}


@spaces.GPU
def run_ckd(session_id, *values):
    result = orchestrator.run(
        _patient_data(CKD_FIELDS, values),
        run_ckd=True,
        run_diabetes=False,
        session_id=session_id or "default",
    )
    return json.dumps(result, indent=2), result.get("explanation", ""), result


@spaces.GPU
def run_diabetes(session_id, ckd_result, *values):
    patient_data = _patient_data(DIABETES_FIELDS, values)
    result = orchestrator.run(
        patient_data,
        run_ckd=False,
        run_diabetes=True,
        session_id=session_id or "default",
    )
    if isinstance(ckd_result, dict) and ckd_result.get("ckd"):
        result["ckd"] = ckd_result["ckd"]
        result["explanation"] = llm.generate_explanation(result)
    return json.dumps(result, indent=2), result.get("explanation", ""), result


def skip_ckd():
    return "CKD skipped. Complete the diabetes profile below, or open Results when ready.", _updates(True)


def skip_diabetes():
    return "Diabetes skipped. Open Results to view the completed analysis.", _updates(True)


def clear_results():
    return "", "No analysis has been run yet."


with gr.Blocks(title="Patient Health Insights") as demo:
    session_id = gr.State("")
    ckd_result_state = gr.State({})

    with gr.Column(visible=True, elem_classes="auth-box") as auth_view:
        gr.HTML('<div class="hero"><h1>Patient Health Insights</h1><p>Sign in to run private health model analyses.</p></div>')
        with gr.Group(elem_classes="section"):
            email = gr.Textbox(label="Email", placeholder="you@example.com")
            password = gr.Textbox(label="Password", type="password")
            with gr.Row():
                login_btn = gr.Button("Log in", variant="primary", elem_classes="primary-btn")
                signup_btn = gr.Button("Create account")
            auth_status = gr.Markdown("Use an existing account or create one.", elem_classes="status")

    with gr.Column(visible=False) as app_view:
        gr.HTML('<div class="hero"><h1>Patient Health Insights</h1><p>Complete either model, skip what you do not need, then review the explanation on the Results page.</p></div>')
        with gr.Row():
            signed_in_as = gr.Markdown()
            logout_btn = gr.Button("Log out")
        session_id.value = ""
        with gr.Tabs():
            with gr.Tab("1. Kidney model"):
                gr.Markdown("Enter every kidney-health field, then continue to the next model. You may skip this model.", elem_classes="step-note")
                ckd_inputs = _build_feature_inputs(CKD_FIELDS, encoders=registry.ckd_encoders)
                with gr.Row():
                    ckd_btn = gr.Button("Run kidney model", variant="primary", elem_classes="primary-btn")
                    skip_ckd_btn = gr.Button("Skip kidney model")
                ckd_status = gr.Markdown("", elem_classes="status")
            with gr.Tab("2. Diabetes model"):
                gr.Markdown("Enter every diabetes-health field, then open Results. You may skip this model.", elem_classes="step-note")
                diabetes_inputs = _build_feature_inputs(DIABETES_FIELDS)
                with gr.Row():
                    diabetes_btn = gr.Button("Run diabetes model", variant="primary", elem_classes="primary-btn")
                    skip_diabetes_btn = gr.Button("Skip diabetes model")
                diabetes_status = gr.Markdown("", elem_classes="status")
            with gr.Tab("3. Results and explanation"):
                gr.Markdown("### Explainable AI results", elem_classes="section-title")
                gr.Markdown("This page separates the model predictions, evidence, and plain-language explanation.", elem_classes="section-copy")
                explanation_output = gr.Textbox(label="Plain-language explanation", lines=8, elem_classes="result-box")
                with gr.Accordion("Technical model output", open=False):
                    result_output = gr.Code(label="Structured result", language="json", lines=16)
                clear_btn = gr.Button("Clear results")

    login_btn.click(login, [email, password], [auth_status, auth_view, app_view, session_id, signed_in_as])
    signup_btn.click(create_account, [email, password], auth_status)
    logout_btn.click(logout, [], [auth_status, auth_view, app_view, session_id, signed_in_as])
    ckd_btn.click(run_ckd, [session_id, *ckd_inputs], [result_output, explanation_output, ckd_result_state]).then(
        lambda: "Kidney model complete. Review it on Results or continue to the diabetes model.",
        [], ckd_status
    )
    diabetes_btn.click(run_diabetes, [session_id, ckd_result_state, *diabetes_inputs], [result_output, explanation_output, ckd_result_state]).then(
        lambda: "Diabetes model complete. Open Results to review the explanation.", [], diabetes_status
    )
    skip_ckd_btn.click(skip_ckd, [], [ckd_status, app_view])
    skip_diabetes_btn.click(skip_diabetes, [], [diabetes_status, app_view])
    clear_btn.click(clear_results, [], [result_output, explanation_output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        css=CSS,
        theme=gr.themes.Soft(),
    )
