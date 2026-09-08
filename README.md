---
title: Patient Health Insights
sdk: gradio
app_file: app.py
---

# Patient Health Insights

Integrated ML + SHAP + orchestrator + LLM explanation system, per the
project proposal. Users create an account or log in with an email and
password stored in SQLite. The app guides them through separate CKD and
diabetes model pages before showing explainable-AI results on a Results page.

## Folder structure

```
ckd_diabetes_app/
├── app.py                # Gradio UI + wiring — HF Spaces entry point
├── orchestrator.py        # Central coordination layer
├── model_registry.py      # Loads all saved model artifacts once
├── shap_engine.py          # SHAP explanations (patient-specific + global)
├── memory_store.py         # SQLite-based lightweight memory
├── auth_store.py           # SQLite account storage with hashed passwords
├── llm_engine.py           # LLM interpretation layer (+ template fallback)
├── requirements.txt
├── README.md
└── models/                 # PUT YOUR SAVED .pkl FILES HERE (see below)
    ├── ckd_random_forest_model.pkl
    ├── ckd_num_imputer.pkl
    ├── ckd_cat_imputer.pkl
    ├── ckd_label_encoders.pkl
    ├── ckd_feature_order.pkl
    ├── diabetes_xgboost_model.pkl
    ├── diabetes_feature_order.pkl
    ├── ckd_background_sample.pkl        (optional, improves SHAP baseline)
    └── diabetes_background_sample.pkl   (optional, improves SHAP baseline)
```

## Step 1 — Copy your trained models into `models/`

From your Colab notebook, after training, save a small background
sample too (used for better SHAP baselines):

```python
import joblib
joblib.dump(X_train.sample(min(100, len(X_train)), random_state=42),
            'ckd_background_sample.pkl')
joblib.dump(X_train.sample(min(100, len(X_train)), random_state=42),
            'diabetes_background_sample.pkl')
```

Then download all `.pkl` files from Colab and place them inside this
project's `models/` folder before deploying.

## Step 2 — Test locally (recommended before deploying)

```bash
pip install -r requirements.txt
python app.py
```

This launches a local Gradio server (usually http://127.0.0.1:7860).
The app starts in deterministic template mode by default, so it works
without downloading a large language model. To enable the optional LLM
locally, set `CKD_USE_LLM_MODEL=true` before starting the app.

## Step 3 — Deploy to Render

This repository includes `render.yaml`. In Render, choose **New > Blueprint**
and connect the GitHub repository. Render will install the dependencies and
run `python app.py`; the app reads Render's `PORT` environment variable.

SQLite is local to the running service. For a production deployment, attach a
persistent Render disk or move account and prediction storage to a managed database.

## Step 4 — Deploy to Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Create a new Space, choose **Gradio**, and choose **CPU basic** hardware.
3. Do not add a README or `.gitignore` when creating the Space; this project already contains them.
4. From this project folder, run:

```bash
git init
git branch -M main
git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
git add .
git commit -m "Deploy patient health insights app"
git push
```

When Git asks for a password, use a Hugging Face access token with
write permission. The Space will install `requirements.txt` and run
`app.py` automatically. The model files in `models/` are included in
the deployment; `memory.db` is intentionally excluded because the Space
creates its own runtime database.

### Optional: enable the LLM explanation layer

After the Space starts, open **Settings > Variables and secrets > New variable**
and add:

```text
Name: CKD_USE_LLM_MODEL
Value: true
```

The LLM downloads on first boot and may need more memory than the free CPU
tier. Template explanations are the recommended default for a reliable demo.

### If the free CPU tier struggles with the LLM
`Qwen/Qwen2.5-1.5B-Instruct` is small enough for most free CPU Spaces,
but if you hit memory/timeout issues:
- Remove or set `CKD_USE_LLM_MODEL=false` in the Space variables to run in
  template-only mode (still fully functional — it just skips free-text
  generation and uses the deterministic, evidence-grounded explanation)
- Or switch `MODEL_NAME` in `llm_engine.py` to an even smaller model,
  e.g. `Qwen/Qwen2.5-0.5B-Instruct`

## Notes on scope (per proposal Sections 5.2, 10, 23)
- The LLM only phrases evidence already computed by SHAP — it is not
  asked to add medical knowledge or produce a diagnosis.
- The memory store is session-scoped structured evidence only, not a
  medical record.
- This system is a research/educational prototype and is explicitly
  out of scope for clinical use without external validation.
