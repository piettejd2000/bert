# BERT MI Analysis — Cloud Deploy: Session Handoff (v2 — DEPLOYED)

*Resume snapshot.*
**STATUS: LIVE.** The BERT Motivational-Interviewing analysis app (v1.0) is deployed and public on Streamlit Community Cloud, loading both models from HuggingFace Hub. The build is complete; what remains is optional polish and unrelated cleanup.

- **Live app:** https://9dnbb24oe5fhqeuqeurucn.streamlit.app *(auto subdomain — can be renamed in app Settings → General)*
- **GitHub:** https://github.com/piettejd2000/bert *(public)*
- **Models on HF Hub:** `piettejd2000/mi-behavior-roberta`, `piettejd2000/mi-global-roberta` *(public)*

---

## 1. The Goal (achieved)
Move the latest functional version of the BERT/RoBERTa MI transcript-analysis app from the local research tree into a clean Dropbox/GitHub repo, host the models somewhere that fits within GitHub limits, and deploy as a public Streamlit app. Same overall pattern as the `rag_mi_cbt` project; the key divergence was hosting the models on HuggingFace Hub instead of in the repo.

---

## 2. What the App Does
A **Streamlit dashboard** (`mi_results_dashboard.py`) that ingests an MI therapy transcript and scores it with **two separately-trained RoBERTa models**:
- **`behavior_model`** — individual MI behavior classification (per utterance).
- **`global_model`** — global MI dimension scoring.

It produces JSON / CSV / PDF reports (PDF via `reportlab`).

---

## 3. Key Locations

**Live / cloud:**
- App: https://9dnbb24oe5fhqeuqeurucn.streamlit.app
- GitHub repo: `github.com/piettejd2000/bert` (public), branch `main`, main file `v1.0/mi_results_dashboard.py`, Python **3.11**.
- HF model repos (public): `piettejd2000/mi-behavior-roberta` (standard HF dir), `piettejd2000/mi-global-roberta` (holds `global_model.pt` + config JSONs).

**Deploy repo (local, in Dropbox for backup):** `~/Dropbox/GitHub/bert/`
- `v1.0/` — the app: `mi_results_dashboard.py`, `analyze_global_dimensions_sliding_window.py`, `analyze_global_dimensions_json.py`, `mi_excellence_booster.py`, `requirements.txt`. (Plus local `*_OLD.py` backups — gitignored, not deployed.)
- `models_shared/` — local copies of `behavior_model/` + `global_model/` (~950 MB). **Gitignored — backup only; the cloud uses HF, not these.**
- `.gitignore`, `.streamlit/config.toml` (root).

**Source (research tree — DO NOT `rm`):** `~/healthcare_rl/mi_prototype/trained_models/bert/`
- Live local models, plus `training_archive/` (the renamed `Google Drive?` folder = Colab training/reproducibility bundle + README).
- This whole tree is *untracked* inside the grandparent git repo at `~/healthcare_rl/mi_prototype/` (origin = `rag_mi_cbt_v2.0`).

---

## 4. Deployed Architecture (how it loads models)
Both analyzers load from HuggingFace Hub, so the same code runs locally and on the cloud (downloads + caches by repo ID):
- **behavior_model** → `AutoModelForSequenceClassification.from_pretrained("piettejd2000/mi-behavior-roberta")`.
- **global_model** → `hf_hub_download("piettejd2000/mi-global-roberta", "global_model.pt")` → `torch.load` into the custom `RobertaForGlobalDimensions` class (wraps `roberta-base`).
- Constants `BEHAVIOR_MODEL_REPO` / `GLOBAL_MODEL_REPO` at the top of each analyzer.

**In-memory caching (the speed-up):** model loading is memoized at module level in each analyzer — `_BEHAVIOR_MODEL_CACHE` / `_GLOBAL_MODEL_CACHE`, with `load_behavior_model()` wrapping `_load_behavior_model_uncached()` and a `load_global_model(device)` function. The module persists across Streamlit reruns, so models load **once per session**; only the first analysis after a cold start pays the load cost.

---

## 5. Completed This Build

**Local cleanup (research `bert/` dir):**
- Deleted `Google Drive?/trained_model` (477 MB) — MD5-confirmed duplicate of `behavior_model`.
- Renamed `Google Drive?` → `training_archive`, added a README; kept `MI_RoBERTa_base/` (955 MB) as the reproducibility bundle (its inner models duplicate the live ones; only `data/` + notebooks are unique).
- Removed cruft (`__pycache__/`, `batch_output.log`, `.DS_Store`).

**Model hosting:** uploaded both models to HF Hub via the `hf` CLI (`hf upload …`); verified file lists with `list_repo_files`.

**Code edits (cloud-portability + speed), all syntax-checked:**
- Analyzers: HF-based loading + module-level model caching (see §4).
- Dashboard: `APP_DIR = Path(__file__).resolve().parent` replacing the hardcoded `sys.path.append`; JSON output redirected to `tempfile.gettempdir()/mi_analysis_output`; subprocess fallback + `.python_config` made relative to `APP_DIR` (cloud-safe); "example transcript" picker repointed to a relative `v1.0/examples/` folder (empty unless populated); removed redundant in-function `import tempfile`/`subprocess` that caused an `UnboundLocalError`.
- `requirements.txt`: CPU-only torch via `--extra-index-url https://download.pytorch.org/whl/cpu`, plus `transformers==4.53.3`, `huggingface_hub<1.0`, `streamlit`, `pandas`, `numpy`, `matplotlib`, `reportlab`.

**Repo plumbing & deploy:**
- Root `.gitignore` (ignores `models_shared/`, model files, `.venv/`, caches, secrets, `*_OLD.py`).
- Root `.streamlit/config.toml` (light theme, telemetry off).
- `git init` → committed 7 files (no models, no secrets, no OLD backups) → pushed to public `github.com/piettejd2000/bert`.
- Deployed on Streamlit Community Cloud (main file `v1.0/mi_results_dashboard.py`, Python 3.11, no secrets needed). **Verified working** — analyzes a transcript end-to-end, models pull from HF on first run.

---

## 6. Open Items (optional / not blocking)
1. **Custom subdomain** — rename the cryptic auto URL in app **Settings → General** (e.g. `mi-bert-analysis.streamlit.app`) before sharing widely.
2. **Cold-start expectation** — the free tier sleeps when idle; the first visitor after a sleep triggers a re-download + load of the models (~1–2 min) before caching kicks in. Warn anyone you share the link with.
3. **Bundle example transcripts** (optional) — drop a few `.txt` files into `v1.0/examples/` and push, to make the "Select Example Transcript" picker work on the cloud.
4. **Trim the archive** (optional) — `training_archive/MI_RoBERTa_base/` holds duplicate model copies; deleting those inner model dirs reclaims ~950 MB locally (keep `data/` + notebooks).
5. **Future v1.1 / v1.2** — UI-only versions go in new `v1.x/` subfolders and reuse the same HF repos (no new model uploads needed).

**Separate, unrelated to the deploy:** the `~/healthcare_rl/mi_prototype/transcripts/` recovery (~1,000 transcripts deleted on **June 10**, predating this session). Recovery in progress via **Dropbox Rewind** to just before the June-10 deletion. Git is not the path — those files were untracked (only 55 were ever committed).

---

## 7. Risks / Gotchas
- **Memory** — `behavior_model` + `global_model` + `roberta-base` ≈ ~1 GB loaded together. It deployed and runs on the free tier, so it currently fits, but it's near the ceiling; watch logs if behavior changes.
- **HF dependency** — the app needs the two HF repos to stay public and present; don't delete or privatize them.
- **Cold start** re-downloads models (cache is ephemeral across restarts) — expected, not a bug.

---

## 8. Critical Rules
- ⚠️ Do NOT `rm` the research tree `~/healthcare_rl/mi_prototype/trained_models/bert/` (holds `training_archive`).
- ⚠️ NEVER commit large model files — they live on HuggingFace Hub; `models_shared/` is gitignored.
- ⚠️ NEVER commit the HF token or any secret.
- ⚠️ Don't delete/privatize the HF model repos the live app depends on.
- ⚠️ Don't touch the live RAG deployments.
- Verify changes by RUNNING the app (one full transcript analysis), not by static analysis.

---

## 9. Quick Reference
```
Live app:        https://9dnbb24oe5fhqeuqeurucn.streamlit.app
GitHub:          github.com/piettejd2000/bert  (main, file: v1.0/mi_results_dashboard.py, Py 3.11)
HF models:       piettejd2000/mi-behavior-roberta , piettejd2000/mi-global-roberta
Deploy repo:     ~/Dropbox/GitHub/bert/  (app in v1.0/, models in models_shared/ [gitignored])
Source (keep):   ~/healthcare_rl/mi_prototype/trained_models/bert/
```
