import io
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from PIL import Image

# PDF generation (local)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from eagt.config import load_config
from eagt.strategy.mapping import map_affect_to_strategy, AFFECT_LABELS
from eagt.genai.generative_engine import GenerativeEngine, GenResponse


# ====================== Page Config & CSS ======================
st.set_page_config(
    page_title="EAGT — Emotion-Aware Generative AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --primary:#1f2937; --muted:#6b7280; --accent:#3b82f6;
        --badge:#eef2ff; --badge-border:#c7d2fe; --badge-text:#1e3a8a;
        --pill:#ecfeff; --pill-border:#a5f3fc; --pill-text:#155e75;
      }
      .eagt-badge {
        display:inline-block; padding:3px 10px; border-radius:10px;
        background:var(--badge); color:var(--badge-text);
        border:1px solid var(--badge-border); font-size:12px; margin-right:6px;
      }
      .eagt-pill {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background:var(--pill); color:var(--pill-text);
        border:1px solid var(--pill-border); font-size:11px; margin-right:6px;
      }
      .eagt-muted { color:var(--muted); font-size:12px; }
      .eagt-divider { border-top:1px dashed #e5e7eb; margin:12px 0 12px 0; }
      .metric-card {
        border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; background:#fafafa;
      }
      .section-card {
        border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; background:#fff;
      }
      .risk-low { color:#065f46; font-weight:600; }
      .risk-med { color:#92400e; font-weight:600; }
      .risk-high { color:#b91c1c; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ====================== Session State ======================
def init_session_state():
    if "chat" not in st.session_state:
        st.session_state.chat: List[dict] = []
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "consent" not in st.session_state:
        st.session_state.consent = False
    if "rag_docs" not in st.session_state:
        st.session_state.rag_docs: List[str] = []
    if "gen_knobs" not in st.session_state:
        st.session_state.gen_knobs = {
            "max_new_tokens": None,
            "temperature": None,
            "top_p": None,
            "use_rag": None,
            "model_name": None,
        }
    if "class_logs" not in st.session_state:
        st.session_state.class_logs: Optional[pd.DataFrame] = None
    if "cohort_name" not in st.session_state:
        st.session_state.cohort_name = "Cohort A"
    if "engage_map" not in st.session_state:
        st.session_state.engage_map = {
            "frustration": 0.40,
            "confusion":  0.55,  # productive struggle can be mid
            "boredom":    0.30,
            "engagement": 0.85,
        }
    # Detector config (local-only)
    if "detector_cfg" not in st.session_state:
        st.session_state.detector_cfg = {
            "enable_local_detectors": False,
            "use_webcam_image": False,
            "use_audio_file": False,
            "local_only": True,  # enforce: nothing leaves device
            "last_cam_affect": None,
            "last_aud_affect": None,
        }


# ====================== Utilities ======================
def heuristic_confidence_from_strategy(style: str, pedagogy: str) -> float:
    score = 0.5
    if "step-by-step" in style or "clear" in style:
        score += 0.1
    if "checks for understanding" in pedagogy or "analogy" in pedagogy:
        score += 0.1
    if "novelty" in pedagogy or "challenge" in pedagogy:
        score += 0.05
    return float(np.clip(score, 0.0, 1.0))


def build_engine(cfg, model_name_override=None, max_new_tokens=None, temperature=None, top_p=None, use_rag=None):
    model_name = model_name_override or cfg.genai.model_name
    return GenerativeEngine(
        model_name=model_name,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.genai.max_new_tokens,
        temperature=temperature if temperature is not None else cfg.genai.temperature,
        top_p=top_p if top_p is not None else cfg.genai.top_p,
        rag_corpus_dir=cfg.genai.rag_corpus_dir,
        use_rag=use_rag if use_rag is not None else cfg.genai.use_rag,
    )


def ensure_engine(cfg, model_name=None):
    must_rebuild = False
    if st.session_state.engine is None:
        must_rebuild = True
    else:
        cur = getattr(st.session_state.engine, "model_name", None)
        if model_name and model_name != cur:
            must_rebuild = True
    if must_rebuild:
        st.session_state.engine = build_engine(
            cfg,
            model_name_override=model_name,
            max_new_tokens=st.session_state.gen_knobs["max_new_tokens"],
            temperature=st.session_state.gen_knobs["temperature"],
            top_p=st.session_state.gen_knobs["top_p"],
            use_rag=st.session_state.gen_knobs["use_rag"],
        )


def ingest_uploaded_docs(files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    tmp_dir = Path(".rag_session")
    tmp_dir.mkdir(exist_ok=True)
    added = 0
    for f in files or []:
        if not f.name.lower().endswith(".txt"):
            continue
        text = f.getvalue().decode("utf-8", errors="ignore")
        out = tmp_dir / f.name
        out.write_text(text, encoding="utf-8")
        if out.as_posix() not in st.session_state.rag_docs:
            st.session_state.rag_docs.append(out.as_posix())
            added += 1
    if added:
        cfg = load_config(CFG_PATH)
        cfg.genai.rag_corpus_dir = ".rag_session"  # type: ignore[attr-defined]
        st.session_state.engine = build_engine(
            cfg,
            model_name_override=getattr(st.session_state.engine, "model_name", None),
            max_new_tokens=st.session_state.gen_knobs["max_new_tokens"],
            temperature=st.session_state.gen_knobs["temperature"],
            top_p=st.session_state.gen_knobs["top_p"],
            use_rag=st.session_state.gen_knobs["use_rag"],
        )
    return added


def add_message(role: str, content: str, meta: dict = None):
    st.session_state.chat.append({"role": role, "content": content, "meta": meta or {}})


def download_transcript_button():
    if not st.session_state.chat:
        return
    buf = io.StringIO()
    for m in st.session_state.chat:
        role = m["role"].upper()
        meta = m.get("meta", {})
        aff = meta.get("affect")
        strat = meta.get("strategy")
        buf.write(f"{role}: {m['content']}\n")
        if aff or strat:
            buf.write(f"  META: affect={aff}, strategy={strat}\n")
        buf.write("\n")
    st.download_button(
        label="💾 Download Transcript",
        data=buf.getvalue(),
        file_name="eagt_transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )


def synthesize_chat_metrics_df(chat: List[dict], cohort: str = "Cohort A", student: str = "Anonymous"):
    rows = []
    t = 0
    for m in chat:
        if m["role"] == "user":
            t += 1
            aff = m.get("meta", {}).get("affect", "engagement")
            eng = st.session_state.engage_map.get(aff, 0.6)
            rows.append(
                {"cohort": cohort, "student": student, "turn": t, "role": "user", "affect": aff, "engagement": eng}
            )
        else:
            meta = m.get("meta", {})
            t += 1
            aff = meta.get("strategy", "—")
            conf = float(meta.get("confidence", 0.5))
            rows.append(
                {"cohort": cohort, "student": student, "turn": t, "role": "assistant",
                 "affect": aff, "engagement": np.nan, "confidence": conf}
            )
    return pd.DataFrame(rows)


def compute_ema(series: pd.Series, span: int = 5) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def risk_level_from_ema(ema_val: float, thr_low: float = 0.45, thr_med: float = 0.6) -> Tuple[str, str]:
    """
    Returns (label, css_class)
    """
    if np.isnan(ema_val):
        return ("—", "risk-low")
    if ema_val < thr_low:
        return ("High", "risk-high")
    elif ema_val < thr_med:
        return ("Medium", "risk-med")
    else:
        return ("Low", "risk-low")


def make_student_report_pdf(student_id: str, cohort: str, df_user: pd.DataFrame) -> bytes:
    """
    Generate a minimal PDF with KPIs + recent affect rows.
    Extend this to embed charts by exporting Altair → PNG (requires selenium or vega-lite export).
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    styles = getSampleStyleSheet()

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, h - 2 * cm, f"EAGT Student Report — {student_id}")
    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, h - 2.7 * cm, f"Cohort: {cohort}")

    # KPIs
    last_n = df_user.tail(5)
    ema = compute_ema(last_n["engagement"].astype(float), span=5).iloc[-1] if not last_n.empty else np.nan
    mean_eng = df_user["engagement"].mean() if "engagement" in df_user else np.nan
    risk, _css = risk_level_from_ema(ema)

    y = h - 3.6 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Key Metrics")
    y -= 0.6 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2.2 * cm, y, f"Mean Engagement: {mean_eng:.2f}" if not np.isnan(mean_eng) else "Mean Engagement: —")
    y -= 0.45 * cm
    c.drawString(2.2 * cm, y, f"EMA (last 5): {ema:.2f}" if not np.isnan(ema) else "EMA (last 5): —")
    y -= 0.45 * cm
    c.drawString(2.2 * cm, y, f"Risk Level: {risk}")

    # Recent rows
    y -= 0.9 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Recent Affective States")
    y -= 0.5 * cm

    # Build table data
    rows = [["Turn/Step", "Affect/Strategy", "Engagement"]]
    for _, r in last_n.iterrows():
        step = r.get("step", r.get("turn", "—"))
        aff = r.get("affect", "—")
        eng = r.get("engagement", "—")
        rows.append([str(step), str(aff), f"{eng:.2f}" if isinstance(eng, (int, float, np.floating)) else "—"])

    table = Table(rows, colWidths=[4*cm, 7*cm, 3*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f3f4f6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#e5e7eb")),
    ]))

    # Draw table
    tw, th = table.wrapOn(c, w - 4*cm, y - 2*cm)
    table.drawOn(c, 2 * cm, y - th)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# ====================== Local Detector Hooks (placeholders) ======================
def infer_affect_from_frame(img: Image.Image) -> str:
    """
    visual affect inference — purely local and simple.
    Heuristic: darker frames → confusion/frustration; brighter → engagement; mid → boredom.
    Replace with your CNN/LSTM pipeline.
    """
    gray = img.convert("L")
    mean = np.asarray(gray).mean()
    if mean < 60:
        return "frustration"
    elif mean < 110:
        return "confusion"
    elif mean < 170:
        return "boredom"
    else:
        return "engagement"


def infer_affect_from_audio(file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """
    audio affect inference — purely local and simple.
    Heuristic: average absolute amplitude → low (boredom), mid (confusion), high (engagement/frustration).
    Replace with wav2vec2 or a proper model.
    """
    try:
        data = file.getvalue()
        import soundfile as sf
        import io as _io
        wav, sr = sf.read(_io.BytesIO(data))
        amp = float(np.mean(np.abs(wav)))
        if amp < 0.02:
            return "boredom"
        elif amp < 0.05:
            return "confusion"
        elif amp < 0.09:
            return "engagement"
        else:
            return "frustration"
    except Exception:
        # Fallback if soundfile not available
        return "confusion"


# ====================== Load Config & Initialize ======================
CFG_PATH = "configs/default.yaml"
cfg = load_config(CFG_PATH)
init_session_state()


# ====================== Sidebar Controls ======================
with st.sidebar:
    st.header("⚙️ Controls")

    with st.expander("Privacy & Ethics (required) ⚖️", expanded=True):
        st.markdown(
            """
**Datasets** — Config paths use DAiSEE/SEMAINE examples; replace with your corpora.  
**LLM Size** — Default **GPT-2 (117M)** for demo; swap to larger instruction-tuned LLMs for research.  
**Privacy** — Webcam/mic are biometric. Use **consent**, **data minimization**, **transparency**.
            """
        )
        st.session_state.consent = st.checkbox("I agree to the Privacy & Ethics statement.", value=st.session_state.consent)

    st.markdown("---")

    st.subheader("Generation")
    model_name = st.selectbox(
        "Language Model",
        options=["gpt2", "meta-llama/Llama-2-7b-chat-hf (requires setup)", "mistralai/Mistral-7B-Instruct (requires setup)"],
        index=0,
        help="Local HF models; larger models require setup.",
    )
    max_new = st.slider("Max New Tokens", 64, 512, cfg.genai.max_new_tokens, 16)
    temp = st.slider("Temperature", 0.1, 1.5, cfg.genai.temperature, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, cfg.genai.top_p, 0.05)
    use_rag = st.toggle("Enable Retrieval-Augmentation (RAG)", value=cfg.genai.use_rag)

    st.session_state.gen_knobs.update(
        dict(model_name=model_name, max_new_tokens=max_new, temperature=temp, top_p=top_p, use_rag=use_rag)
    )

    st.markdown("---")
    st.subheader("RAG Notes")
    st.caption("Upload small `.txt` (definitions, theorems, worked examples).")
    files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    if files:
        if not st.session_state.consent:
            st.warning("Please accept the Privacy & Ethics statement before uploading.")
        else:
            n = ingest_uploaded_docs(files)
            if n:
                st.success(f"Added {n} file(s). RAG re-indexed.")

    st.markdown("---")
    st.subheader("Session")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.toast("Chat history cleared.", icon="🧹")

# Ensure engine for current model (fallback to gpt2 for demo)
ensure_engine(cfg, model_name="gpt2")


# ====================== Top Title ======================
st.title("🎓 Emotion-Aware Generative AI Tutor (EAGT)")
st.caption("Multimodal affect detection → strategy mapping → affect-sensitive generation.")


# ====================== Tabs ======================
tab_tutor, tab_teacher, tab_settings = st.tabs(["👩‍🎓 Tutor", "🧑‍🏫 Teacher Dashboard", "🧰 Settings & Docs"])


# ====================== TAB 1: Tutor ======================
with tab_tutor:
    left, right = st.columns([0.54, 0.46])

    with left:
        st.subheader("Ask the Tutor")
        with st.expander("🧪 Example Questions", expanded=False):
            st.markdown(
                """
- “I don’t understand the **chain rule** with nested functions.”  
- “Why is **variance** the average of squared deviations?”  
- “How do I debug a **binary search** that loops forever?”  
- “What is the intuition behind **eigenvectors** in PCA?”
                """
            )

        # Local detector controls
        det = st.session_state.detector_cfg
        with st.expander("🔒 Local Detectors (Optional)"):
            st.caption("Runs locally. No frames or audio leave your machine.")
            det["enable_local_detectors"] = st.toggle("Enable Local Detectors", value=det["enable_local_detectors"])
            if det["enable_local_detectors"]:
                det["use_webcam_image"] = st.toggle("Use Webcam Frame (still image)")
                det["use_audio_file"] = st.toggle("Use Audio File (WAV/MP3)")

                if det["use_webcam_image"]:
                    cam_img = st.camera_input("Capture a frame")
                    if cam_img is not None:
                        img = Image.open(cam_img)
                        aff_cam = infer_affect_from_frame(img)
                        det["last_cam_affect"] = aff_cam
                        st.success(f"Webcam Frame Affect (local): **{aff_cam}**")

                if det["use_audio_file"]:
                    aud_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])
                    if aud_file is not None:
                        aff_aud = infer_affect_from_audio(aud_file)
                        det["last_aud_affect"] = aff_aud
                        st.success(f"Audio Affect (local): **{aff_aud}**")

        # Affect picker (overrides: local detectors if available)
        affect_source = "manual"
        affect_label = st.selectbox("Affective State", AFFECT_LABELS, index=1)
        if det["enable_local_detectors"]:
            # Prefer audio, then camera, else manual
            if det.get("last_aud_affect"):
                affect_label = det["last_aud_affect"]
                affect_source = "local-audio"
            elif det.get("last_cam_affect"):
                affect_label = det["last_cam_affect"]
                affect_source = "local-camera"

        # Strategy preview
        strat_preview = st.empty()
        question = st.text_area("Your Question", placeholder="E.g., I don’t understand the chain rule.", height=140)

        _strat = map_affect_to_strategy(affect_label)
        strat_preview.markdown(
            f"""
            <span class="eagt-badge">{_strat.name}</span>
            <span class="eagt-pill">{_strat.style}</span>
            <span class="eagt-pill">{_strat.pedagogy.split(';')[0][:64]}…</span>
            """,
            unsafe_allow_html=True,
        )

        gc1, gc2 = st.columns([0.4, 0.6])
        with gc1:
            run_btn = st.button("🧠 Generate Response", use_container_width=True)
        with gc2:
            download_transcript_button()

        if run_btn:
            if not st.session_state.consent:
                st.error("Consent required. Please confirm Privacy & Ethics.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                add_message("user", question, {"affect": affect_label, "source": affect_source})
                strat = map_affect_to_strategy(affect_label)

                with st.spinner("Thinking…"):
                    t0 = time.time()
                    resp: GenResponse = st.session_state.engine.generate(
                        question, strat.style, strat.pedagogy
                    )
                    dt = time.time() - t0

                add_message(
                    "assistant",
                    resp.text or "_(No output — check model config.)_",
                    {
                        "strategy": strat.name,
                        "style": strat.style,
                        "pedagogy": strat.pedagogy,
                        "latency_sec": round(dt, 2),
                        "retrieved": resp.retrieved or [],
                        "confidence": round(heuristic_confidence_from_strategy(strat.style, strat.pedagogy), 2),
                    },
                )
                st.toast(f"Response in {dt:.2f}s", icon="✅")

    with right:
        st.subheader("Dialogue")
        if not st.session_state.chat:
            st.info("No messages yet. Ask a question to begin.")
        else:
            for m in st.session_state.chat:
                role = m["role"]
                meta = m.get("meta", {})
                if role == "user":
                    with st.chat_message("user", avatar="🧑‍🎓"):
                        aff = meta.get("affect")
                        src = meta.get("source", "manual")
                        if aff:
                            st.markdown(
                                f"<span class='eagt-badge'>Affect: {aff} ({src})</span>",
                                unsafe_allow_html=True,
                            )
                        st.write(m["content"])
                else:
                    with st.chat_message("assistant", avatar="🧑‍🏫"):
                        strat = meta.get("strategy", "—")
                        style = meta.get("style", "—")
                        latency = meta.get("latency_sec", "—")
                        conf = meta.get("confidence", 0.5)
                        st.markdown(
                            f"""
                            <span class="eagt-badge">{strat}</span>
                            <span class="eagt-pill">style: {style}</span>
                            <span class="eagt-pill">latency: {latency}s</span>
                            <span class="eagt-pill">confidence: {int(conf*100)}%</span>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("<div class='eagt-divider'></div>", unsafe_allow_html=True)
                        st.write(m["content"])
                        if meta.get("retrieved"):
                            with st.expander("📚 Supporting Notes (RAG)"):
                                for d in meta["retrieved"]:
                                    st.markdown(f"- {d}")

    st.markdown(
        "<div class='eagt-divider'></div><div class='eagt-muted'>Note: Local detectors here are placeholders. Replace with your actual models for production.</div>",
        unsafe_allow_html=True,
    )


# ====================== TAB 2: Teacher Dashboard ======================
with tab_teacher:
    st.subheader("Aggregated Analytics")
    st.caption("Upload class logs or analyze the current session. Engagement is synthesized from affect states for demo.")

    col_u, col_opts = st.columns([0.45, 0.55])
    with col_u:
        uploaded = st.file_uploader("Upload Class Logs (CSV)", type=["csv"],
                                    help="Schema: student_id, timestamp, affect, [task], [duration_s], [score]")
        if uploaded:
            try:
                df_logs = pd.read_csv(uploaded)
                # Normalize
                needed = {"student_id", "timestamp", "affect"}
                if not needed.issubset(df_logs.columns):
                    raise ValueError("CSV must include at least: student_id, timestamp, affect")
                df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")
                df_logs["student"] = df_logs["student_id"].astype(str)
                df_logs["engagement"] = df_logs["affect"].str.lower().map(st.session_state.engage_map).fillna(0.6)
                df_logs = df_logs.sort_values(["student", "timestamp"])
                df_logs["session_idx"] = df_logs.groupby("student").cumcount() + 1
                df_logs["cohort"] = st.session_state.cohort_name
                st.session_state.class_logs = df_logs
                st.success(f"Loaded {len(df_logs)} rows from {uploaded.name}")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    with col_opts:
        st.text_input("Cohort Name", value=st.session_state.cohort_name, key="cohort_name")
        source = st.radio("Data Source", ["Current Session", "Uploaded CSV"], horizontal=True)

    if source == "Current Session":
        data = synthesize_chat_metrics_df(st.session_state.chat, cohort=st.session_state.cohort_name)
        if data.empty:
            st.info("No session data yet. Ask the tutor to generate some dialogue.")
    else:
        data = st.session_state.class_logs
        if data is None or data.empty:
            st.warning("Please upload a CSV log first.")
            data = pd.DataFrame()

    # ------ Student List Panel with Risk Flags ------
    st.markdown("### 👥 Student List & Risk Flags")
    if not data.empty:
        if "student" not in data.columns:
            data["student"] = "Anonymous"

        # Build per-student EMA over last 5 user steps
        df_user = data[data.get("role", "user") == "user"] if "role" in data.columns else data.copy()
        # unify 'step'
        if "turn" in df_user.columns:
            df_user["step"] = df_user["turn"]
        else:
            df_user["step"] = df_user.get("session_idx", pd.Series(np.arange(1, len(df_user) + 1), index=df_user.index))

        panel_rows = []
        for sid, grp in df_user.groupby("student"):
            g = grp.sort_values("step")
            last5 = g.tail(5)
            ema = compute_ema(last5["engagement"].astype(float), span=5).iloc[-1] if not last5.empty else np.nan
            risk, css = risk_level_from_ema(ema)
            panel_rows.append({"student": sid, "steps": int(g["step"].max()), "ema_last5": ema, "risk": risk, "css": css})

        panel_df = pd.DataFrame(panel_rows).sort_values(["risk", "ema_last5"], ascending=[True, False])
        # Display with colored risk text
        st.dataframe(
            panel_df[["student", "steps", "ema_last5", "risk"]]
            .rename(columns={"ema_last5": "EMA(last5)"}),
            use_container_width=True,
        )

        # Select a student for drill-down
        student_pick = st.selectbox("Drill-down Student", options=panel_df["student"].tolist())
        dsel = data[data["student"] == student_pick].copy()
        # Turn/step
        if "turn" in dsel.columns:
            dsel["step"] = dsel["turn"]
        else:
            dsel["step"] = dsel.get("session_idx", np.arange(1, len(dsel) + 1))

        # KPI cards
        col_k1, col_k2, col_k3 = st.columns(3)
        mean_eng = dsel.loc[dsel.get("role", "user") == "user", "engagement"].mean() if "role" in dsel.columns else dsel["engagement"].mean()
        last5 = dsel.loc[dsel.get("role", "user") == "user"].tail(5) if "role" in dsel.columns else dsel.tail(5)
        ema = compute_ema(last5["engagement"].astype(float), span=5).iloc[-1] if not last5.empty else np.nan
        risk, css = risk_level_from_ema(ema)

        col_k1.markdown(f"<div class='metric-card'><b>Student</b><br><span class='eagt-muted'>{student_pick}</span></div>", unsafe_allow_html=True)
        col_k2.markdown(f"<div class='metric-card'><b>Mean Engagement</b><br><span class='eagt-muted'>{mean_eng:.2f if not np.isnan(mean_eng) else '—'}</span></div>", unsafe_allow_html=True)
        col_k3.markdown(f"<div class='metric-card'><b>Risk (EMA last5)</b><br><span class='{ 'risk-high' if risk=='High' else 'risk-med' if risk=='Medium' else 'risk-low'}'>{risk}</span></div>", unsafe_allow_html=True)

        # Charts: trendline + affect distribution
        c1, c2 = st.columns([0.58, 0.42])

        with c1:
            st.markdown("**Engagement Trendline**")
            dplot = dsel.copy()
            if "role" in dplot.columns:
                dplot = dplot[dplot["role"] == "user"]
            dplot["ema"] = dplot.groupby("student")["engagement"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
            if not dplot.empty:
                chart = (
                    alt.Chart(dplot)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("step:Q", title="Step"),
                        y=alt.Y("ema:Q", title="Engagement (EMA)"),
                        tooltip=["step", "affect", alt.Tooltip("ema:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Not enough engagement data to plot.")

        with c2:
            st.markdown("**Affect Distribution**")
            if "affect" in dsel.columns:
                d_aff = dsel.copy()
                if "role" in d_aff.columns:
                    d_aff = d_aff[d_aff["role"] == "user"]
                counts = d_aff["affect"].value_counts().reset_index()
                counts.columns = ["affect", "count"]
                if not counts.empty:
                    bar = (
                        alt.Chart(counts)
                        .mark_bar()
                        .encode(x=alt.X("affect:N", title="Affect"), y=alt.Y("count:Q", title="Count"), tooltip=["affect", "count"])
                        .properties(height=280)
                    )
                    st.altair_chart(bar, use_container_width=True)
                else:
                    st.info("No affect data for this student.")

        st.markdown("<div class='eagt-divider'></div>", unsafe_allow_html=True)

        # ---- Drill-down PDF report ----
        st.markdown("**Drill-down Report (PDF)**")
        # Build a user-only frame for report (engagement rows)
        df_user_only = dsel[dsel.get("role", "user") == "user"] if "role" in dsel.columns else dsel.copy()
        if df_user_only.empty:
            st.info("No user rows available for this student.")
        else:
            pdf_bytes = make_student_report_pdf(student_pick, st.session_state.cohort_name, df_user_only)
            st.download_button(
                "⬇️ Download Student Report (PDF)",
                data=pdf_bytes,
                file_name=f"eagt_report_{student_pick.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
    else:
        st.info("No data to analyze yet.")


# ====================== TAB 3: Settings & Docs ======================
with tab_settings:
    st.subheader("Settings & Documentation")
    st.markdown(
        """
**Datasets** — Replace config paths in `configs/` with **DAiSEE** or **SEMAINE** (or your classroom data).  
**Language Model Size** — Default **GPT-2** is a small demo model; swap with larger instruction-tuned LLMs.  
**Privacy & Ethics** — Affect sensing is biometric. Use **informed consent**, **data minimization**, **transparency**; consider **on-device inference**, **federated learning**, **fairness audits**.
        """
    )
    st.markdown("<div class='eagt-divider'></div>", unsafe_allow_html=True)

    st.markdown("**Live Config (read-only)**")
    CFG_PATH = "configs/default.yaml"
    cfg = load_config(CFG_PATH)
    cfg_view = {
        "data": cfg.data.__dict__,
        "preprocess": cfg.preprocess.__dict__,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
        "genai": cfg.genai.__dict__,
        "server": cfg.server.__dict__,
    }
    st.json(cfg_view)

    st.markdown("<div class='eagt-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """
**How this demo works**  
1) You pick (or detect locally) an **affective state**.  
2) We map it to a **pedagogical strategy** and **tone**.  
3) The LLM generates an **affect-sensitive explanation**, optionally augmented with **RAG** from notes.  

**Next steps**  
- Plug in your multimodal detector (vision/audio/behavior).  
- Swap the LLM and fine-tune for educational style control.  
- Feed real telemetry into the Teacher Dashboard for live monitoring.
        """
    )
