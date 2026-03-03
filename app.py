"""
Streamlit Web Application for ECG Classification
Classification of Life-Threatening Arrhythmia ECG Signals Using Deep Learning

Author: Mohamad AlJasem
Website: https://aljasem.eu.org
GitHub: https://github.com/m-aljasem/ecg-arrhythmia-classifier-AI
"""

import os
import pathlib
import traceback
import tempfile
import pickle

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import wfdb

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#0a0e1a; --surface:#111827; --card:#161f2e; --border:#1e2d45;
    --accent:#00d4aa; --accent2:#0ea5e9; --danger:#f43f5e; --warn:#f59e0b;
    --text:#e2e8f0; --muted:#64748b; --radius:12px;
    --font-body:'IBM Plex Sans',sans-serif; --font-mono:'IBM Plex Mono',monospace;
}
html,body,[class*="css"]{font-family:var(--font-body)!important;background-color:var(--bg)!important;color:var(--text)!important;}
.main .block-container{padding:2rem 2.5rem!important;max-width:1400px;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.ecg-header{background:linear-gradient(135deg,#0d1b2e,#0a1628,#061020);border:1px solid var(--border);border-radius:var(--radius);padding:2rem 2.5rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.ecg-header::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(0,212,170,.06),transparent 60%),radial-gradient(ellipse at 80% 50%,rgba(14,165,233,.06),transparent 60%);pointer-events:none;}
.ecg-header h1{font-size:2rem!important;font-weight:600!important;color:var(--text)!important;margin:0 0 .4rem!important;letter-spacing:-.02em;}
.ecg-header .subtitle{color:var(--muted);font-size:.95rem;font-weight:300;}
.badge-row{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem;}
.badge{font-family:var(--font-mono);font-size:.75rem;padding:.25rem .75rem;border-radius:999px;border:1px solid;}
.badge-norm{background:rgba(0,212,170,.12);border-color:rgba(0,212,170,.3);color:#00d4aa;}
.badge-mi{background:rgba(244,63,94,.12);border-color:rgba(244,63,94,.3);color:#f43f5e;}
.badge-sttc{background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.3);color:#f59e0b;}
.badge-cd{background:rgba(14,165,233,.12);border-color:rgba(14,165,233,.3);color:#0ea5e9;}
.badge-hyp{background:rgba(167,139,250,.12);border-color:rgba(167,139,250,.3);color:#a78bfa;}
.section-title{font-size:.8rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:1.25rem;display:flex;align-items:center;gap:.5rem;}
.section-title::after{content:'';flex:1;height:1px;background:var(--border);}
.stat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.75rem;}
.stat-tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;}
.stat-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.stat-value{font-size:1rem;font-weight:600;color:var(--text);font-family:var(--font-mono);margin-top:.15rem;}
.diag-row{display:flex;align-items:center;gap:.75rem;padding:.6rem .75rem;border-radius:8px;margin-bottom:.4rem;border:1px solid var(--border);background:var(--surface);}
.diag-row.positive{border-color:rgba(244,63,94,.4);background:rgba(244,63,94,.06);}
.diag-label{flex:1;font-size:.88rem;}
.diag-code{font-family:var(--font-mono);font-size:.75rem;color:var(--muted);min-width:38px;}
.diag-pct{font-family:var(--font-mono);font-size:.9rem;font-weight:600;min-width:48px;text-align:right;}
.diag-bar-wrap{flex:1.5;height:6px;background:var(--border);border-radius:3px;overflow:hidden;}
.diag-bar{height:100%;border-radius:3px;}
.status-pill{font-size:.7rem;font-family:var(--font-mono);padding:.15rem .5rem;border-radius:999px;white-space:nowrap;}
.pill-pos{background:rgba(244,63,94,.2);color:#f43f5e;border:1px solid rgba(244,63,94,.4);}
.pill-neg{background:rgba(0,212,170,.12);color:#00d4aa;border:1px solid rgba(0,212,170,.3);}
.status-ok{color:#00d4aa!important;}
.status-err{color:#f43f5e!important;}
hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0ea5e9)!important;color:#0a0e1a!important;border:none!important;font-weight:600!important;border-radius:8px!important;padding:.6rem 2rem!important;font-family:var(--font-body)!important;font-size:.95rem!important;letter-spacing:.01em!important;}
.stButton>button:hover{opacity:.88!important;}
.stNumberInput input,.stSelectbox select,.stTextInput input{background:var(--surface)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-family:var(--font-mono)!important;}
[data-testid="stFileUploader"]{background:var(--surface)!important;border:1px dashed var(--border)!important;border-radius:var(--radius)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST PATH RESOLUTION
# Strategy: walk up from both __file__ and cwd looking for models/model03.keras.
# This handles:
#   - Local runs from any directory
#   - Streamlit Cloud (clones repo to /mount/src/<repo>, sets cwd = repo root)
#   - Editable installs or unusual layouts
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd        = os.getcwd()

    candidates = [script_dir, cwd]
    # Walk up parent directories from both starting points
    for base in (script_dir, cwd):
        for parent in pathlib.Path(base).parents:
            candidates.append(str(parent))

    # Deduplicate while preserving order
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    for root in unique:
        if os.path.isfile(os.path.join(root, "models", "model03.keras")):
            return root

    # Nothing found — fall back to cwd; errors will surface in the UI
    return cwd


_HERE        = _find_repo_root()
MODELS_DIR   = os.path.join(_HERE, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "model03.keras")
SCALERS_PATH = os.path.join(MODELS_DIR, "scalers.pkl")
SAMPLE_DIR   = os.path.join(_HERE, "sample-data")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_DESCRIPTIONS = {
    'NORM': 'Normal ECG',
    'MI':   'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD':   'Conduction Disturbance',
    'HYP':  'Hypertrophy',
}
CLASS_COLORS = {
    'NORM': '#00d4aa',
    'MI':   '#f43f5e',
    'STTC': '#f59e0b',
    'CD':   '#0ea5e9',
    'HYP':  '#a78bfa',
}
LEAD_NAMES  = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
WINDOW_SIZE = 800

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    try:
        # compile=False avoids optimizer config mismatches across TF versions
        return keras.models.load_model(path, compile=False)
    except Exception:
        st.error("Model load failed — see traceback below.")
        st.code(traceback.format_exc(), language="text")
        return None


@st.cache_resource
def load_scalers(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        st.error("Scalers load failed — see traceback below.")
        st.code(traceback.format_exc(), language="text")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE FILE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_sample_files():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    bases = set()
    for f in os.listdir(SAMPLE_DIR):
        name, ext = os.path.splitext(f)
        if ext.lower() in ('.hea', '.dat'):
            bases.add(name)
    return sorted(bases)

# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_metadata(age, sex, height, weight, inf1, inf2, pacemaker, x_scaler):
    meta = pd.DataFrame({
        'age':                 [float(age) if age else 0.0],
        'sex':                 [float(sex)],
        'height':              [float(height) if height and height >= 50 else 0.0],
        'weight':              [float(weight) if weight else 0.0],
        'infarction_stadium1': [float(inf1)],
        'infarction_stadium2': [float(inf2)],
        'pacemaker':           [float(pacemaker)],
    })
    if x_scaler is not None:
        return x_scaler.transform(meta.values)
    return meta.values


def preprocess_ecg_signal(ecg, y_scaler, window_size=800):
    if ecg.shape[0] > window_size:
        start = (ecg.shape[0] - window_size) // 2
        ecg = ecg[start:start + window_size, :]
    elif ecg.shape[0] < window_size:
        ecg = np.pad(ecg, ((0, window_size - ecg.shape[0]), (0, 0)), mode='edge')

    ecg = ecg.reshape(1, ecg.shape[0], ecg.shape[1])

    if y_scaler is not None:
        shape = ecg.shape
        ecg = y_scaler.transform(ecg.reshape(-1, shape[-1])).reshape(shape)

    return ecg.astype('float32')

# ─────────────────────────────────────────────────────────────────────────────
# WFDB LOADER  — fixes embedded record-name mismatch on uploaded files
# ─────────────────────────────────────────────────────────────────────────────
def load_wfdb_from_bytes(hea_bytes: bytes, dat_bytes: bytes, original_hea_name: str):
    """
    The .hea file's first token is the record name. wfdb looks for a .dat file
    with *that* name, not whatever filename the user uploaded. We rewrite the
    header's first token to match the filename so both files are consistent.
    """
    base  = original_hea_name.rsplit('.', 1)[0]   # e.g. "00010_lr"
    lines = hea_bytes.decode('utf-8', errors='replace').splitlines()

    if lines:
        parts = lines[0].split()
        if parts and parts[0] != base:
            parts[0] = base
            lines[0] = ' '.join(parts)
            hea_bytes = '\n'.join(lines).encode('utf-8')

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, f"{base}.hea"), 'wb') as f:
            f.write(hea_bytes)
        with open(os.path.join(tmp, f"{base}.dat"), 'wb') as f:
            f.write(dat_bytes)

        rec = wfdb.rdrecord(os.path.join(tmp, base))
        return rec.p_signal, rec.fs if rec.fs else 500

# ─────────────────────────────────────────────────────────────────────────────
# ECG PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_ecg_clinical(ecg_data, fs=500, title="12-Lead ECG"):
    n_leads   = min(ecg_data.shape[1], 12)
    n_samples = ecg_data.shape[0]
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(n_leads, 1, figsize=(16, n_leads * 1.25),
                             facecolor='#0a0d14', sharex=True)
    if n_leads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_facecolor('#0a0d14')
        ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
        ax.set_yticks(np.arange(-2, 2.01, 0.1), minor=True)
        ax.grid(True, which='minor', color='#1a2e1e', linewidth=0.4, alpha=0.7)
        ax.set_xticks(np.arange(0, t[-1], 0.2))
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.grid(True, which='major', color='#1e3a22', linewidth=0.7, alpha=0.9)
        ax.plot(t, ecg_data[:, i], color='#00e5a0', linewidth=0.9, antialiased=True)
        lead = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f'L{i+1}'
        ax.text(0.005, 0.85, lead, transform=ax.transAxes, color='#aaaaaa',
                fontsize=8, fontweight='bold', fontfamily='monospace', va='top')
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-1.5, 1.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2d45')
            spine.set_linewidth(0.5)

    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel('Time (s)', color='#64748b', fontsize=8)
    fig.suptitle(title, color='#94a3b8', fontsize=11, fontweight='600',
                 fontfamily='monospace', y=1.002)
    plt.tight_layout(h_pad=0.15)
    return fig


def plot_single_lead(ecg_data, lead_idx=1, fs=500):
    n_samples = ecg_data.shape[0]
    t = np.arange(n_samples) / fs
    fig, ax = plt.subplots(figsize=(14, 2.8), facecolor='#0a0d14')
    ax.set_facecolor('#0a0d14')
    ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
    ax.set_yticks(np.arange(-2, 2.01, 0.1), minor=True)
    ax.grid(True, which='minor', color='#1a2e1e', linewidth=0.4, alpha=0.7)
    ax.set_xticks(np.arange(0, t[-1], 0.2))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, which='major', color='#1e3a22', linewidth=0.7, alpha=0.9)
    ax.plot(t, ecg_data[:, lead_idx], color='#00e5a0', linewidth=1.1, antialiased=True)
    lead = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f'L{lead_idx+1}'
    ax.set_title(f'Lead {lead} — rhythm strip', color='#64748b',
                 fontsize=9, fontfamily='monospace', loc='left', pad=6)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.6, 1.6)
    ax.tick_params(colors='#64748b', labelsize=7)
    ax.set_xlabel('Time (s)', color='#64748b', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2d45')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS HTML
# ─────────────────────────────────────────────────────────────────────────────
def results_html(predictions, threshold):
    rows = ""
    for code, prob in zip(SUPERCLASSES, predictions):
        desc   = CLASS_DESCRIPTIONS[code]
        pct    = prob * 100
        color  = CLASS_COLORS[code]
        is_pos = prob >= threshold
        pill   = ('<span class="status-pill pill-pos">DETECTED</span>'
                  if is_pos else
                  '<span class="status-pill pill-neg">NEGATIVE</span>')
        rows += f"""
        <div class="diag-row {'positive' if is_pos else ''}">
            <span class="diag-code">{code}</span>
            <span class="diag-label">{desc}</span>
            <div class="diag-bar-wrap">
                <div class="diag-bar" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <span class="diag-pct" style="color:{color};">{pct:.1f}%</span>
            {pill}
        </div>"""
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="ecg-header">
        <h1>🫀 ECG Arrhythmia Classifier</h1>
        <p class="subtitle">12-lead ECG deep learning classification · PTB-XL dataset · Model 03</p>
        <div class="badge-row">
            <span class="badge badge-norm">NORM — Normal</span>
            <span class="badge badge-mi">MI — Myocardial Infarction</span>
            <span class="badge badge-sttc">STTC — ST/T Change</span>
            <span class="badge badge-cd">CD — Conduction Disturbance</span>
            <span class="badge badge-hyp">HYP — Hypertrophy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-title">⚙ Model</div>', unsafe_allow_html=True)

        model   = None
        scalers = None

        model_found   = os.path.exists(MODEL_PATH)
        scalers_found = os.path.exists(SCALERS_PATH)

        if model_found:
            model = load_model(MODEL_PATH)
            if model is not None:
                st.markdown('<span class="status-ok">✓ model03.keras loaded</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-err">✗ Model found but failed to load</span>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✗ model03.keras not found</span>',
                        unsafe_allow_html=True)

        if scalers_found:
            scalers = load_scalers(SCALERS_PATH)
            if scalers is not None:
                st.markdown('<span class="status-ok">✓ scalers.pkl loaded</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-err">✗ Scalers found but failed to load</span>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✗ scalers.pkl not found</span>',
                        unsafe_allow_html=True)

        # Auto-expanded when something is missing so the user sees it immediately
        with st.expander("🔍 Path debug", expanded=not (model_found and scalers_found)):
            st.code(
                f"repo root : {_HERE}\n"
                f"cwd       : {os.getcwd()}\n"
                f"__file__  : {os.path.abspath(__file__)}\n"
                f"model     : {'OK' if model_found else 'MISSING'}\n"
                f"           {MODEL_PATH}\n"
                f"scalers   : {'OK' if scalers_found else 'MISSING'}\n"
                f"           {SCALERS_PATH}\n"
                f"sample-dir: {'OK' if os.path.isdir(SAMPLE_DIR) else 'MISSING'}\n"
                f"           {SAMPLE_DIR}",
                language="text",
            )

        st.markdown("---")
        st.markdown('<div class="section-title">🎚 Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider("Positive diagnosis threshold", 0.0, 1.0, 0.5, 0.05,
                              label_visibility="collapsed")
        st.caption(f"Current: **{threshold:.0%}** — probabilities above this are flagged positive")

        st.markdown("---")
        st.markdown('<div class="section-title">📖 How to use</div>', unsafe_allow_html=True)
        st.markdown("""
1. Fill in patient metadata
2. Pick an ECG source
3. Preview the signal
4. Hit **Run Classification**
        """)

    # ── Main columns ─────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<div class="section-title">👤 Patient Metadata</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age    = st.number_input("Age (yrs)", 0, 120, 50, 1)
            height = st.number_input("Height (cm)", 50, 250, 170, 1)
            inf1   = st.selectbox("Infarction Stadium 1", [0,1,2,3,4,5],
                                  format_func=lambda x:{0:"None",1:"I",2:"I-II",3:"II",4:"II-III",5:"III"}[x])
        with c2:
            sex    = st.selectbox("Sex", [0.0, 1.0],
                                  format_func=lambda x:"Female" if x==0.0 else "Male")
            weight = st.number_input("Weight (kg)", 20, 300, 70, 1)
            inf2   = st.selectbox("Infarction Stadium 2", [0,1,2,3],
                                  format_func=lambda x:{0:"None",1:"I",2:"II",3:"III"}[x])

        pacemaker = st.checkbox("Patient has a pacemaker")
        pm_val    = 1.0 if pacemaker else 0.0
        bmi       = weight / (height / 100) ** 2 if height > 0 else 0

        st.markdown(f"""
        <div style="margin-top:1rem;" class="stat-grid">
            <div class="stat-tile"><div class="stat-label">Age</div><div class="stat-value">{age} yrs</div></div>
            <div class="stat-tile"><div class="stat-label">Sex</div><div class="stat-value">{"Male" if sex==1.0 else "Female"}</div></div>
            <div class="stat-tile"><div class="stat-label">BMI</div><div class="stat-value">{bmi:.1f}</div></div>
            <div class="stat-tile"><div class="stat-label">Pacemaker</div><div class="stat-value">{"Yes" if pacemaker else "No"}</div></div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">📡 ECG Signal Input</div>',
                    unsafe_allow_html=True)

        method = st.radio("Source", ["Sample Records", "Upload CSV", "Upload WFDB"],
                          horizontal=True)

        ecg_data     = None
        record_title = "ECG Signal"
        fs           = 500

        if method == "Sample Records":
            sample_files = get_sample_files()
            if not sample_files:
                st.warning(f"No sample files found in `{SAMPLE_DIR}/`.")
            else:
                chosen = st.radio("Pick a record", sample_files, horizontal=True)
                if chosen:
                    try:
                        rec          = wfdb.rdrecord(os.path.join(SAMPLE_DIR, chosen))
                        ecg_data     = rec.p_signal
                        fs           = rec.fs if rec.fs else 500
                        record_title = chosen
                        st.success(f"✓ {ecg_data.shape[0]} samples × {ecg_data.shape[1]} leads @ {fs} Hz")
                    except Exception as e:
                        st.error(f"Could not load `{chosen}`: {e}")

        elif method == "Upload CSV":
            st.caption("No header · rows = time samples · columns = leads (12 expected)")
            up = st.file_uploader("Drop CSV here", type=['csv'])
            if up:
                try:
                    ecg_data     = pd.read_csv(up, header=None).values
                    record_title = up.name
                    st.success(f"✓ {ecg_data.shape[0]} samples × {ecg_data.shape[1]} leads")
                except Exception as e:
                    st.error(f"CSV error: {e}")

        else:
            st.caption("Upload matching .hea and .dat files from one WFDB record")
            ch, cd = st.columns(2)
            with ch:
                hea = st.file_uploader("Header (.hea)", type=['hea'])
            with cd:
                dat = st.file_uploader("Signal  (.dat)", type=['dat'])

            if hea and dat:
                try:
                    # Read bytes before the file objects are consumed
                    hea_bytes    = hea.read()
                    dat_bytes    = dat.read()
                    ecg_data, fs = load_wfdb_from_bytes(hea_bytes, dat_bytes, hea.name)
                    record_title = hea.name.replace('.hea', '')
                    st.success(f"✓ {ecg_data.shape[0]} samples × {ecg_data.shape[1]} leads @ {fs} Hz")
                except Exception as e:
                    st.error(f"WFDB error: {e}")
                    st.code(traceback.format_exc(), language="text")

    # ── ECG Preview ───────────────────────────────────────────────────────────
    if ecg_data is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">🔬 ECG Wave Preview</div>',
                    unsafe_allow_html=True)

        tab_full, tab_rhythm = st.tabs(["12-Lead View", "Rhythm Strip"])

        with tab_full:
            win_sec = st.slider("Visible window (seconds)", 1, 20, 10, 1, key="ecg_win")
            n_disp  = min(win_sec * fs, ecg_data.shape[0])
            fig     = plot_ecg_clinical(ecg_data[:n_disp], fs=fs, title=record_title)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab_rhythm:
            n_avail  = min(ecg_data.shape[1], 12)
            lead_idx = st.selectbox(
                "Lead", list(range(n_avail)),
                format_func=lambda i: LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"Lead {i+1}",
                index=1)
            fig2 = plot_single_lead(ecg_data, lead_idx=lead_idx, fs=fs)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    # ── Run Classification ────────────────────────────────────────────────────
    st.markdown("---")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run = st.button("🚀  Run Classification", use_container_width=True, type="primary")

    if run:
        if model is None:
            st.error("Model not loaded — check the **Path debug** panel in the sidebar.")
        elif scalers is None:
            st.error("Scalers not loaded — check the **Path debug** panel in the sidebar.")
        elif ecg_data is None:
            st.error("Please load an ECG signal first.")
        else:
            with st.spinner("Analysing ECG…"):
                try:
                    meta_in = preprocess_metadata(
                        age, sex, height, weight, inf1, inf2, pm_val,
                        scalers.get('x_scaler'))
                    ecg_in  = preprocess_ecg_signal(
                        ecg_data, scalers.get('y_scaler'), WINDOW_SIZE)

                    preds = model.predict([meta_in, ecg_in], verbose=0)[0]

                    st.markdown('<div class="section-title">📋 Classification Results</div>',
                                unsafe_allow_html=True)

                    rl, rr = st.columns([1.1, 1], gap="large")

                    with rl:
                        st.markdown(results_html(preds, threshold), unsafe_allow_html=True)
                        pos_codes = [SUPERCLASSES[i] for i, p in enumerate(preds)
                                     if p >= threshold]
                        if not pos_codes:
                            st.success("✅ No significant arrhythmia detected above threshold.")
                        else:
                            labels = [f"**{c}** ({CLASS_DESCRIPTIONS[c]})" for c in pos_codes]
                            st.warning("⚠️ Detected: " + " · ".join(labels))

                    with rr:
                        fig_bar, ax = plt.subplots(figsize=(7, 3.5), facecolor='#111827')
                        ax.set_facecolor('#111827')
                        bars = ax.barh(SUPERCLASSES, preds * 100,
                                       color=[CLASS_COLORS[c] for c in SUPERCLASSES],
                                       height=0.55, alpha=0.85)
                        ax.axvline(threshold * 100, color='#f43f5e', linestyle='--',
                                   linewidth=1.5, alpha=0.8, label=f'Threshold {threshold:.0%}')
                        for bar, p in zip(bars, preds):
                            ax.text(bar.get_width() + 1,
                                    bar.get_y() + bar.get_height() / 2,
                                    f"{p*100:.1f}%", va='center', ha='left',
                                    color='#94a3b8', fontsize=9, fontfamily='monospace')
                        ax.set_xlim(0, 115)
                        ax.set_xlabel('Probability (%)', color='#64748b', fontsize=9)
                        ax.tick_params(colors='#94a3b8', labelsize=9)
                        ax.legend(fontsize=8, labelcolor='#94a3b8',
                                  facecolor='#1e293b', edgecolor='#334155')
                        ax.grid(True, axis='x', color='#1e2d45', linewidth=0.7, alpha=0.6)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#1e2d45')
                        plt.tight_layout()
                        st.pyplot(fig_bar, use_container_width=True)
                        plt.close(fig_bar)

                    st.markdown("""
                    <div style="margin-top:1rem;padding:.75rem 1rem;background:#161f2e;
                                border:1px solid #1e2d45;border-left:3px solid #f59e0b;
                                border-radius:8px;font-size:.82rem;color:#94a3b8;">
                        <strong style="color:#f59e0b;">⚕ Medical Disclaimer</strong> —
                        For research and educational purposes only. Not a substitute for
                        professional clinical diagnosis.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.code(traceback.format_exc(), language="text")

    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;color:#334155;
                font-family:'IBM Plex Mono',monospace;font-size:.75rem;">
        ECG Arrhythmia Classifier · PTB-XL Dataset · Streamlit &amp; TensorFlow
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()