"""
ECG Arrhythmia Classifier — Streamlit App
Designed to be embedded inside an iframe within an external HTML website.

Author: Mohamad AlJasem · https://aljasem.eu.org
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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# NUCLEAR HIDE — kills header, toolbar, sidebar and its toggle arrow
# Targets every known selector across Streamlit versions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Kill ALL chrome ── */
#MainMenu                                { display: none !important; }
header[data-testid="stHeader"]           { display: none !important; }
footer                                   { display: none !important; }
[data-testid="stToolbar"]                { display: none !important; }
[data-testid="stDecoration"]             { display: none !important; }
[data-testid="stStatusWidget"]           { display: none !important; }

/* Sidebar panel */
[data-testid="stSidebar"]               { display: none !important; }

/* Sidebar collapse/expand toggle arrow button */
[data-testid="collapsedControl"]         { display: none !important; }
button[kind="header"]                    { display: none !important; }
.st-emotion-cache-1dp5vir               { display: none !important; }

/* Catch-all for any remaining top bar */
.stAppDeployButton                       { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }

/* ── Design tokens ── */
:root {
    --surface:   rgba(255,255,255,0.04);
    --surface2:  rgba(255,255,255,0.07);
    --border:    rgba(255,255,255,0.10);
    --accent:    #00d4aa;
    --danger:    #f43f5e;
    --warn:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    9px;
    --font-body: 'IBM Plex Sans', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background: transparent !important;
    color: var(--text) !important;
}
.main  { background: transparent !important; }
.block-container {
    padding: 1rem 1.25rem 2rem !important;
    max-width: 100% !important;
}

/* ── Top status bar ── */
.top-bar {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .5rem .85rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: .85rem;
    flex-wrap: wrap;
}
.model-pill {
    font-family: var(--font-mono);
    font-size: .7rem;
    padding: .17rem .58rem;
    border-radius: 999px;
    border: 1px solid;
    white-space: nowrap;
}
.pill-ok  { color:#00d4aa; border-color:rgba(0,212,170,.35); background:rgba(0,212,170,.08); }
.pill-err { color:#f43f5e; border-color:rgba(244,63,94,.35);  background:rgba(244,63,94,.08); }

/* ── Section labels ── */
.sec-label {
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 .6rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}
.sec-label::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Stat chips ── */
.stat-row  { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.6rem; }
.stat-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: .28rem .65rem;
    font-family: var(--font-mono);
    font-size: .78rem;
    color: var(--text);
    white-space: nowrap;
}
.stat-chip span { color:var(--muted); font-size:.66rem; margin-right:.28rem; }

/* ── Diagnosis result rows ── */
.diag-row {
    display: flex;
    align-items: center;
    gap: .55rem;
    padding: .44rem .65rem;
    border-radius: 6px;
    margin-bottom: .28rem;
    border: 1px solid var(--border);
    background: var(--surface);
}
.diag-row.positive { border-color:rgba(244,63,94,.4); background:rgba(244,63,94,.05); }
.diag-row.norm-pos { border-color:rgba(0,212,170,.4); background:rgba(0,212,170,.05); }
.diag-code  { font-family:var(--font-mono); font-size:.7rem; color:var(--muted); min-width:34px; }
.diag-label { flex:1; font-size:.82rem; }
.diag-bar-wrap { flex:1.1; height:4px; background:rgba(255,255,255,.08); border-radius:2px; overflow:hidden; }
.diag-bar   { height:100%; border-radius:2px; }
.diag-pct   { font-family:var(--font-mono); font-size:.82rem; font-weight:600; min-width:42px; text-align:right; }
.status-pill {
    font-size:.62rem; font-family:var(--font-mono);
    padding:.1rem .4rem; border-radius:999px; white-space:nowrap;
}
.s-pos { background:rgba(244,63,94,.18); color:#f43f5e; border:1px solid rgba(244,63,94,.35); }
.s-neg { background:rgba(0,212,170,.10); color:#00d4aa; border:1px solid rgba(0,212,170,.25); }

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg,#00d4aa,#0ea5e9) !important;
    color: #050d18 !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 7px !important;
    padding: .45rem 1.4rem !important;
    font-family: var(--font-body) !important;
    font-size: .88rem !important;
}
.stButton > button:hover { opacity:.85 !important; }

div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 7px !important;
}
.stRadio > div  { gap:.35rem !important; }
.stRadio label span { font-size:.82rem !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    border-radius: 5px 5px 0 0 !important;
    color: var(--muted) !important;
    font-size: .8rem !important;
    padding: .25rem .8rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    background: var(--surface2) !important;
}
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: 7px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def _find_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd        = os.getcwd()
    candidates = [script_dir, cwd]
    for base in (script_dir, cwd):
        for parent in pathlib.Path(base).parents:
            candidates.append(str(parent))
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    for root in unique:
        if os.path.isfile(os.path.join(root, "models", "model03.keras")):
            return root
    return cwd

_HERE        = _find_repo_root()
MODEL_PATH   = os.path.join(_HERE, "models", "model03.keras")
SCALERS_PATH = os.path.join(_HERE, "models", "scalers.pkl")
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
LEAD_NAMES  = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
WINDOW_SIZE = 800

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    try:
        return keras.models.load_model(path, compile=False)
    except Exception:
        return None

@st.cache_resource
def load_scalers(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

model   = load_model(MODEL_PATH)
scalers = load_scalers(SCALERS_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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
    return x_scaler.transform(meta.values) if x_scaler is not None else meta.values


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


def load_wfdb_from_bytes(hea_bytes: bytes, dat_bytes: bytes, original_hea_name: str):
    base  = original_hea_name.rsplit('.', 1)[0]
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
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
_BG  = '#0a0d14'
_GM  = '#1a2e1e'
_GMJ = '#1e3a22'
_SIG = '#00e5a0'
_SP  = '#1e2d45'

def plot_ecg_clinical(ecg_data, fs=500, title=""):
    n = min(ecg_data.shape[1], 12)
    t = np.arange(ecg_data.shape[0]) / fs
    fig, axes = plt.subplots(n, 1, figsize=(13, n * 1.05), facecolor=_BG, sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.set_facecolor(_BG)
        ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
        ax.set_yticks(np.arange(-2, 2.01, 0.1), minor=True)
        ax.grid(True, which='minor', color=_GM,  linewidth=0.35, alpha=0.7)
        ax.set_xticks(np.arange(0, t[-1], 0.2))
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.grid(True, which='major', color=_GMJ, linewidth=0.6, alpha=0.9)
        ax.plot(t, ecg_data[:, i], color=_SIG, linewidth=0.8, antialiased=True)
        lead = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f'L{i+1}'
        ax.text(0.006, 0.82, lead, transform=ax.transAxes, color='#9ca3af',
                fontsize=7, fontweight='bold', fontfamily='monospace', va='top')
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-1.5, 1.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SP); sp.set_linewidth(0.4)
    axes[-1].tick_params(labelbottom=True, colors='#64748b', labelsize=6.5)
    axes[-1].set_xlabel('Time (s)', color='#64748b', fontsize=7)
    if title:
        fig.suptitle(title, color='#94a3b8', fontsize=8.5, fontfamily='monospace', y=1.002)
    plt.tight_layout(h_pad=0.08)
    return fig


def plot_single_lead(ecg_data, lead_idx=1, fs=500):
    t = np.arange(ecg_data.shape[0]) / fs
    fig, ax = plt.subplots(figsize=(13, 2.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
    ax.set_yticks(np.arange(-2, 2.01, 0.1), minor=True)
    ax.grid(True, which='minor', color=_GM,  linewidth=0.35, alpha=0.7)
    ax.set_xticks(np.arange(0, t[-1], 0.2))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, which='major', color=_GMJ, linewidth=0.6, alpha=0.9)
    ax.plot(t, ecg_data[:, lead_idx], color=_SIG, linewidth=0.95, antialiased=True)
    lead = LEAD_NAMES[lead_idx] if lead_idx < len(LEAD_NAMES) else f'L{lead_idx+1}'
    ax.set_title(f'Lead {lead} — rhythm strip', color='#64748b',
                 fontsize=7.5, fontfamily='monospace', loc='left', pad=4)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.6, 1.6)
    ax.tick_params(colors='#64748b', labelsize=6.5)
    ax.set_xlabel('Time (s)', color='#64748b', fontsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SP)
    plt.tight_layout()
    return fig


def plot_results_bar(preds, threshold):
    fig, ax = plt.subplots(figsize=(5.5, 2.8), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    bars = ax.barh(SUPERCLASSES, preds * 100,
                   color=[CLASS_COLORS[c] for c in SUPERCLASSES],
                   height=0.48, alpha=0.88)
    ax.axvline(threshold * 100, color='#f43f5e', linestyle='--',
               linewidth=1.2, alpha=0.8, label=f'Threshold {threshold:.0%}')
    for bar, p in zip(bars, preds):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va='center', ha='left',
                color='#94a3b8', fontsize=7.5, fontfamily='monospace')
    ax.set_xlim(0, 118)
    ax.set_xlabel('Probability (%)', color='#64748b', fontsize=7.5)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.legend(fontsize=7, labelcolor='#94a3b8', facecolor='#161f2e', edgecolor='#1e2d45')
    ax.grid(True, axis='x', color='#1e2d45', linewidth=0.5, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e2d45')
    plt.tight_layout()
    return fig


def results_html(predictions, threshold):
    rows = ""
    for code, prob in zip(SUPERCLASSES, predictions):
        pct    = prob * 100
        color  = CLASS_COLORS[code]
        is_pos = prob >= threshold
        row_cls = ("norm-pos" if code == 'NORM' else "positive") if is_pos else ""
        pill    = (f'<span class="status-pill s-pos">DETECTED</span>'
                   if is_pos else
                   f'<span class="status-pill s-neg">NEGATIVE</span>')
        rows += f"""
        <div class="diag-row {row_cls}">
            <span class="diag-code">{code}</span>
            <span class="diag-label">{CLASS_DESCRIPTIONS[code]}</span>
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
    model_ok   = model is not None
    scalers_ok = scalers is not None

    # ── Top bar: status + threshold ──────────────────────────────────────────
    bar_l, bar_r = st.columns([2.5, 1], gap="medium")
    with bar_l:
        ok  = lambda lbl: f'<span class="model-pill pill-ok">✓ {lbl}</span>'
        err = lambda lbl: f'<span class="model-pill pill-err">✗ {lbl}</span>'
        st.markdown(
            '<div class="top-bar">'
            f'  {ok("model") if model_ok else err("model missing")}'
            f'  {ok("scalers") if scalers_ok else err("scalers missing")}'
            '</div>',
            unsafe_allow_html=True,
        )
    with bar_r:
        threshold = st.slider("thr", 0.0, 1.0, 0.5, 0.05,
                              label_visibility="collapsed",
                              help="Probability threshold for positive detection")
        st.caption(f"Threshold: **{threshold:.0%}**")

    # ── Three-column layout ──────────────────────────────────────────────────
    col_meta, col_ecg, col_res = st.columns([0.9, 1.5, 1.1], gap="medium")

    # ── PATIENT ──────────────────────────────────────────────────────────────
    with col_meta:
        st.markdown('<div class="sec-label">👤 Patient</div>', unsafe_allow_html=True)
        ra, rb = st.columns(2)
        with ra:
            age    = st.number_input("Age", 0, 120, 50, 1)
            height = st.number_input("Height cm", 50, 250, 170, 1)
        with rb:
            sex    = st.selectbox("Sex", [0.0, 1.0],
                                  format_func=lambda x: "F" if x == 0.0 else "M")
            weight = st.number_input("Weight kg", 20, 300, 70, 1)
        inf1 = st.selectbox("Inf. Stadium 1", [0,1,2,3,4,5],
                            format_func=lambda x:{0:"—",1:"I",2:"I-II",3:"II",4:"II-III",5:"III"}[x])
        inf2 = st.selectbox("Inf. Stadium 2", [0,1,2,3],
                            format_func=lambda x:{0:"—",1:"I",2:"II",3:"III"}[x])
        pacemaker = st.checkbox("Pacemaker")
        pm_val    = 1.0 if pacemaker else 0.0
        bmi       = weight / (height / 100) ** 2 if height > 0 else 0
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-chip"><span>BMI</span>{bmi:.1f}</div>'
            f'<div class="stat-chip"><span>PM</span>{"Yes" if pacemaker else "No"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── ECG INPUT + PREVIEW ───────────────────────────────────────────────────
    with col_ecg:
        st.markdown('<div class="sec-label">📡 ECG Signal</div>', unsafe_allow_html=True)

        method = st.radio("src", ["Sample Records", "Upload CSV", "Upload WFDB"],
                          horizontal=True, label_visibility="collapsed")

        ecg_data     = None
        record_title = ""
        fs           = 500

        if method == "Sample Records":
            sample_files = get_sample_files()
            if not sample_files:
                st.caption("No samples found in `sample-data/`")
            else:
                chosen = st.radio("rec", sample_files, horizontal=True,
                                  label_visibility="collapsed")
                if chosen:
                    try:
                        rec          = wfdb.rdrecord(os.path.join(SAMPLE_DIR, chosen))
                        ecg_data     = rec.p_signal
                        fs           = rec.fs if rec.fs else 500
                        record_title = chosen
                        st.caption(f"✓ {ecg_data.shape[0]} × {ecg_data.shape[1]} leads @ {fs} Hz")
                    except Exception as e:
                        st.error(f"Load error: {e}")

        elif method == "Upload CSV":
            st.caption("No header · rows = samples · cols = leads")
            up = st.file_uploader("csv", type=['csv'], label_visibility="collapsed")
            if up:
                try:
                    ecg_data     = pd.read_csv(up, header=None).values
                    record_title = up.name
                    st.caption(f"✓ {ecg_data.shape[0]} × {ecg_data.shape[1]} leads")
                except Exception as e:
                    st.error(f"CSV error: {e}")

        else:
            st.caption("Upload matching .hea + .dat pair")
            uh, ud = st.columns(2)
            with uh:
                hea = st.file_uploader(".hea", type=['hea'], label_visibility="collapsed")
            with ud:
                dat = st.file_uploader(".dat", type=['dat'], label_visibility="collapsed")
            if hea and dat:
                try:
                    ecg_data, fs = load_wfdb_from_bytes(hea.read(), dat.read(), hea.name)
                    record_title = hea.name.replace('.hea', '')
                    st.caption(f"✓ {ecg_data.shape[0]} × {ecg_data.shape[1]} leads @ {fs} Hz")
                except Exception as e:
                    st.error(f"WFDB error: {e}")
                    st.code(traceback.format_exc(), language="text")

        if ecg_data is not None:
            st.markdown('<div class="sec-label" style="margin-top:.6rem;">🔬 Preview</div>',
                        unsafe_allow_html=True)
            t12, trhy = st.tabs(["12-Lead", "Rhythm Strip"])
            with t12:
                win = st.slider("win", 1, 20, 8, 1, key="win12",
                                label_visibility="collapsed")
                nd  = min(win * fs, ecg_data.shape[0])
                fig = plot_ecg_clinical(ecg_data[:nd], fs=fs, title=record_title)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with trhy:
                n_av = min(ecg_data.shape[1], 12)
                lid  = st.selectbox("lead", list(range(n_av)),
                                    format_func=lambda i: LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i+1}",
                                    index=1, label_visibility="collapsed")
                fig2 = plot_single_lead(ecg_data, lead_idx=lid, fs=fs)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

    # ── RUN + RESULTS ─────────────────────────────────────────────────────────
    with col_res:
        st.markdown('<div class="sec-label">🚀 Run</div>', unsafe_allow_html=True)
        run = st.button("Run Classification", use_container_width=True, type="primary")

        if run:
            if not model_ok:
                st.error("Model not loaded.")
            elif not scalers_ok:
                st.error("Scalers not loaded.")
            elif ecg_data is None:
                st.error("Load an ECG signal first.")
            else:
                with st.spinner("Analysing…"):
                    try:
                        meta_in = preprocess_metadata(
                            age, sex, height, weight, inf1, inf2, pm_val,
                            scalers.get('x_scaler'))
                        ecg_in  = preprocess_ecg_signal(
                            ecg_data, scalers.get('y_scaler'), WINDOW_SIZE)
                        preds   = model.predict([meta_in, ecg_in], verbose=0)[0]
                        st.session_state['preds']     = preds
                        st.session_state['threshold'] = threshold
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        st.code(traceback.format_exc(), language="text")

        if 'preds' in st.session_state:
            preds    = st.session_state['preds']
            st.markdown(results_html(preds, threshold), unsafe_allow_html=True)

            pos_codes = [SUPERCLASSES[i] for i, p in enumerate(preds) if p >= threshold]
            non_norm  = [c for c in pos_codes if c != 'NORM']
            if not pos_codes:
                st.success("No findings above threshold")
            elif not non_norm:
                st.success("Normal ECG")
            else:
                st.warning("Flagged: " + " · ".join(non_norm))

            fig_b = plot_results_bar(preds, threshold)
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)

            st.markdown(
                '<div style="margin-top:.5rem;padding:.5rem .65rem;'
                'background:rgba(245,158,11,.07);'
                'border:1px solid rgba(245,158,11,.2);'
                'border-left:3px solid #f59e0b;border-radius:6px;'
                'font-size:.7rem;color:#94a3b8;line-height:1.4;">'
                '<strong style="color:#f59e0b;">⚕ Disclaimer</strong> — '
                'Research &amp; education only. Not for clinical use.'
                '</div>',
                unsafe_allow_html=True,
            )


if __name__ == '__main__':
    main()