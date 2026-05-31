import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BP Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8e4dc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e;
    border-right: 1px solid #2a2d38;
}

/* Headers */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8e4dc !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 4px;
    padding: 1rem;
}

[data-testid="stMetricValue"] {
    color: #c8f0a0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem !important;
}

[data-testid="stMetricLabel"] {
    color: #888 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

[data-testid="stMetricDelta"] svg { display: none; }

/* Buttons */
.stButton > button {
    background: #c8f0a0;
    color: #0d0f14;
    border: none;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.6rem 2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #a8d880;
    transform: translateY(-1px);
}

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #c8f0a0 !important;
}

/* Number inputs */
.stNumberInput input {
    background: #13161e !important;
    border: 1px solid #2a2d38 !important;
    color: #e8e4dc !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
}

/* Select box */
.stSelectbox > div > div {
    background: #13161e !important;
    border: 1px solid #2a2d38 !important;
    color: #e8e4dc !important;
}

/* Info / success boxes */
.stAlert {
    background: #13161e !important;
    border: 1px solid #2a2d38 !important;
    color: #e8e4dc !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13161e;
    border: 1px dashed #2a2d38;
    border-radius: 4px;
}

/* Divider */
hr { border-color: #2a2d38; }

/* Result panel */
.result-box {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-left: 3px solid #c8f0a0;
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.result-box h3 {
    margin-top: 0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #888 !important;
    font-family: 'DM Mono', monospace !important;
}

.big-bp {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #c8f0a0;
    line-height: 1;
}

.bp-unit {
    font-size: 1rem;
    color: #888;
    margin-left: 0.3rem;
}

.change-positive { color: #ff8a80; }
.change-negative { color: #c8f0a0; }
.change-zero     { color: #888; }

.tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 0.5rem;
}
.tag-normal   { background: #1a2e1a; color: #c8f0a0; border: 1px solid #3a5e3a; }
.tag-elevated { background: #2e2a1a; color: #f0d890; border: 1px solid #5e541a; }
.tag-high     { background: #2e1a1a; color: #f09080; border: 1px solid #5e2a1a; }

.model-stat {
    font-size: 0.8rem;
    color: #888;
}
.model-stat span {
    color: #c8f0a0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def classify_bp(sys, dia):
    if sys < 120 and dia < 80:
        return "Normal", "tag-normal"
    elif sys < 130 and dia < 80:
        return "Elevated", "tag-elevated"
    elif sys < 140 or dia < 90:
        return "High — Stage 1", "tag-high"
    else:
        return "High — Stage 2", "tag-high"


def change_class(val):
    if val > 0.5:  return "change-positive"
    if val < -0.5: return "change-negative"
    return "change-zero"


def change_arrow(val):
    if val > 0.5:  return "▲"
    if val < -0.5: return "▼"
    return "—"


@st.cache_data(show_spinner=False)
def train_models(csv_bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))

    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Auto-detect column names (Slovenian or English fallback)
    def find_col(candidates):
        for c in candidates:
            for col in df.columns:
                if c.lower() in col.lower():
                    return col
        return None

    sys_before = find_col(["Sistolični Krvni Tlak(Prej)", "baseline_sys", "sys_before", "systolic_before"])
    dia_before = find_col(["Diastolični Krvni Tlak(prej)", "baseline_dia", "dia_before", "diastolic_before"])
    sys_25     = find_col(["Sistolični Krvni Tlak (2.5mg)", "sys_2.5", "systolic_2.5"])
    dia_25     = find_col(["Diastolični Krvni Tlak(2.5mg)", "dia_2.5", "diastolic_2.5"])
    sys_5      = find_col(["Sistolični Krvni Tlak (5mg)", "sys_5", "systolic_5"])
    dia_5      = find_col(["Diastolični Krvni Tlak(5mg)", "dia_5", "diastolic_5"])
    sys_10     = find_col(["Sistolični Krvni Tlak (10mg)", "sys_10", "systolic_10"])
    dia_10     = find_col(["Diastolični Krvni Tlak(10mg)", "dia_10", "diastolic_10"])

    missing = [n for n, c in [
        ("sys_before", sys_before), ("dia_before", dia_before),
        ("sys_2.5mg", sys_25), ("dia_2.5mg", dia_25),
        ("sys_5mg", sys_5),   ("dia_5mg", dia_5),
        ("sys_10mg", sys_10), ("dia_10mg", dia_10),
    ] if c is None]
    if missing:
        return None, None, None, None, f"Could not find columns: {', '.join(missing)}"

    rows = []
    for _, row in df.iterrows():
        b_sys = row[sys_before]
        b_dia = row[dia_before]
        for dose, s_col, d_col in [(2.5, sys_25, dia_25), (5, sys_5, dia_5), (10, sys_10, dia_10)]:
            s, d = row[s_col], row[d_col]
            if pd.notna(s) and pd.notna(d):
                rows.append([b_sys, b_dia, dose, s - b_sys, d - b_dia])

    data = pd.DataFrame(rows, columns=["baseline_sys", "baseline_dia", "dose", "sys_change", "dia_change"])

    X      = data[["baseline_sys", "baseline_dia", "dose"]]
    y_sys  = data["sys_change"]
    y_dia  = data["dia_change"]

    X_train, X_test, ys_train, ys_test = train_test_split(X, y_sys, test_size=0.25, random_state=42)
    _,       _,      yd_train, yd_test = train_test_split(X, y_dia, test_size=0.25, random_state=42)

    sys_model = LinearRegression().fit(X_train, ys_train)
    dia_model = LinearRegression().fit(X_train, yd_train)

    stats = {
        "sys_mae": mean_absolute_error(ys_test, sys_model.predict(X_test)),
        "sys_r2":  r2_score(ys_test, sys_model.predict(X_test)),
        "dia_mae": mean_absolute_error(yd_test, dia_model.predict(X_test)),
        "dia_r2":  r2_score(yd_test, dia_model.predict(X_test)),
        "n_samples": len(data),
    }
    return sys_model, dia_model, stats, data, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 BP Predictor")
    st.markdown("<p style='color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em'>Data & Model</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])

    st.markdown("---")
    st.markdown("<p style='color:#888;font-size:0.7rem'>Expected columns (Slovenian or English column names accepted):<br><br>• Sistolični/Diastolični Krvni Tlak (Prej)<br>• …(2.5mg), …(5mg), …(10mg)</p>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# Blood Pressure Response Predictor")
st.markdown("<p style='color:#888;margin-top:-0.5rem'>Linear regression · Amlodipine dose response</p>", unsafe_allow_html=True)
st.markdown("---")

if uploaded is None:
    st.info("Upload a CSV file in the sidebar to train the model, then enter patient values below.")
    st.stop()

# Train
with st.spinner("Training models…"):
    sys_model, dia_model, stats, data_ml, err = train_models(uploaded.read())

if err:
    st.error(f"**Column detection failed:** {err}")
    st.stop()

# Model stats row
st.markdown("#### Model performance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Systolic MAE",  f"{stats['sys_mae']:.2f} mmHg")
c2.metric("Systolic R²",   f"{stats['sys_r2']:.3f}")
c3.metric("Diastolic MAE", f"{stats['dia_mae']:.2f} mmHg")
c4.metric("Diastolic R²",  f"{stats['dia_r2']:.3f}")
c5.metric("Training rows", stats['n_samples'])

st.markdown("---")

# ── Prediction form ───────────────────────────────────────────────────────────
st.markdown("#### Patient input")

col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    baseline_sys = st.number_input(
        "Baseline Systolic BP (mmHg)",
        min_value=80, max_value=220, value=140, step=1,
        help="Systolic blood pressure before medication"
    )

with col_b:
    baseline_dia = st.number_input(
        "Baseline Diastolic BP (mmHg)",
        min_value=40, max_value=140, value=90, step=1,
        help="Diastolic blood pressure before medication"
    )

with col_c:
    dose = st.selectbox(
        "Amlodipine Dose",
        options=[2.5, 5.0, 10.0],
        format_func=lambda x: f"{x} mg",
        index=1,
    )

st.markdown("")
predict_btn = st.button("Predict BP Response", use_container_width=False)

# ── Prediction output ─────────────────────────────────────────────────────────
if predict_btn:
    user_X = pd.DataFrame([{
        "baseline_sys": float(baseline_sys),
        "baseline_dia": float(baseline_dia),
        "dose": float(dose),
    }])

    sys_change = sys_model.predict(user_X)[0]
    dia_change = dia_model.predict(user_X)[0]
    final_sys  = baseline_sys + sys_change
    final_dia  = baseline_dia + dia_change

    label, tag_cls = classify_bp(final_sys, final_dia)

    st.markdown("---")
    st.markdown("#### Prediction result")

    left, right = st.columns([3, 2])

    with left:
        st.markdown(f"""
        <div class="result-box">
            <h3>Predicted BP after {dose} mg amlodipine</h3>
            <div class="big-bp">{final_sys:.0f}<span class="bp-unit">/</span>{final_dia:.0f}<span class="bp-unit"> mmHg</span></div>
            <div class="tag {tag_cls}">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        sc = change_class(sys_change)
        dc = change_class(dia_change)
        sa = change_arrow(sys_change)
        da = change_arrow(dia_change)

        st.markdown(f"""
        <div class="result-box">
            <h3>Changes from baseline</h3>
            <p style="margin:0.3rem 0">
                <span style="color:#888">Systolic &nbsp;</span>
                <span class="{sc}">{sa} {abs(sys_change):.1f} mmHg</span>
            </p>
            <p style="margin:0.3rem 0">
                <span style="color:#888">Diastolic </span>
                <span class="{dc}">{da} {abs(dia_change):.1f} mmHg</span>
            </p>
            <p style="margin:0.8rem 0 0; font-size:0.72rem; color:#555">
                Baseline: {baseline_sys}/{baseline_dia} mmHg
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Dose comparison table
    st.markdown("#### Dose comparison")
    rows_cmp = []
    for d in [2.5, 5.0, 10.0]:
        xp = pd.DataFrame([{"baseline_sys": float(baseline_sys), "baseline_dia": float(baseline_dia), "dose": d}])
        sc_ = sys_model.predict(xp)[0]
        dc_ = dia_model.predict(xp)[0]
        rows_cmp.append({
            "Dose": f"{d} mg",
            "Predicted Systolic": f"{baseline_sys + sc_:.0f}",
            "Systolic Δ": f"{sc_:+.1f}",
            "Predicted Diastolic": f"{baseline_dia + dc_:.0f}",
            "Diastolic Δ": f"{dc_:+.1f}",
            "Classification": classify_bp(baseline_sys + sc_, baseline_dia + dc_)[0],
        })

    cmp_df = pd.DataFrame(rows_cmp)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ── Data explorer (optional) ──────────────────────────────────────────────────
with st.expander("Explore training data"):
    st.dataframe(data_ml.describe().round(2), use_container_width=True)
    st.markdown(f"<p class='model-stat'>Total samples: <span>{len(data_ml)}</span></p>", unsafe_allow_html=True)
