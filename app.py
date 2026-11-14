"""
🚕 NYC Taxi Fare Predictor — Dark-Themed Streamlit Dashboard
Deep Feedforward Neural Network (MLP 128→64→32→1) trained on ~1.9M NYC taxi trips (2009–2015)
"""

import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd
import streamlit as st

import features
from predictor import FarePredictor

# ──────────────────────────── paths & constants ────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Geo-fence matching the training data
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -74.3, -73.7, 40.5, 41.0

# Popular locations for quick selection
PLACES = {
    "📍 Times Square":                  (40.7580, -73.9855),
    "🚂 Grand Central Terminal":        (40.7527, -73.9772),
    "🚉 Penn Station":                  (40.7506, -73.9935),
    "🌳 Central Park":                  (40.7745, -73.9712),
    "🏛️ Columbia University":           (40.8075, -73.9626),
    "💰 Wall Street":                   (40.7060, -74.0088),
    "🌉 Brooklyn Bridge":              (40.7061, -73.9969),
    "✈️ JFK Airport":                   (40.6413, -73.7781),
    "✈️ LaGuardia Airport":             (40.7769, -73.8740),
    "✈️ Newark Airport (EWR)":          (40.6895, -74.1745),
    "🗽 Statue of Liberty Ferry":       (40.6892, -74.0445),
    "🏟️ Yankee Stadium":               (40.8296, -73.9262),
    "🏥 NYU Langone Medical Center":    (40.7421, -73.9739),
}

# ──────────────────────────── custom dark CSS ──────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root dark palette ── */
:root {
    --bg-primary:    #0E1117;
    --bg-secondary:  #161B22;
    --bg-card:       #1C2333;
    --bg-card-hover: #222D3F;
    --accent:        #FBBF24;
    --accent-dim:    #D97706;
    --accent-glow:   rgba(251,191,36,0.15);
    --text-primary:  #E6EDF3;
    --text-secondary:#8B949E;
    --text-muted:    #6E7681;
    --border:        #30363D;
    --success:       #3FB950;
    --info:          #58A6FF;
    --warning:       #D29922;
    --danger:        #F85149;
    --gradient-gold: linear-gradient(135deg, #FBBF24 0%, #F59E0B 50%, #D97706 100%);
    --gradient-dark: linear-gradient(135deg, #1C2333 0%, #161B22 100%);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Remove default Streamlit padding for wide feel */
.block-container { padding-top: 1rem !important; max-width: 1200px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -30%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(251,191,36,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--gradient-gold);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 750px;
}
.hero-badge {
    display: inline-block;
    background: rgba(251,191,36,0.12);
    color: var(--accent);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--border);
}
.section-header h2 {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.section-icon {
    font-size: 1.5rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    min-width: 200px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.3rem;
    transition: all 0.25s ease;
}
.metric-card:hover {
    border-color: var(--accent-dim);
    box-shadow: 0 4px 20px var(--accent-glow);
    transform: translateY(-2px);
}
.metric-card.highlight {
    border-color: var(--accent);
    background: linear-gradient(135deg, rgba(251,191,36,0.06) 0%, var(--bg-card) 100%);
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1.1;
}
.metric-value.green  { color: var(--success); }
.metric-value.blue   { color: var(--info); }
.metric-value.white  { color: var(--text-primary); }
.metric-sub {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* ── Info cards ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}
.info-card.gold-border { border-left: 4px solid var(--accent); }
.info-card.blue-border { border-left: 4px solid var(--info); }

/* ── Architecture diagram ── */
.arch-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}
.arch-block {
    text-align: center;
    padding: 14px 18px;
    border-radius: 10px;
    min-width: 90px;
}
.arch-block.input {
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    color: var(--info);
}
.arch-block.hidden {
    background: rgba(251,191,36,0.10);
    border: 1px solid rgba(251,191,36,0.25);
    color: var(--accent);
}
.arch-block.output {
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.3);
    color: var(--success);
}
.arch-label { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
.arch-value { font-size: 1.3rem; font-weight: 800; margin-top: 2px; }
.arch-arrow { font-size: 1.4rem; color: var(--text-muted); padding: 0 4px; }

/* ── Comparison table ── */
.comp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    margin: 1rem 0;
}
.comp-table th {
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 12px 14px;
    text-align: left;
    border-bottom: 2px solid var(--border);
}
.comp-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
    color: var(--text-primary);
}
.comp-table tr:last-child td { border-bottom: none; }
.comp-table tr:hover td { background: var(--bg-card-hover); }
.comp-table .mlp-row { background: rgba(251,191,36,0.06); }
.comp-table .mlp-row td { font-weight: 600; }
.rank-badge {
    display: inline-block;
    width: 22px; height: 22px;
    border-radius: 50%;
    text-align: center;
    line-height: 22px;
    font-size: 0.7rem;
    font-weight: 700;
}
.rank-1 { background: var(--accent); color: #000; }
.rank-2 { background: var(--text-secondary); color: #000; }
.rank-3 { background: #8B6914; color: #fff; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}

/* ── Streamlit overrides ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
.stSelectbox label, .stNumberInput label, .stDateInput label, .stTimeInput label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── load model ───────────────────────────────────
@st.cache_resource
def load_model():
    return FarePredictor(HERE)

@st.cache_data
def load_config():
    with open(os.path.join(HERE, "feature_config.json")) as fh:
        return json.load(fh)

model = load_model()
cfg   = load_config()
tm    = cfg.get("test_metrics", {})

# ──────────────────────────── sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size:3rem;">🚕</div>
        <div style="font-size:1.1rem; font-weight:700; color:#FBBF24; margin-top:4px;">NYC Taxi Fare</div>
        <div style="font-size:0.78rem; color:#8B949E;">Deep Learning Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📍 Pickup Location")
    pu_names = list(PLACES) + ["📝 Custom Coordinates"]
    pu_choice = st.selectbox("Select pickup", pu_names, index=0, label_visibility="collapsed")
    if pu_choice == "📝 Custom Coordinates":
        c1, c2 = st.columns(2)
        pu_lat = c1.number_input("Pickup Lat", value=40.7580, format="%.5f", step=0.001)
        pu_lon = c2.number_input("Pickup Lon", value=-73.9855, format="%.5f", step=0.001)
    else:
        pu_lat, pu_lon = PLACES[pu_choice]

    st.markdown("##### 📍 Drop-off Location")
    do_names = list(PLACES) + ["📝 Custom Coordinates"]
    do_choice = st.selectbox("Select drop-off", do_names, index=7, label_visibility="collapsed")  # JFK
    if do_choice == "📝 Custom Coordinates":
        c1, c2 = st.columns(2)
        do_lat = c1.number_input("Dropoff Lat", value=40.6413, format="%.5f", step=0.001)
        do_lon = c2.number_input("Dropoff Lon", value=-73.7781, format="%.5f", step=0.001)
    else:
        do_lat, do_lon = PLACES[do_choice]

    st.markdown("---")
    st.markdown("##### 🕐 Trip Details")
    date = st.date_input("Pickup Date", value=dt.date(2014, 6, 15),
                         min_value=dt.date(2009, 1, 1), max_value=dt.date(2015, 6, 30))
    time = st.time_input("Pickup Time", value=dt.time(18, 0))
    pax  = st.number_input("Passengers", min_value=1, max_value=6, value=1, step=1)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0; color:#6E7681; font-size:0.72rem;">
        Model trained on Kaggle NYC Taxi<br>Fare Prediction dataset<br>
        <span style="color:#FBBF24;">~1.9M trips · 2009–2015</span>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────── validation ───────────────────────────────────
inside = (all(LON_MIN <= x <= LON_MAX for x in (pu_lon, do_lon))
          and all(LAT_MIN <= y <= LAT_MAX for y in (pu_lat, do_lat)))

# ──────────────────────────── hero banner ──────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Deep Learning Project</div>
    <div class="hero-title">NYC Taxi Fare Predictor</div>
    <div class="hero-subtitle">
        A deep feedforward neural network (MLP) that predicts New York City taxi fares
        using pickup/drop-off coordinates, trip timing, and passenger count.
        Trained on <strong>~1.9 million</strong> historical trips from the Kaggle NYC Taxi Fare dataset.
    </div>
</div>
""", unsafe_allow_html=True)

if not inside:
    st.error(
        f"⚠️ Both pickup and drop-off must lie inside the training area: "
        f"latitude {LAT_MIN}–{LAT_MAX}, longitude {LON_MIN}–{LON_MAX}. "
        f"Please adjust your coordinates in the sidebar."
    )
    st.stop()

# ──────────────────────────── prediction ───────────────────────────────────
raw = pd.DataFrame({
    "pickup_datetime":  [pd.Timestamp(dt.datetime.combine(date, time))],
    "pickup_longitude": [pu_lon],  "pickup_latitude":  [pu_lat],
    "dropoff_longitude": [do_lon], "dropoff_latitude":  [do_lat],
    "passenger_count":  [int(pax)],
})

fare      = max(float(model.predict(raw)[0]), 2.5)
dist_km   = float(features.haversine_km(pu_lat, pu_lon, do_lat, do_lon))
dist_mi   = dist_km * 0.621371
feats_row = features.make_features(raw)
airport   = bool(feats_row["airport_trip"].iloc[0])
rush_hour = bool(feats_row["is_rush_hour"].iloc[0])
is_night  = bool(feats_row["is_night"].iloc[0])

# ──────────────────────────── prediction results ───────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-icon">💰</span>
    <h2>Fare Prediction</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card highlight">
        <div class="metric-label">Estimated Fare</div>
        <div class="metric-value">${fare:,.2f}</div>
        <div class="metric-sub">USD (2009–2015 pricing)</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Straight-Line Distance</div>
        <div class="metric-value blue">{dist_km:.2f} km</div>
        <div class="metric-sub">{dist_mi:.2f} miles</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Airport Trip</div>
        <div class="metric-value {'green' if airport else 'white'}">{'✅ Yes' if airport else '❌ No'}</div>
        <div class="metric-sub">{'Within airport zone radius' if airport else 'Standard city trip'}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Time Context</div>
        <div class="metric-value white">{'🌙 Night' if is_night else ('🚦 Rush Hour' if rush_hour else '☀️ Regular')}</div>
        <div class="metric-sub">{date.strftime('%b %d, %Y')} at {time.strftime('%I:%M %p')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if dist_km < 0.1:
    st.warning("⚠️ Pickup and drop-off are practically the same location — the estimate may be unreliable.")

# ──────────────────────────── map ──────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-icon">🗺️</span>
    <h2>Trip Route</h2>
</div>
""", unsafe_allow_html=True)

map_data = pd.DataFrame({
    "lat": [pu_lat, do_lat],
    "lon": [pu_lon, do_lon],
})

col_map, col_details = st.columns([3, 1])
with col_map:
    st.map(map_data, zoom=10, use_container_width=True)
with col_details:
    st.markdown(f"""
    <div class="info-card gold-border">
        <div style="font-weight:700; color:#FBBF24; margin-bottom:8px;">📍 Pickup</div>
        <div style="font-size:0.85rem; color:#E6EDF3;">{pu_choice.split(' ', 1)[-1] if '📝' not in pu_choice else 'Custom'}</div>
        <div style="font-size:0.75rem; color:#8B949E; margin-top:2px;">{pu_lat:.4f}°N, {abs(pu_lon):.4f}°W</div>
    </div>
    <div class="info-card blue-border">
        <div style="font-weight:700; color:#58A6FF; margin-bottom:8px;">📍 Drop-off</div>
        <div style="font-size:0.85rem; color:#E6EDF3;">{do_choice.split(' ', 1)[-1] if '📝' not in do_choice else 'Custom'}</div>
        <div style="font-size:0.75rem; color:#8B949E; margin-top:2px;">{do_lat:.4f}°N, {abs(do_lon):.4f}°W</div>
    </div>
    <div class="info-card" style="border-left:4px solid var(--success);">
        <div style="font-weight:700; color:#3FB950; margin-bottom:8px;">🚖 Passengers</div>
        <div style="font-size:1.3rem; font-weight:800; color:#E6EDF3;">{int(pax)}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────── model performance ────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-icon">📊</span>
    <h2>Model Performance</h2>
</div>
""", unsafe_allow_html=True)

if tm:
    st.markdown(f"""
    <div class="info-card gold-border" style="margin-bottom:1rem;">
        <div style="font-size:0.88rem; color:#E6EDF3; line-height:1.6;">
            Evaluated on <strong style="color:#FBBF24;">{int(tm.get('Rows', 0)):,}</strong> unseen taxi trips.
            The model achieves an average error of
            <strong style="color:#3FB950;">${tm.get('MAE', 0):.2f}</strong> per trip,
            with an R² score of <strong style="color:#58A6FF;">{tm.get('R2', 0):.4f}</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value green">${tm.get('MAE', 0):.4f}</div>
            <div class="metric-sub">Average prediction error</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Root Mean Squared Error</div>
            <div class="metric-value blue">${tm.get('RMSE', 0):.4f}</div>
            <div class="metric-sub">Penalizes larger errors</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{tm.get('R2', 0):.4f}</div>
            <div class="metric-sub">Variance explained</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Prediction Bias</div>
            <div class="metric-value white">${tm.get('Bias (pred - actual)', 0):+.4f}</div>
            <div class="metric-sub">Nearly zero — well calibrated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────── model comparison ─────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-icon">🏆</span>
    <h2>Model Comparison</h2>
</div>
""", unsafe_allow_html=True)

comparison_data = [
    ("XGBoost",                  "2.5708", "1.3612", "0.9225", "1"),
    ("MLP 128→64→32 (This App)", "2.6249", "1.3962", "0.9192", "2"),
    ("Random Forest",            "2.7489", "1.4928", "0.9114", "3"),
    ("SVR (RBF)",                "3.1798", "1.5686", "0.8814", "4"),
    ("KNN (k=10)",               "3.2597", "1.9190", "0.8754", "5"),
    ("Linear Regression (33 ft.)","3.4093","1.9745", "0.8636", "6"),
    ("Linear Regression (9 raw)","7.6066", "4.9021", "0.3213", "7"),
    ("Mean Predictor (baseline)","9.2330", "5.8533", "~0.0",   "8"),
]

table_rows = ""
for name, rmse, mae, r2, rank in comparison_data:
    is_mlp = "MLP" in name
    row_class = ' class="mlp-row"' if is_mlp else ''
    rank_class = f"rank-{rank}" if int(rank) <= 3 else ""
    badge = f'<span class="rank-badge {rank_class}">{rank}</span>' if rank_class else f'<span style="color:#6E7681;">{rank}</span>'
    highlight = ' style="color:#FBBF24;"' if is_mlp else ''
    table_rows += f"""<tr{row_class}>
        <td>{badge}</td>
        <td{highlight}>{name}</td>
        <td>${rmse}</td>
        <td>${mae}</td>
        <td>{r2}</td>
    </tr>"""

st.markdown(f"""
<table class="comp-table">
    <thead>
        <tr>
            <th style="width:50px;">#</th>
            <th>Model</th>
            <th>Test RMSE</th>
            <th>Test MAE</th>
            <th>Test R²</th>
        </tr>
    </thead>
    <tbody>{table_rows}</tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card" style="margin-top:0.5rem;">
    <div style="font-size:0.82rem; color:#8B949E; line-height:1.6;">
        💡 The MLP deployed here ranks <strong style="color:#FBBF24;">2nd</strong> overall, just behind XGBoost.
        It was chosen for deployment because it is lightweight (14,721 parameters),
        needs no external ML library at inference time (runs on pure NumPy), and loads in milliseconds.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────── network architecture ─────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-icon">🧠</span>
    <h2>Network Architecture</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="arch-container">
    <div class="arch-block input">
        <div class="arch-label">Input</div>
        <div class="arch-value">33</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-block hidden">
        <div class="arch-label">Hidden 1</div>
        <div class="arch-value">128</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-block hidden">
        <div class="arch-label">Hidden 2</div>
        <div class="arch-value">64</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-block hidden">
        <div class="arch-label">Hidden 3</div>
        <div class="arch-value">32</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-block output">
        <div class="arch-label">Output</div>
        <div class="arch-value">1</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Training config details
arch_details = cfg.get("final_params", {})
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Parameters</div>
        <div class="metric-value white">14,721</div>
        <div class="metric-sub">Trainable weights</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Activation</div>
        <div class="metric-value blue">ReLU</div>
        <div class="metric-sub">All hidden layers</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Optimizer</div>
        <div class="metric-value green">Adam</div>
        <div class="metric-sub">lr = {arch_details.get('lr', 0.0042)}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Loss Function</div>
        <div class="metric-value">MSE</div>
        <div class="metric-sub">Mean Squared Error</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Batch Size</div>
        <div class="metric-value white">{arch_details.get('batch_size', 2048):,}</div>
        <div class="metric-sub">{cfg.get('final_epochs', 60)} epochs trained</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────── features used ────────────────────────────────
with st.expander("📋 Features Used by the Model (33 total)"):
    model_feats = cfg.get("model_features", [])
    groups = {
        "🌍 Raw Coordinates": [f for f in model_feats if f in
            ("pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count")],
        "📏 Distance & Bearing": [f for f in model_feats if f in
            ("dlat_km","dlon_km","dist_km","bearing_sin","bearing_cos")],
        "🕐 Time Features": [f for f in model_feats if f in
            ("hour","dayofweek","month","years_since_2009",
             "is_weekend","is_night","is_rush_hour","post_fare_hike",
             "hour_sin","hour_cos","dow_sin","dow_cos")],
        "✈️ Landmark Distances": [f for f in model_feats if f.startswith(("pu_dist_","do_dist_"))],
        "🏷️ Airport Flags": [f for f in model_feats if f in
            ("pickup_airport","dropoff_airport","airport_trip")],
    }
    for grp, feats in groups.items():
        if feats:
            st.markdown(f"**{grp}** ({len(feats)})")
            st.code(", ".join(feats), language=None)

# ──────────────────────────── how it works ─────────────────────────────────
with st.expander("ℹ️ How It Works & Limitations"):
    st.markdown("""
**Input Processing:**
- You provide pickup/drop-off coordinates, date, time, and passenger count
- The app engineers **33 features**: distances (Haversine, Manhattan), direction (bearing),
  temporal signals (hour, day, cyclical encodings), landmark distances (JFK, LGA, EWR, Times Square),
  and context flags (airport, night, rush hour, post-2012 fare hike)

**Inference:**
- Features are standardized using the same StandardScaler fit on training data
- The MLP runs forward pass: 33 → 128 → 64 → 32 → 1 (ReLU activations, linear output)
- Output is de-standardized to get the fare in USD

**Limitations:**
- 💲 **Historical pricing** — fares reflect 2009–2015 levels (includes the Sep 2012 fare increase)
- 📏 **Straight-line only** — the model uses Haversine distance, not actual driving routes
- 🚗 **No traffic/tolls/tips** — these factors are unknown to the model
- 📊 **High fares under-predicted** — trips above ~$40 are under-predicted by ~$3 on average
- ✈️ **Airport trips** — have larger prediction errors due to flat-rate pricing complexities
    """)

# ──────────────────────────── footer ───────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="margin-bottom:6px;">
        <span style="color:#FBBF24; font-weight:600;">NYC Taxi Fare Predictor</span> ·
        Deep Feedforward Neural Network
    </div>
    <div>
        Built with Streamlit · Model: Keras MLP (33→128→64→32→1) · Dataset: Kaggle NYC Taxi Fare Prediction
    </div>
    <div style="margin-top:4px;">
        Lab Assignment 02 — New York Taxi Fare Prediction Using Deep Feedforward Neural Networks
    </div>
</div>
""", unsafe_allow_html=True)
