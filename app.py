import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="User Segment Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Main background */
.stApp {
    background: #0a0e1a;
}

/* Hero section */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6ee7f7 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #8892a4;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 2.5rem;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #6ee7f7;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    margin-top: 2rem;
}

/* Card wrapper */
.card {
    background: #111827;
    border: 1px solid #1f2d42;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
}

/* Segment result card */
.result-card {
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}

.result-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 1rem;
}

.trait-chip {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: #e2e8f0;
    margin: 4px 4px 4px 0;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6ee7f7, #a78bfa) !important;
}

div[data-testid="stNumberInput"] input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6ee7f7, #a78bfa) !important;
    color: #0a0e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(110, 231, 247, 0.3) !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #1f2d42;
    margin: 2rem 0;
}

/* Plotly chart background */
.js-plotly-plot .plotly, .js-plotly-plot .plotly bg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ─── TRAIN THE MODEL ONCE (cached) ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = pd.read_csv("user_profiles_for_ads.csv")

    features = [
        'Age', 'Gender', 'Income Level',
        'Time Spent Online (hrs/weekday)',
        'Time Spent Online (hrs/weekend)',
        'Likes and Reactions',
        'Click-Through Rates (CTR)'
    ]

    numeric_features = [
        'Time Spent Online (hrs/weekday)',
        'Time Spent Online (hrs/weekend)',
        'Likes and Reactions',
        'Click-Through Rates (CTR)'
    ]
    categorical_features = ['Age', 'Gender', 'Income Level']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=5, random_state=42, n_init=10))
    ])

    X = data[features]
    pipeline.fit(X)
    return pipeline, data, features, numeric_features, categorical_features


# ─── SEGMENT METADATA ───────────────────────────────────────────────────────────
SEGMENT_INFO = {
    0: {
        "name": "Weekend Warriors",
        "emoji": "🏄",
        "color": "#f97316",
        "bg": "linear-gradient(135deg, #1a0a00 0%, #2d1500 100%)",
        "border": "#f97316",
        "badge_bg": "rgba(249,115,22,0.18)",
        "badge_color": "#fdba74",
        "description": "High weekend online activity with moderate weekday usage. Engage meaningfully with content and respond well to weekend-timed campaigns.",
        "traits": ["Weekend-heavy browsing", "Moderate CTR", "Ad-responsive", "Male-skewed", "Mid-high income"],
        "ad_tip": "Schedule ads on Friday evenings through Sunday. Lifestyle, travel, and entertainment perform best.",
    },
    1: {
        "name": "Engaged Professionals",
        "emoji": "💼",
        "color": "#6ee7f7",
        "bg": "linear-gradient(135deg, #00101a 0%, #001f2d 100%)",
        "border": "#6ee7f7",
        "badge_bg": "rgba(110,231,247,0.12)",
        "badge_color": "#67e8f9",
        "description": "Balanced daily usage with the highest likes and reactions. High earners who actively engage with content across the week.",
        "traits": ["Balanced activity", "Highest engagement", "High income (100k+)", "Male-skewed", "Strong CTR"],
        "ad_tip": "Premium product ads and professional services perform well. Target during lunch and evening hours.",
    },
    2: {
        "name": "Low-Key Users",
        "emoji": "🌙",
        "color": "#a78bfa",
        "bg": "linear-gradient(135deg, #0d0015 0%, #1a0030 100%)",
        "border": "#a78bfa",
        "badge_bg": "rgba(167,139,250,0.12)",
        "badge_color": "#c4b5fd",
        "description": "Moderate online presence with consistent but understated engagement. Lower CTR means they need compelling, non-intrusive ad formats.",
        "traits": ["Consistent but quiet", "Lower CTR", "Mid income (60k-80k)", "Male-skewed", "Weekend browsing"],
        "ad_tip": "Native ads and content marketing outperform display ads. Focus on value and trust-building.",
    },
    3: {
        "name": "Active Explorers",
        "emoji": "🔭",
        "color": "#34d399",
        "bg": "linear-gradient(135deg, #001510 0%, #002820 100%)",
        "border": "#34d399",
        "badge_bg": "rgba(52,211,153,0.12)",
        "badge_color": "#6ee7b7",
        "description": "High overall online time but lower reaction counts. Avid browsers who explore widely — breadth over depth in content consumption.",
        "traits": ["High time online", "Lower reactions", "Female-skewed", "Mid income (60k-80k)", "Broad interests"],
        "ad_tip": "Retargeting and discovery-style ads work well. Cover multiple interest categories for best reach.",
    },
    4: {
        "name": "Budget Browsers",
        "emoji": "💡",
        "color": "#f472b6",
        "bg": "linear-gradient(135deg, #1a0010 0%, #2d0020 100%)",
        "border": "#f472b6",
        "badge_bg": "rgba(244,114,182,0.12)",
        "badge_color": "#f9a8d4",
        "description": "Moderate activity and the lowest engagement metrics. Price-sensitive segment that responds to value-driven messaging and discounts.",
        "traits": ["Moderate activity", "Lowest engagement", "Female-skewed", "Lowest income (0-20k)", "Value-driven"],
        "ad_tip": "Promotions, deals, and freemium offers convert best. Keep ad creative simple and direct.",
    },
}


# ─── RADAR CHART ────────────────────────────────────────────────────────────────
def make_radar(user_values, cluster_id, data, numeric_features):
    cluster_data = data[data['Cluster'] == cluster_id]
    cluster_means = cluster_data[numeric_features].mean()

    user_vals = [user_values[f] for f in numeric_features]
    cluster_vals = cluster_means.values.tolist()

    # Normalise both against dataset min/max
    col_min = data[numeric_features].min().values
    col_max = data[numeric_features].max().values
    user_norm = [(v - mn) / (mx - mn + 1e-9) for v, mn, mx in zip(user_vals, col_min, col_max)]
    cluster_norm = [(v - mn) / (mx - mn + 1e-9) for v, mn, mx in zip(cluster_vals, col_min, col_max)]

    labels = ['Weekday Online', 'Weekend Online', 'Likes & Reactions', 'CTR']
    color = SEGMENT_INFO[cluster_id]["color"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_norm + [cluster_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Segment Avg',
        line=dict(color=color, width=2),
        fillcolor=color.replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in color else color + '26',
        opacity=0.85
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='You',
        line=dict(color='#f8fafc', width=2, dash='dot'),
        fillcolor='rgba(248,250,252,0.08)',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1f2d42', tickfont=dict(color='#475569', size=10)),
            angularaxis=dict(tickfont=dict(color='#94a3b8', size=12), gridcolor='#1f2d42'),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=20, l=40, r=40),
        height=340,
    )
    return fig


# ─── MAIN UI ────────────────────────────────────────────────────────────────────
pipeline, data, features, numeric_features, categorical_features = train_model()

# Fit clusters on data if not already there
if 'Cluster' not in data.columns:
    data['Cluster'] = pipeline.named_steps['cluster'].labels_

st.markdown('<div class="hero-title">User Segment<br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter a user\'s profile to discover which of the 5 behavioural segments they belong to — and what that means for advertising strategy.</div>', unsafe_allow_html=True)

# ── DEMOGRAPHIC SECTION ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox("Age Group", ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
with col2:
    gender = st.selectbox("Gender", ['Female', 'Male'])
with col3:
    income = st.selectbox("Income Level", ['0-20k', '20k-40k', '40k-60k', '60k-80k', '80k-100k', '100k+'])

# ── ONLINE BEHAVIOUR SECTION ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">🕐 Online Behaviour</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)

with col4:
    weekday_hrs = st.slider("Weekday Online Time (hrs/day)", 0.0, 8.0, 2.5, step=0.1,
                            help="Average hours spent online on a typical weekday")
with col5:
    weekend_hrs = st.slider("Weekend Online Time (hrs/day)", 0.0, 10.0, 4.0, step=0.1,
                            help="Average hours spent online on a typical weekend day")

# ── ENGAGEMENT SECTION ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">💬 Engagement & Ad Metrics</div>', unsafe_allow_html=True)
col6, col7 = st.columns(2)

with col6:
    likes = st.number_input("Likes & Reactions (total)", min_value=0, max_value=15000, value=4500, step=100,
                            help="Total number of likes and reactions made by the user")
with col7:
    ctr = st.number_input("Click-Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.12,
                          step=0.001, format="%.3f",
                          help="Proportion of ad impressions that result in a click (0.00 – 1.00)")

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICT BUTTON ───────────────────────────────────────────────────────────────
predict_clicked = st.button("🎯  Predict My Segment")

if predict_clicked:
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Income Level': income,
        'Time Spent Online (hrs/weekday)': weekday_hrs,
        'Time Spent Online (hrs/weekend)': weekend_hrs,
        'Likes and Reactions': likes,
        'Click-Through Rates (CTR)': ctr,
    }])

    cluster_id = int(pipeline.predict(input_df)[0])
    seg = SEGMENT_INFO[cluster_id]

    st.markdown("<hr>", unsafe_allow_html=True)

    # Result card
    st.markdown(f"""
    <div class="result-card" style="background:{seg['bg']}; border:1px solid {seg['border']}33;">
        <span class="result-badge" style="background:{seg['badge_bg']}; color:{seg['badge_color']};">
            Cluster {cluster_id}
        </span>
        <div class="result-title" style="color:{seg['color']};">
            {seg['emoji']} {seg['name']}
        </div>
        <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin-bottom:1rem;">
            {seg['description']}
        </p>
        <div style="margin-bottom:1.2rem;">
            {''.join(f'<span class="trait-chip">{t}</span>' for t in seg['traits'])}
        </div>
        <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:0.9rem 1.1rem;">
            <span style="font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:{seg['color']}; font-weight:600;">
                💡 Ad Strategy
            </span>
            <p style="color:#e2e8f0; margin:0.4rem 0 0 0; font-size:0.9rem; line-height:1.5;">
                {seg['ad_tip']}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Radar chart
    st.markdown('<div class="section-label" style="margin-top:2rem;">📡 Your Profile vs Segment Average</div>', unsafe_allow_html=True)

    user_values = {
        'Time Spent Online (hrs/weekday)': weekday_hrs,
        'Time Spent Online (hrs/weekend)': weekend_hrs,
        'Likes and Reactions': likes,
        'Click-Through Rates (CTR)': ctr,
    }
    radar_fig = make_radar(user_values, cluster_id, data, numeric_features)
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

    # Cluster size context
    cluster_sizes = data['Cluster'].value_counts(normalize=True) * 100
    segment_pct = cluster_sizes[cluster_id]

    st.markdown(f"""
    <div style="text-align:center; color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
        This segment represents <strong style="color:{seg['color']};">{segment_pct:.1f}%</strong> of the user base in the training dataset.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#374151; font-size:0.78rem; padding-bottom:1rem;">
    Powered by K-Means clustering · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)