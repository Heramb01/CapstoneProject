import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0e1a; }

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
    margin-bottom: 2.5rem;
}
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
.result-card {
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-top: 1.5rem;
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
div[data-testid="stNumberInput"] input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
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
}
.stButton > button:hover {
    box-shadow: 0 8px 24px rgba(110, 231, 247, 0.3) !important;
}
hr { border: none; border-top: 1px solid #1f2d42; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── SEGMENT METADATA ───────────────────────────────────────────────────────────
SEGMENT_INFO = {
    0: {
        "name": "Weekend Warriors", "emoji": "🏄", "color": "#f97316",
        "bg": "linear-gradient(135deg, #1a0a00, #2d1500)",
        "border": "#f97316", "badge_bg": "rgba(249,115,22,0.18)", "badge_color": "#fdba74",
        "description": "High weekend online activity with moderate weekday usage. Engage meaningfully with content and respond well to weekend-timed campaigns.",
        "traits": ["Weekend-heavy browsing", "Moderate CTR", "Ad-responsive", "Male-skewed", "Mid-high income"],
        "ad_tip": "Schedule ads on Friday evenings through Sunday. Lifestyle, travel, and entertainment perform best.",
    },
    1: {
        "name": "Engaged Professionals", "emoji": "💼", "color": "#6ee7f7",
        "bg": "linear-gradient(135deg, #00101a, #001f2d)",
        "border": "#6ee7f7", "badge_bg": "rgba(110,231,247,0.12)", "badge_color": "#67e8f9",
        "description": "Balanced daily usage with the highest likes and reactions. High earners who actively engage with content across the week.",
        "traits": ["Balanced activity", "Highest engagement", "High income (100k+)", "Male-skewed", "Strong CTR"],
        "ad_tip": "Premium product ads and professional services perform well. Target during lunch and evening hours.",
    },
    2: {
        "name": "Low-Key Users", "emoji": "🌙", "color": "#a78bfa",
        "bg": "linear-gradient(135deg, #0d0015, #1a0030)",
        "border": "#a78bfa", "badge_bg": "rgba(167,139,250,0.12)", "badge_color": "#c4b5fd",
        "description": "Moderate online presence with consistent but understated engagement. Lower CTR means they need compelling, non-intrusive ad formats.",
        "traits": ["Consistent but quiet", "Lower CTR", "Mid income (60k-80k)", "Male-skewed", "Weekend browsing"],
        "ad_tip": "Native ads and content marketing outperform display ads. Focus on value and trust-building.",
    },
    3: {
        "name": "Active Explorers", "emoji": "🔭", "color": "#34d399",
        "bg": "linear-gradient(135deg, #001510, #002820)",
        "border": "#34d399", "badge_bg": "rgba(52,211,153,0.12)", "badge_color": "#6ee7b7",
        "description": "High overall online time but lower reaction counts. Avid browsers who explore widely — breadth over depth.",
        "traits": ["High time online", "Lower reactions", "Female-skewed", "Mid income (60k-80k)", "Broad interests"],
        "ad_tip": "Retargeting and discovery-style ads work well. Cover multiple interest categories for best reach.",
    },
    4: {
        "name": "Budget Browsers", "emoji": "💡", "color": "#f472b6",
        "bg": "linear-gradient(135deg, #1a0010, #2d0020)",
        "border": "#f472b6", "badge_bg": "rgba(244,114,182,0.12)", "badge_color": "#f9a8d4",
        "description": "Moderate activity and the lowest engagement metrics. Price-sensitive segment that responds to value-driven messaging and discounts.",
        "traits": ["Moderate activity", "Lowest engagement", "Female-skewed", "Lowest income (0-20k)", "Value-driven"],
        "ad_tip": "Promotions, deals, and freemium offers convert best. Keep ad creative simple and direct.",
    },
}


# ─── MODEL + PRE-COMPUTATIONS (cached once) ──────────────────────────────────────
# ROOT FIX FOR THE ValueError:
# The original code called data[data['Cluster'] == cluster_id] inside make_radar()
# at prediction time. But train_model() was a @st.cache_resource function that
# returned the raw DataFrame WITHOUT a Cluster column — the column was only
# added AFTER the call, so it was missing when make_radar() ran.
#
# Solution: add the Cluster column INSIDE the cached function so it is always
# present on the returned DataFrame. Additionally, pre-compute all per-cluster
# statistics inside the cached function so the radar chart never needs to filter
# the DataFrame at runtime at all.

@st.cache_resource
def load_everything():
    data = pd.read_csv("user_profiles_for_ads.csv")

    features = [
        'Age', 'Gender', 'Income Level',
        'Time Spent Online (hrs/weekday)',
        'Time Spent Online (hrs/weekend)',
        'Likes and Reactions',
        'Click-Through Rates (CTR)',
    ]
    numeric_features = [
        'Time Spent Online (hrs/weekday)',
        'Time Spent Online (hrs/weekend)',
        'Likes and Reactions',
        'Click-Through Rates (CTR)',
    ]
    categorical_features = ['Age', 'Gender', 'Income Level']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=5, random_state=42, n_init=10)),
    ])

    X = data[features]
    pipeline.fit(X)

    # Attach cluster labels to data
    data['Cluster'] = pipeline.named_steps['cluster'].labels_

    # PCA for 2-D cluster map
    X_proc = pipeline.named_steps['preprocessor'].transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_proc)
    data['PCA_1'] = coords[:, 0]
    data['PCA_2'] = coords[:, 1]

    # Dataset-wide min/max for normalisation
    col_min = data[numeric_features].min().values
    col_max = data[numeric_features].max().values

    # Pre-compute per-cluster normalised means (used by radar charts)
    raw_means = data.groupby('Cluster')[numeric_features].mean()
    norm_means = (raw_means - raw_means.min()) / (raw_means.max() - raw_means.min() + 1e-9)

    return (
        pipeline, pca, data,
        features, numeric_features, categorical_features,
        col_min, col_max, norm_means,
    )


# ─── CHART: Radar — User vs Segment Average ──────────────────────────────────────
def make_radar(user_values, cluster_id, numeric_features, col_min, col_max, norm_means):
    labels = ['Weekday Online', 'Weekend Online', 'Likes & Reactions', 'CTR']
    color = SEGMENT_INFO[cluster_id]["color"]

    # Normalise user inputs against dataset range
    user_norm = [
        float(np.clip((user_values[f] - col_min[i]) / (col_max[i] - col_min[i] + 1e-9), 0, 1))
        for i, f in enumerate(numeric_features)
    ]
    cluster_norm = norm_means.loc[cluster_id].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_norm + [cluster_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name=f'{SEGMENT_INFO[cluster_id]["name"]} Avg',
        line=dict(color=color, width=2),
        fillcolor=color + '30',
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='You',
        line=dict(color='#f8fafc', width=2, dash='dot'),
        fillcolor='rgba(248,250,252,0.10)',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1f2d42',
                            tickfont=dict(color='#475569', size=9)),
            angularaxis=dict(tickfont=dict(color='#94a3b8', size=12), gridcolor='#1f2d42'),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94a3b8', size=12), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=50, r=50), height=380,
        title=dict(text="Your Digital Personality Shape",
                   font=dict(color='#94a3b8', size=13), x=0.5),
    )
    return fig


# ─── CHART: All 5 Segments Radar ─────────────────────────────────────────────────
def make_all_radar(norm_means):
    labels = ['Weekday Online', 'Weekend Online', 'Likes & Reactions', 'CTR']
    fig = go.Figure()
    for cid, seg in SEGMENT_INFO.items():
        vals = norm_means.loc[cid].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name=f'{seg["emoji"]} {seg["name"]}',
            line=dict(color=seg["color"], width=2),
            fillcolor=seg["color"] + '18',
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1f2d42',
                            tickfont=dict(color='#475569', size=9)),
            angularaxis=dict(tickfont=dict(color='#94a3b8', size=12), gridcolor='#1f2d42'),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94a3b8', size=11), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=50, r=50), height=440,
        title=dict(text="All 5 Segment Profiles Compared",
                   font=dict(color='#94a3b8', size=13), x=0.5),
    )
    return fig


# ─── CHART: PCA Cluster Map ───────────────────────────────────────────────────────
def make_pca_map(data, user_pca, cluster_id):
    fig = go.Figure()
    for cid, seg in SEGMENT_INFO.items():
        subset = data[data['Cluster'] == cid]
        fig.add_trace(go.Scatter(
            x=subset['PCA_1'], y=subset['PCA_2'],
            mode='markers',
            name=f'{seg["emoji"]} {seg["name"]}',
            marker=dict(color=seg["color"], size=5, opacity=0.5, line=dict(width=0)),
            hovertemplate=f'<b>{seg["name"]}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>',
        ))
    # Current user — glowing star
    user_color = SEGMENT_INFO[cluster_id]["color"]
    fig.add_trace(go.Scatter(
        x=[user_pca[0]], y=[user_pca[1]],
        mode='markers+text',
        name='⭐ You',
        marker=dict(symbol='star', color=user_color, size=24, opacity=1.0,
                    line=dict(color='#ffffff', width=1.5)),
        text=['  You'],
        textposition='middle right',
        textfont=dict(color='#f8fafc', size=13, family='Space Grotesk'),
        hovertemplate=(
            f'<b>You → {SEGMENT_INFO[cluster_id]["name"]}</b><br>'
            'PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1424',
        xaxis=dict(title='Principal Component 1', gridcolor='#1f2d42', zeroline=False,
                   tickfont=dict(color='#475569', size=10),
                   title_font=dict(color='#64748b', size=11)),
        yaxis=dict(title='Principal Component 2', gridcolor='#1f2d42', zeroline=False,
                   tickfont=dict(color='#475569', size=10),
                   title_font=dict(color='#64748b', size=11)),
        legend=dict(font=dict(color='#94a3b8', size=11),
                    bgcolor='rgba(13,20,36,0.85)', bordercolor='#1f2d42', borderwidth=1),
        margin=dict(t=50, b=40, l=50, r=30), height=460,
        title=dict(text="Where You Fit Among 1,000 Users",
                   font=dict(color='#94a3b8', size=13), x=0.5),
    )
    return fig


# ─── LOAD ───────────────────────────────────────────────────────────────────────
(pipeline, pca_model, data,
 features, numeric_features, categorical_features,
 col_min, col_max, norm_means) = load_everything()

# ─── HERO ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">User Segment<br>Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter a user\'s profile to discover which of the 5 behavioural '
    'segments they belong to — and what that means for advertising strategy.</div>',
    unsafe_allow_html=True,
)

# ─── INPUT FORM ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.selectbox("Age Group", ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
with c2:
    gender = st.selectbox("Gender", ['Female', 'Male'])
with c3:
    income = st.selectbox("Income Level", ['0-20k', '20k-40k', '40k-60k', '60k-80k', '80k-100k', '100k+'])

st.markdown('<div class="section-label">🕐 Online Behaviour</div>', unsafe_allow_html=True)
c4, c5 = st.columns(2)
with c4:
    weekday_hrs = st.slider("Weekday Online Time (hrs/day)", 0.0, 8.0, 2.5, step=0.1,
                            help="Average hours online on a typical weekday")
with c5:
    weekend_hrs = st.slider("Weekend Online Time (hrs/day)", 0.0, 10.0, 4.0, step=0.1,
                            help="Average hours online on a typical weekend day")

st.markdown('<div class="section-label">💬 Engagement & Ad Metrics</div>', unsafe_allow_html=True)
c6, c7 = st.columns(2)
with c6:
    likes = st.number_input("Likes & Reactions (total)", min_value=0, max_value=15000,
                            value=4500, step=100)
with c7:
    ctr = st.number_input("Click-Through Rate (CTR)", min_value=0.0, max_value=1.0,
                          value=0.12, step=0.001, format="%.3f")

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🎯  Predict My Segment")

# ─── RESULTS ────────────────────────────────────────────────────────────────────
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

    # Project user into PCA space for the cluster map
    user_proc = pipeline.named_steps['preprocessor'].transform(input_df)
    user_pca = pca_model.transform(user_proc)[0]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Segment card ──────────────────────────────────────────────────────────
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
            <span style="font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
                         color:{seg['color']}; font-weight:600;">💡 Ad Strategy</span>
            <p style="color:#e2e8f0; margin:0.4rem 0 0 0; font-size:0.9rem; line-height:1.5;">
                {seg['ad_tip']}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cluster_pct = data['Cluster'].value_counts(normalize=True)[cluster_id] * 100
    st.markdown(
        f'<div style="text-align:center;color:#64748b;font-size:0.85rem;margin:0.8rem 0;">'
        f'This segment represents <strong style="color:{seg["color"]};">{cluster_pct:.1f}%</strong>'
        f' of the 1,000 users in the training dataset.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 1 — Radar: Your profile vs your segment average
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">📡 Your Profile vs Segment Average</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The **filled polygon** shows the average profile of your assigned segment. "
        "The **dotted outline** shows your own values. A spike beyond the average means "
        "you stand out on that dimension — e.g. a Weekend Warrior with unusually high CTR."
    )
    user_vals = {
        'Time Spent Online (hrs/weekday)': weekday_hrs,
        'Time Spent Online (hrs/weekend)': weekend_hrs,
        'Likes and Reactions': likes,
        'Click-Through Rates (CTR)': ctr,
    }
    st.plotly_chart(
        make_radar(user_vals, cluster_id, numeric_features, col_min, col_max, norm_means),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 2 — PCA Cluster Map
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">🗺️ Your Place on the Cluster Map</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "PCA compresses all 7 features into 2 dimensions so the full 1,000-user landscape "
        "can be visualised at once. Each dot is one training user, coloured by segment. "
        "The **⭐ star** is you — see exactly where you land among the crowd."
    )
    st.plotly_chart(
        make_pca_map(data, user_pca, cluster_id),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 3 — All 5 Segments Radar
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">🕸️ All 5 Segment Profiles Compared</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "How do all five segments differ from each other? Weekend Warriors bulge on the weekend "
        "axis. Engaged Professionals fill out the top-right. Budget Browsers stay compact near "
        "the centre. Compare your segment's shape to the others."
    )
    st.plotly_chart(
        make_all_radar(norm_means),
        use_container_width=True, config={"displayModeBar": False},
    )

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#374151;font-size:0.78rem;padding-bottom:1rem;">'
    'Powered by K-Means clustering · PCA · Scikit-learn · Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
