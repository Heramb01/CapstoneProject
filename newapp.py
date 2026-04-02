import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="User Segment Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0a0e1a;
}
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
.chart-box {
    background: #111827;
    border: 1px solid #1f2d42;
    border-radius: 16px;
    padding: 1.2rem 1rem 0.5rem 1rem;
    margin-top: 0.5rem;
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
    box-shadow: 0 8px 24px rgba(110,231,247,0.3) !important;
}
hr {
    border: none;
    border-top: 1px solid #1f2d42;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
FEATURES = [
    'Age', 'Gender', 'Income Level',
    'Time Spent Online (hrs/weekday)',
    'Time Spent Online (hrs/weekend)',
    'Likes and Reactions',
    'Click-Through Rates (CTR)',
]
NUMERIC = [
    'Time Spent Online (hrs/weekday)',
    'Time Spent Online (hrs/weekend)',
    'Likes and Reactions',
    'Click-Through Rates (CTR)',
]
CATEGORICAL = ['Age', 'Gender', 'Income Level']
RADAR_LABELS = ['Weekday\nOnline', 'Weekend\nOnline', 'Likes &\nReactions', 'CTR']

SEGMENTS = {
    0: dict(
        name="Weekend Warriors", emoji="🏄", color="#f97316",
        bg="linear-gradient(135deg,#1a0a00,#2d1500)",
        border="#f97316", badge_bg="rgba(249,115,22,0.18)", badge_color="#fdba74",
        desc="High weekend online activity with moderate weekday usage. Respond well to weekend-timed campaigns.",
        traits=["Weekend-heavy browsing","Moderate CTR","Ad-responsive","Male-skewed","Mid-high income"],
        tip="Schedule ads Friday evening through Sunday. Lifestyle, travel, and entertainment perform best.",
    ),
    1: dict(
        name="Engaged Professionals", emoji="💼", color="#6ee7f7",
        bg="linear-gradient(135deg,#00101a,#001f2d)",
        border="#6ee7f7", badge_bg="rgba(110,231,247,0.12)", badge_color="#67e8f9",
        desc="Balanced daily usage with the highest likes and reactions. High earners active across the week.",
        traits=["Balanced activity","Highest engagement","High income (100k+)","Male-skewed","Strong CTR"],
        tip="Premium products and professional services perform well. Target during lunch and evening hours.",
    ),
    2: dict(
        name="Low-Key Users", emoji="🌙", color="#a78bfa",
        bg="linear-gradient(135deg,#0d0015,#1a0030)",
        border="#a78bfa", badge_bg="rgba(167,139,250,0.12)", badge_color="#c4b5fd",
        desc="Moderate online presence with consistent but understated engagement. Lower CTR needs non-intrusive formats.",
        traits=["Consistent but quiet","Lower CTR","Mid income (60k-80k)","Male-skewed","Weekend browsing"],
        tip="Native ads and content marketing outperform display ads. Focus on value and trust-building.",
    ),
    3: dict(
        name="Active Explorers", emoji="🔭", color="#34d399",
        bg="linear-gradient(135deg,#001510,#002820)",
        border="#34d399", badge_bg="rgba(52,211,153,0.12)", badge_color="#6ee7b7",
        desc="Highest overall online time but lower reaction counts. Breadth-over-depth browsers.",
        traits=["High time online","Lower reactions","Female-skewed","Mid income (60k-80k)","Broad interests"],
        tip="Retargeting and discovery-style ads work well. Cover multiple interest categories for best reach.",
    ),
    4: dict(
        name="Budget Browsers", emoji="💡", color="#f472b6",
        bg="linear-gradient(135deg,#1a0010,#2d0020)",
        border="#f472b6", badge_bg="rgba(244,114,182,0.12)", badge_color="#f9a8d4",
        desc="Moderate activity, lowest engagement metrics. Price-sensitive — responds to value-driven messaging.",
        traits=["Moderate activity","Lowest engagement","Female-skewed","Lowest income (0-20k)","Value-driven"],
        tip="Promotions, deals, and freemium offers convert best. Keep ad creative simple and direct.",
    ),
}

# ── MODEL (trained + cached once at startup) ─────────────────────────────────
# THE FIX: everything — labels, PCA coords, normalised means — is computed
# INSIDE the cached function and returned as plain data structures.
# No DataFrame mutation outside the cache, no runtime filtering by Cluster.

@st.cache_resource(show_spinner="Training model…")
def load_model():
    df = pd.read_csv("user_profiles_for_ads.csv")

    pre = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    pipe = Pipeline(steps=[
        ('pre', pre),
        ('km',  KMeans(n_clusters=5, random_state=42, n_init=10)),
    ])
    pipe.fit(df[FEATURES])

    labels = pipe.named_steps['km'].labels_
    df['Cluster'] = labels                          # ← assigned inside cache

    # PCA for scatter map
    X_proc = pipe.named_steps['pre'].transform(df[FEATURES])
    pca    = PCA(n_components=2, random_state=42)
    xy     = pca.fit_transform(X_proc)
    df['px'] = xy[:, 0]
    df['py'] = xy[:, 1]

    # Per-feature normalisation bounds
    f_min = df[NUMERIC].min().values
    f_max = df[NUMERIC].max().values

    # Normalised cluster means (0-1) — used by radar charts
    raw   = df.groupby('Cluster')[NUMERIC].mean()
    normed = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    return pipe, pca, df, f_min, f_max, normed


pipe, pca_model, df, f_min, f_max, cluster_norm_means = load_model()

# ── CHART BUILDERS ────────────────────────────────────────────────────────────

def chart_radar_user_vs_avg(user_vals_norm, cluster_id):
    """Radar: user's normalised values vs assigned cluster average."""
    seg   = SEGMENTS[cluster_id]
    avg   = cluster_norm_means.loc[cluster_id].tolist()
    user  = user_vals_norm                              # already 0-1

    fig = go.Figure()

    # Segment average — filled polygon
    fig.add_trace(go.Scatterpolar(
        r     = avg + [avg[0]],
        theta = RADAR_LABELS + [RADAR_LABELS[0]],
        fill  = 'toself',
        name  = f'{seg["name"]} avg',
        line  = dict(color=seg['color'], width=2),
        fillcolor = seg['color'] + '28',
    ))

    # User — dotted overlay
    fig.add_trace(go.Scatterpolar(
        r     = user + [user[0]],
        theta = RADAR_LABELS + [RADAR_LABELS[0]],
        fill  = 'toself',
        name  = 'You',
        line  = dict(color='#ffffff', width=2, dash='dot'),
        fillcolor = 'rgba(255,255,255,0.07)',
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor='#1e293b', tickfont=dict(color='#475569', size=9),
            ),
            angularaxis=dict(
                tickfont=dict(color='#94a3b8', size=11),
                gridcolor='#1e293b',
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94a3b8', size=11), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=60, r=60),
        height=380,
        title=dict(
            text="Your Digital Personality Shape vs Segment Average",
            font=dict(color='#94a3b8', size=12), x=0.5,
        ),
    )
    return fig


def chart_all_segments_radar():
    """Radar showing all 5 segment averages overlaid."""
    fig = go.Figure()
    for cid, seg in SEGMENTS.items():
        vals = cluster_norm_means.loc[cid].tolist()
        fig.add_trace(go.Scatterpolar(
            r     = vals + [vals[0]],
            theta = RADAR_LABELS + [RADAR_LABELS[0]],
            fill  = 'toself',
            name  = f'{seg["emoji"]} {seg["name"]}',
            line  = dict(color=seg['color'], width=2),
            fillcolor = seg['color'] + '15',
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor='#1e293b', tickfont=dict(color='#475569', size=9),
            ),
            angularaxis=dict(
                tickfont=dict(color='#94a3b8', size=11),
                gridcolor='#1e293b',
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94a3b8', size=11), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=60, r=60),
        height=420,
        title=dict(
            text="All 5 Segment Profiles — Side by Side",
            font=dict(color='#94a3b8', size=12), x=0.5,
        ),
    )
    return fig


def chart_pca_map(user_xy, cluster_id):
    """PCA scatter map: 1,000 users as dots + current user as a glowing star."""
    fig = go.Figure()

    for cid, seg in SEGMENTS.items():
        mask = df['Cluster'] == cid
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'px'],
            y=df.loc[mask, 'py'],
            mode='markers',
            name=f'{seg["emoji"]} {seg["name"]}',
            marker=dict(
                color=seg['color'], size=5,
                opacity=0.45, line=dict(width=0),
            ),
            hovertemplate='%{text}<extra></extra>',
            text=[seg['name']] * mask.sum(),
        ))

    # Current user — star
    user_color = SEGMENTS[cluster_id]['color']
    user_name  = SEGMENTS[cluster_id]['name']
    fig.add_trace(go.Scatter(
        x=[user_xy[0]],
        y=[user_xy[1]],
        mode='markers+text',
        name='⭐ You',
        marker=dict(
            symbol='star', color=user_color, size=22,
            line=dict(color='#ffffff', width=1.5),
        ),
        text=['  You'],
        textposition='middle right',
        textfont=dict(color='#f1f5f9', size=13),
        hovertemplate=f'<b>You → {user_name}</b><extra></extra>',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0d1424',
        xaxis=dict(
            title='Principal Component 1',
            gridcolor='#1e293b', zeroline=False,
            tickfont=dict(color='#475569', size=9),
            title_font=dict(color='#64748b', size=11),
        ),
        yaxis=dict(
            title='Principal Component 2',
            gridcolor='#1e293b', zeroline=False,
            tickfont=dict(color='#475569', size=9),
            title_font=dict(color='#64748b', size=11),
        ),
        legend=dict(
            font=dict(color='#94a3b8', size=10),
            bgcolor='rgba(13,20,36,0.9)',
            bordercolor='#1e293b', borderwidth=1,
        ),
        margin=dict(t=50, b=40, l=50, r=30),
        height=460,
        title=dict(
            text="Where You Fit Among 1,000 Users  (PCA Cluster Map)",
            font=dict(color='#94a3b8', size=12), x=0.5,
        ),
    )
    return fig


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">User Segment<br>Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter a user\'s profile to discover which of the 5 behavioural '
    'segments they belong to — and what that means for advertising strategy.</div>',
    unsafe_allow_html=True,
)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age    = st.selectbox("Age Group",    ['18-24','25-34','35-44','45-54','55-64','65+'])
with c2:
    gender = st.selectbox("Gender",       ['Female','Male'])
with c3:
    income = st.selectbox("Income Level", ['0-20k','20k-40k','40k-60k','60k-80k','80k-100k','100k+'])

st.markdown('<div class="section-label">🕐 Online Behaviour</div>', unsafe_allow_html=True)
c4, c5 = st.columns(2)
with c4:
    weekday_hrs = st.slider("Weekday Online Time (hrs/day)",  0.0, 8.0,  2.5, 0.1)
with c5:
    weekend_hrs = st.slider("Weekend Online Time (hrs/day)",  0.0, 10.0, 4.0, 0.1)

st.markdown('<div class="section-label">💬 Engagement & Ad Metrics</div>', unsafe_allow_html=True)
c6, c7 = st.columns(2)
with c6:
    likes = st.number_input("Likes & Reactions (total)", min_value=0, max_value=15000,
                            value=4500, step=100)
with c7:
    ctr   = st.number_input("Click-Through Rate (CTR)",  min_value=0.0, max_value=1.0,
                            value=0.12, step=0.001, format="%.3f")

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🎯  Predict My Segment")

# ── PREDICTION + OUTPUT ───────────────────────────────────────────────────────
if predict:

    # 1. Build input row
    row = pd.DataFrame([{
        'Age':                             age,
        'Gender':                          gender,
        'Income Level':                    income,
        'Time Spent Online (hrs/weekday)': weekday_hrs,
        'Time Spent Online (hrs/weekend)': weekend_hrs,
        'Likes and Reactions':             likes,
        'Click-Through Rates (CTR)':       ctr,
    }])

    # 2. Predict cluster
    cluster_id = int(pipe.predict(row)[0])
    seg = SEGMENTS[cluster_id]

    # 3. Normalise user's numeric values for radar (0-1 scale)
    raw_vals = [weekday_hrs, weekend_hrs, float(likes), ctr]
    user_norm = [
        float(np.clip((v - f_min[i]) / (f_max[i] - f_min[i] + 1e-9), 0.0, 1.0))
        for i, v in enumerate(raw_vals)
    ]

    # 4. Project user into PCA space for cluster map
    user_proc = pipe.named_steps['pre'].transform(row)
    user_xy   = pca_model.transform(user_proc)[0]

    # ── Segment result card ───────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card" style="background:{seg['bg']}; border:1px solid {seg['border']}33;">
        <span class="result-badge"
              style="background:{seg['badge_bg']}; color:{seg['badge_color']};">
            Cluster {cluster_id}
        </span>
        <div class="result-title" style="color:{seg['color']};">
            {seg['emoji']} {seg['name']}
        </div>
        <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin-bottom:1rem;">
            {seg['desc']}
        </p>
        <div style="margin-bottom:1.2rem;">
            {''.join(f'<span class="trait-chip">{t}</span>' for t in seg['traits'])}
        </div>
        <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.9rem 1.1rem;">
            <span style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;
                         color:{seg['color']};font-weight:600;">💡 Ad Strategy</span>
            <p style="color:#e2e8f0;margin:0.4rem 0 0 0;font-size:0.9rem;line-height:1.5;">
                {seg['tip']}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pct = df['Cluster'].value_counts(normalize=True)[cluster_id] * 100
    st.markdown(
        f'<p style="text-align:center;color:#64748b;font-size:0.85rem;margin:0.8rem 0 0 0;">'
        f'This segment is <strong style="color:{seg["color"]};">{pct:.1f}%</strong>'
        f' of the 1,000 training users.</p>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════════
    # CHART 1 — Radar: You vs Your Segment Average
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">📡 Your Profile vs Segment Average</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "**Filled polygon** = average profile of your segment.  "
        "**Dotted outline** = your values.  "
        "A spike beyond the average means you stand out on that dimension — "
        "e.g. a Weekend Warrior with unusually high CTR."
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(
        chart_radar_user_vs_avg(user_norm, cluster_id),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # CHART 2 — PCA Cluster Map
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">🗺️ Your Place on the Cluster Map</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "PCA compresses all 7 features into 2 dimensions. "
        "Every dot is one of the 1,000 training users, coloured by segment. "
        "The **⭐ star** is you — see exactly where you land among the crowd. "
        "Hover any dot for its segment name."
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(
        chart_pca_map(user_xy, cluster_id),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # CHART 3 — All 5 Segments Radar
    # ════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">🕸️ All 5 Segment Profiles Compared</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "How do all 5 segments differ from each other?  "
        "Weekend Warriors bulge on the weekend axis.  "
        "Engaged Professionals are elevated across the board.  "
        "Budget Browsers stay compact near the centre.  "
        "Compare your segment's shape to the rest."
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(
        chart_all_segments_radar(),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#374151;font-size:0.78rem;padding-bottom:1rem;">'
    'Powered by K-Means · PCA · Scikit-learn · Streamlit'
    '</p>',
    unsafe_allow_html=True,
)
