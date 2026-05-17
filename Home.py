import streamlit as st

# 1. High-End UI Configuration
st.set_page_config(
    page_title="CryptoCast Pro | Deep Learning Forecasting",
    page_icon="⚡",
    layout="wide"
)

# 2. Advanced Neon Dark-Theme CSS Injector
st.markdown("""
    <style>
    /* Global App Container Tweaks */
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Main Hero Banner text styling */
    .hero-title {
        font-size: 54px !important;
        font-weight: 900;
        
        -webkit-background-clip: text;
        -webkit-text-fill-color: #FF2A4B;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 2px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 22px !important;
        color: #00E676;
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.3); /* Optional glow */
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Section Headings with Gradient Bottom Bars */
    .glow-header {
        font-size: 26px !important;
        font-weight: 700;
        color: #ffffff;
        margin-top: 35px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Elegant Cyberpunk Info Cards */
    .feature-card {
        background: linear-gradient(135deg, #131A2A 0%, #0E1422 100%);
        border: 1px solid #1E293B;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #FF9900;
    }
    .card-icon {
        font-size: 32px;
        margin-bottom: 12px;
    }
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .card-body {
        font-size: 14px;
        color: #94A3B8;
        line-height: 1.6;
    }
    
    /* Skills Badge Pills */
    .skill-pill {
        display: inline-block;
        background: rgba(255, 153, 0, 0.1);
        border: 1px solid rgba(255, 153, 0, 0.3);
        color: #FF9900;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 5px;
    }
    .model-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38BDF8;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Segment with Dynamic Header Image
st.markdown("<div class='hero-title'>CRYPTOCAST</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Multi-Horizon Bitcoin Price Forecasting System Using Deep Learning</div>", unsafe_allow_html=True)

# Section 1: Core Problem & Objectives
st.markdown("<div class='glow-header'>🎯 Project Overview & Objective</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("""
    <p style='color: #E2E8F0; font-size: 16px; line-height: 1.7;'>
    Bitcoin prices are highly volatile and influenced by complex temporal patterns. Accurate short-term forecasting is crucial for traders, analysts, and automated trading systems. Traditional statistical methods struggle to capture non-linear dependencies and long-term temporal relationships present in crypto price movements.
    <br><br>
    This project focuses on building deep learning models that learn from historical price sequences to forecast Bitcoin prices over multiple future horizons.
    </p>
    """, unsafe_allow_html=True)
with col_right:
    # Mini system metrics block
    metric_box1, metric_box2 = st.columns(2)
    # metric_box1.metric(label="Target Asset", value="BTC / USD", delta="Context: 60 Days")
    # metric_box2.metric(label="Inference Latency", value="14.2 ms", delta="-2.1ms (Optimized)")

# Section 2: Interactive Grid Columns for Multi-Horizon Targets
st.markdown("<div class='glow-header'>🔮 Multi-Horizon Forecasting Horizons</div>", unsafe_allow_html=True)
h_col1, h_col2, h_col3 = st.columns(3)

with h_col1:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-icon'>🟢</div>
        <div class='card-title'>Next-Day Prediction (1D)</div>
        <div class='card-body'>High-frequency alpha extraction targeting short-term scalping models and instantaneous directional trend detection.</div>
    </div>
    """, unsafe_allow_html=True)
    
with h_col2:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-icon'>🔵</div>
        <div class='card-title'>3-Day Ahead Horizon (3D)</div>
        <div class='card-body'>Intermediate momentum tracking designed to capture macro swing patterns and micro structural shifts in trading velocity.</div>
    </div>
    """, unsafe_allow_html=True)
    
with h_col3:
    st.markdown("""
    <div class='feature-card'>
        <div class='card-icon'>🟣</div>
        <div class='card-title'>7-Day Ahead Outlook (7D)</div>
        <div class='card-body'>Weekly structural forecasting used for strategic position management, portfolio hedging, and risk mitigation profiles.</div>
    </div>
    """, unsafe_allow_html=True)

# Section 3: Tech Matrix & Badges
st.markdown("<div class='glow-header'>🛠️ Deep Learning Architecture & Core Skills</div>", unsafe_allow_html=True)

tech_col1, tech_col2 = st.columns(2)
with tech_col1:
    st.markdown("<p style='color: white; font-weight: 600; font-size: 16px; margin-bottom: 10px;'>Models Evaluated</p>", unsafe_allow_html=True)
    st.markdown("""
        <span class='model-pill'>🧠 Convolutional Neural Networks (CNN)</span>
        <span class='model-pill'>🧠 Recurrent Neural Networks (RNN)</span>
        <span class='model-pill'>🧠 Long Short-Term Memory (LSTM)</span>
        <span class='model-pill'>🧠 Gated Recurrent Units (GRU)</span>
        <span class='model-pill'>🧠 Transformers (Attention Mechanism)</span>
    """, unsafe_allow_html=True)
    
with tech_col2:
    st.markdown("<p style='color: white; font-weight: 600; font-size: 16px; margin-bottom: 10px;'>Skills Gained</p>", unsafe_allow_html=True)
    st.markdown("""
        <span class='skill-pill'>📊 Time Series Analysis</span>
        <span class='skill-pill'>🧬 Sequence Modeling</span>
        <span class='skill-pill'>📉 Multi-Step Prediction</span>
        <span class='skill-pill'>🧪 Financial Forecast Evaluation</span>
        <span class='skill-pill'>🏗️ ML System Design</span>
    """, unsafe_allow_html=True)


