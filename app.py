import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Anomaly-x | AI Sentinel", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for Rich Aesthetics
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #050a15;
        background-image: radial-gradient(rgba(45, 110, 255, 0.15) 1px, transparent 1px);
        background-size: 30px 30px;
        color: #c9d1d9;
    }
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
    /* Custom Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        animation: fadeIn 0.8s ease-out;
    }
    .safe {
        color: #3fb950;
        text-shadow: 0 0 15px rgba(63, 185, 80, 0.6);
        font-size: 2.5rem;
        margin: 0;
    }
    .danger {
        color: #f85149;
        text-shadow: 0 0 20px rgba(248, 81, 73, 0.8);
        animation: pulse 1s infinite;
        font-size: 2.5rem;
        margin: 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Define column names based on NSL-KDD format
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'label', 'difficulty'
]

# We select 8 crucial features for a simplified, fast UI
features = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate', 'diff_srv_rate']

@st.cache_resource
def load_and_train():
    # Try local data folder first (GitHub friendly), fallback to user's local path
    local_path = "data/KDDTrain+_20Percent.txt"
    absolute_path = r"C:\Users\User\Desktop\Side Projects\datasets\NSL-KDD\KDDTrain+_20Percent.txt"
    
    if os.path.exists(local_path):
        dataset_path = local_path
    elif os.path.exists(absolute_path):
        dataset_path = absolute_path
    else:
        st.error(f"Dataset not found! Please place 'KDDTrain+_20Percent.txt' in the 'data/' directory.")
        return None, None, None
    
    # Load dataset
    df = pd.read_csv(dataset_path, names=columns)
    df_selected = df[features + ['label']].copy()
    
    # Map normal to 0, attacks to 1
    df_selected['target'] = df_selected['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Label Encoding for categorical features
    encoders = {}
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df_selected[col] = le.fit_transform(df_selected[col].astype(str))
        encoders[col] = le
        
    X = df_selected[features]
    y = df_selected['target']
    
    # Train lightweight Random Forest
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Store unique classes for UI dropdowns
    options = {
        'protocol_type': list(encoders['protocol_type'].classes_),
        'service': list(encoders['service'].classes_),
        'flag': list(encoders['flag'].classes_)
    }
    
    return model, encoders, options

st.title("🛡️ Anomaly-x: AI Sentinel")
st.markdown("Real-time Intrusion Detection System powered by Machine Learning on the NSL-KDD matrix.")

with st.spinner("Initializing models and loading dataset..."):
    model, encoders, options = load_and_train()

if model:
    st.sidebar.header("🛠️ Packet Configuration")
    st.sidebar.markdown("Simulate a network packet by tweaking the parameters below:")
    
    protocol_type = st.sidebar.selectbox("Protocol Type", options['protocol_type'], index=options['protocol_type'].index('tcp') if 'tcp' in options['protocol_type'] else 0)
    service = st.sidebar.selectbox("Service", options['service'], index=options['service'].index('http') if 'http' in options['service'] else 0)
    flag = st.sidebar.selectbox("Flag", options['flag'], index=options['flag'].index('SF') if 'SF' in options['flag'] else 0)
    
    src_bytes = st.sidebar.slider("Source Bytes", 0, 50000, 200, help="Number of data bytes from source to destination")
    dst_bytes = st.sidebar.slider("Destination Bytes", 0, 50000, 1000, help="Number of data bytes from destination to source")
    count = st.sidebar.slider("Count", 0, 500, 5, help="Number of connections to the same host in the past two seconds")
    same_srv_rate = st.sidebar.slider("Same Service Rate", 0.0, 1.0, 1.0, help="% of connections to the same service")
    diff_srv_rate = st.sidebar.slider("Different Service Rate", 0.0, 1.0, 0.0, help="% of connections to different services")
    
    # Run Prediction
    input_df = pd.DataFrame([{
        'protocol_type': encoders['protocol_type'].transform([protocol_type])[0],
        'service': encoders['service'].transform([service])[0],
        'flag': encoders['flag'].transform([flag])[0],
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'count': count,
        'same_srv_rate': same_srv_rate,
        'diff_srv_rate': diff_srv_rate
    }])
    
    proba = model.predict_proba(input_df)[0]
    is_anomaly = proba[1] > 0.5
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Active Threat Status")
        if is_anomaly:
            st.markdown(f"<p class='danger'>MALICIOUS</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='safe'>NORMAL</p>", unsafe_allow_html=True)
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'color': '#f85149' if is_anomaly else '#3fb950', 'size': 50}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#30363d"},
                'bar': {'color': '#f85149' if is_anomaly else '#3fb950'},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(63, 185, 80, 0.1)'},
                    {'range': [50, 100], 'color': 'rgba(248, 81, 73, 0.1)'}],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            font={'color': "#c9d1d9", 'family': 'Inter'}, 
            height=250, 
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Intercepted Signature")
        
        categories = ['Source Bytes', 'Dest Bytes', 'Conn Count', 'Same Srv', 'Diff Srv']
        values = [min(src_bytes/10000, 1), min(dst_bytes/10000, 1), min(count/500, 1), same_srv_rate, diff_srv_rate]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
          r=values, theta=categories, fill='toself',
          line_color='#58a6ff' if not is_anomaly else '#f85149',
          fillcolor='rgba(88, 166, 255, 0.3)' if not is_anomaly else 'rgba(248, 81, 73, 0.3)'
        ))
        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
          paper_bgcolor="rgba(0,0,0,0)", font={'color': "#c9d1d9", 'family': 'Inter'},
          height=250, margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown(f"**Protocols**: `{protocol_type}` / `{service}` | **Flag**: `{flag}`")
        st.markdown("</div>", unsafe_allow_html=True)
