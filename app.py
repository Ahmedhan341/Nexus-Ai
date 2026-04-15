import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from utils.data_loader import load_default_data
from utils.model_factory import get_model, all_models
from utils.database import create_table, insert_prediction, read_predictions
from utils.pdf_report import generate_pdf

# =============================
# UI/UX Configuration
# =============================
st.set_page_config(page_title="Nexus AI | Advanced ML Dashboard", layout="wide", initial_sidebar_state="expanded")

def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file): return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def apply_custom_style():
    current_dir = os.path.dirname(__file__)
    bg_path = os.path.join(current_dir, 'dashboard_bg.jpg')
    bin_str = get_base64_of_bin_file(bg_path)
    
    st.markdown(f"""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    * {{ font-family: 'Inter', sans-serif; }}
    
    .stApp {{
        background: #0B0E14;
        color: #E0E6ED;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #11151C !important;
        border-right: 1px solid #1F2937;
    }}
    
    /* Professional Cards */
    .css-card {{
        background: #1A1F2B;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }}
    
    .main-header {{
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }}
    
    .sub-text {{
        color: #94A3B8;
        font-size: 16px;
        margin-bottom: 30px;
    }}
    
    /* Input Styling */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }}
    
    /* Button Styling */
    .stButton button {{
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s ease;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }}
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {{
        color: #34D399 !important;
        font-weight: 700 !important;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# =============================
# Authentication Logic
# =============================
ADMIN_USER = "alhandwan"
ADMIN_PASS = "7777"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    apply_custom_style()
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div style="text-align: center; margin-top: 100px;">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">Nexus AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-text">Secure Access to Intelligence System</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In")
            
            if submit:
                if user == ADMIN_USER and pw == ADMIN_PASS:
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# =============================
# Main Dashboard (Nexus AI)
# =============================
apply_custom_style()

# Sidebar Control
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    st.markdown("---")
    st.subheader("⚙️ System Configuration")
    algo_name = st.selectbox("Machine Learning Model", list(all_models().keys()))
    test_size = st.slider("Test Split Ratio", 0.1, 0.5, 0.2)
    uploaded_file = st.file_uploader("Import Dataset (CSV)", type=["csv"])
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# Header
st.markdown('<h1 class="main-header">Nexus AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Advanced Analytics & Predictive Modeling Engine</p>', unsafe_allow_html=True)

create_table()

# Data Processing
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = load_default_data()

# Layout: Overview
col_data, col_target = st.columns([2, 1])

with col_data:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_target:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("🎯 Target Selection")
    target_col = st.selectbox("Select Target Variable", data.columns)
    st.info("The model will be trained to predict this column.")
    st.markdown('</div>', unsafe_allow_html=True)

# Model Training
X = data.drop(target_col, axis=1)
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = get_model(algo_name)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Performance Metrics
st.markdown("### 📈 Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy Score", f"{accuracy*100:.1f}%")
m2.metric("Active Model", algo_name)
m3.metric("Training Samples", len(X_train))
m4.metric("Test Samples", len(X_test))

# Visual Analysis
st.markdown("### 🔍 Visual Analysis")
v_col1, v_col2 = st.columns(2)

with v_col1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1A1F2B')
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Prediction Accuracy Heatmap", color='white')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with v_col2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Model Benchmarking")
    bench_results = {}
    for name, m in all_models().items():
        m.fit(X_train, y_train)
        bench_results[name] = m.score(X_test, y_test)
    st.bar_chart(pd.Series(bench_results))
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PREDICTION ENGINE (FIXED)
# =============================
st.markdown('<div id="prediction-engine"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<h2 style="color: #60A5FA;">🤖 Nexus Prediction Engine</h2>', unsafe_allow_html=True)

st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.write("Enter the features below to generate a real-time prediction using the trained model.")

# Create a grid for inputs
input_cols = st.columns(3)
user_features = {}
feature_names = list(X.columns)

for i, col_name in enumerate(feature_names):
    with input_cols[i % 3]:
        user_features[col_name] = st.number_input(f"{col_name}", value=float(X[col_name].mean()))

st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
if st.button("🚀 Generate Prediction"):
    input_df = pd.DataFrame([user_features])
    prediction = model.predict(input_df)[0]
    
    # Display Result Prominently
    st.markdown(f"""
    <div style="background: #064E3B; border: 1px solid #059669; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: #34D399; margin: 0;">Prediction Result</h3>
        <h1 style="color: #F8FAFC; margin: 10px 0;">{prediction}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Save to DB
    insert_prediction(user_features, prediction)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# History & Export
st.markdown("### 📂 Data Management")
d_col1, d_col2 = st.columns([2, 1])

with d_col1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Prediction History")
    st.dataframe(read_predictions(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with d_col2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("System Reports")
    if st.button("Generate Intelligence Report (PDF)"):
        generate_pdf(algo_name, accuracy)
        if os.path.exists("report.pdf"):
            with open("report.pdf", "rb") as f:
                st.download_button("Download Report", f, "Nexus_AI_Report.pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# Auto-save model
if not os.path.exists("saved_models"): os.makedirs("saved_models")
joblib.dump(model, "saved_models/nexus_model.pkl")
