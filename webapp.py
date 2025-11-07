import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. Set page config FIRST (do NOT move) ---
st.set_page_config(page_title="Khet Sahayak · Crop Recommendation", layout="wide")

# --- 2. Optional: style only labels as black ---
st.markdown("""
    <style>
    label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
        color: #18683A !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Page Heading ---
st.markdown("""
<div style="background:#fff;border-radius:18px;padding:38px 25px;box-shadow:0 10px 30px rgba(44,68,116,0.08);margin:40px auto 24px auto;max-width:600px;">
    <div style="text-align:center;margin-bottom:36px;">
        <span style="font-size:38px;">🌱</span>
        <h1 style="color:#18683A;">Crop Recommendation</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4. Load Data and Train Model ---
df = pd.read_csv("Crop_recommendation.csv")
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

rf = RandomForestClassifier(n_estimators=60, random_state=42)
rf.fit(X, y)

# --- 5. User Input Form ---
st.write("Enter your soil and climate values to get crop recommendation:")

N = st.number_input("Nitrogen (N)", float(df.N.min()), float(df.N.max()), float(df.N.median()))
P = st.number_input("Phosphorus (P)", float(df.P.min()), float(df.P.max()), float(df.P.median()))
K = st.number_input("Potassium (K)", float(df.K.min()), float(df.K.max()), float(df.K.median()))
temperature = st.number_input("Temperature (°C)", float(df.temperature.min()), float(df.temperature.max()), float(df.temperature.median()))
humidity = st.number_input("Humidity (%)", float(df.humidity.min()), float(df.humidity.max()), float(df.humidity.median()))
ph = st.number_input("pH", float(df.ph.min()), float(df.ph.max()), float(df.ph.median()))
rainfall = st.number_input("Rainfall (mm)", float(df.rainfall.min()), float(df.rainfall.max()), float(df.rainfall.median()))

if st.button("Recommend Crop"):
    X_inp = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = rf.predict(X_inp)[0]
    st.success(f"Recommended crop: **{crop}**")

st.markdown('<div style="text-align:center; color:#111;margin-top:28px;">© 2025 Khet Sahayak. All rights reserved.</div>', unsafe_allow_html=True)


