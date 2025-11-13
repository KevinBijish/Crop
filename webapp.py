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
        color: #00a859 !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------- LANGUAGE SELECTION -----------------
langs = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}
lang = st.selectbox("Language / भाषा / ਭਾਸ਼ਾ", list(langs.keys()), index=0, key="langbox")
cur_lang = langs[lang]

# ------------- LABELS IN THREE LANGUAGES -----------------
labels = {
    'en': {
        'title': "Crop Recommendation",
        'desc': "Enter your soil and climate values to get crop recommendation:",
        'nitrogen': "Nitrogen (N)",
        'phosphorus': "Phosphorus (P)",
        'potassium': "Potassium (K)",
        'temperature': "Temperature (°C)",
        'humidity': "Humidity (%)",
        'ph': "pH",
        'rainfall': "Rainfall (mm)",
        'submit': "Recommend Crop",
        'result': "Recommended crop:",
        'copyright': "© 2025 Khet Sahayak. All rights reserved."
    },
    'hi': {
        'title': "फसल सिफारिश",
        'desc': "फसल की सिफारिश पाने के लिए अपनी मिट्टी और जलवायु के मान दर्ज करें:",
        'nitrogen': "नाइट्रोजन (N)",
        'phosphorus': "फास्फोरस (P)",
        'potassium': "पोटेशियम (K)",
        'temperature': "तापमान (°C)",
        'humidity': "आर्द्रता (%)",
        'ph': "पीएच",
        'rainfall': "वर्षा (मिमी)",
        'submit': "फसल की सिफारिश करें",
        'result': "अनुशंसित फसल:",
        'copyright': "© 2025 खेत सहायक। सर्वाधिकार सुरक्षित।"
    },
    'pa': {
        'title': "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼",
        'desc': "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼ ਲੈਣ ਲਈ ਆਪਣੀ ਮਿੱਟੀ ਅਤੇ ਜਲਵਾਯੂ ਦੀਆਂ ਕੀਮਤਾਂ ਦਾਖਲ ਕਰੋ:",
        'nitrogen': "ਨਾਈਟ੍ਰੋਜਨ (N)",
        'phosphorus': "ਫਾਸਫੋਰਸ (P)",
        'potassium': "ਪੋਟਾਸ਼ੀਅਮ (K)",
        'temperature': "ਤਾਪਮਾਨ (°C)",
        'humidity': "ਨਮੀ (%)",
        'ph': "ਪੀਐਚ",
        'rainfall': "ਵਰਖਾ (ਮਿਮੀ)",
        'submit': "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼ ਕਰੋ",
        'result': "ਸਿਫਾਰਸ਼ ਕੀਤੀ ਫਸਲ:",
        'copyright': "© 2025 ਖੇਤ ਸਹਾਇਕ। ਸਾਰੇ ਅਧਿਕਾਰ ਰਾਖਵੇਂ।"
    }
}[cur_lang]

# --- 3. Page Heading ---
st.markdown(f"""
<div style="background:#fff;border-radius:18px;padding:38px 25px;box-shadow:0 10px 30px rgba(44,68,116,0.08);margin:40px auto 24px auto;max-width:600px;">
    <div style="text-align:center;margin-bottom:36px;">
        <span style="font-size:38px;">🌱</span>
        <h1 style="color:#18683A;">{labels['title']}</h1>
    </div>
    <p style="text-align:center;color:#555;">{labels['desc']}</p>
</div>
""", unsafe_allow_html=True)

# --- 4. Load Data and Train Model ---
df = pd.read_csv("Crop_recommendation.csv")
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']
rf = RandomForestClassifier(n_estimators=60, random_state=42)
rf.fit(X, y)

# --- 5. User Input Form ---
N = st.number_input(labels['nitrogen'], float(df.N.min()), float(df.N.max()), float(df.N.median()))
P = st.number_input(labels['phosphorus'], float(df.P.min()), float(df.P.max()), float(df.P.median()))
K = st.number_input(labels['potassium'], float(df.K.min()), float(df.K.max()), float(df.K.median()))
temperature = st.number_input(labels['temperature'], float(df.temperature.min()), float(df.temperature.max()), float(df.temperature.median()))
humidity = st.number_input(labels['humidity'], float(df.humidity.min()), float(df.humidity.max()), float(df.humidity.median()))
ph = st.number_input(labels['ph'], float(df.ph.min()), float(df.ph.max()), float(df.ph.median()))
rainfall = st.number_input(labels['rainfall'], float(df.rainfall.min()), float(df.rainfall.max()), float(df.rainfall.median()))

if st.button(labels['submit']):
    X_inp = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = rf.predict(X_inp)[0]
    st.success(f"{labels['result']} **{crop}**")

st.markdown(f'<div style="text-align:center; color:#111;margin-top:28px;">{labels["copyright"]}</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center; color:#111;margin-top:28px;">© 2025 Khet Sahayak. All rights reserved.</div>', unsafe_allow_html=True)


