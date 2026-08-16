# src/app.py
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, date, time
from dotenv import load_dotenv
from google import genai
from config import *

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.title("Talknlock Marketing Intelligence")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in your .env file.")

try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("Model artifacts missing. Run generate.py and train.py first.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign Inputs")
    industry = st.selectbox("Industry", INDUSTRIES)
    platform = st.selectbox("Platform", PLATFORMS)
    content_type = st.selectbox("Format", CONTENT_TYPES)
    topic = st.selectbox("Topic", TOPICS)
    
    p_date = st.date_input("Date", value=date.today())
    p_time = st.time_input("Time", value=time(18, 0))
    p_dt = datetime.combine(p_date, p_time)
    
    reach = st.number_input("Target Reach", value=5000)
    spend = st.number_input("Ad Spend ($)", value=100.0)

with col2:
    st.subheader("AI Prediction")
    input_df = pd.DataFrame([{
        "Industry": industry, "Platform": platform, "Content_Type": content_type,
        "Content_Topic": topic, "Posting_Day": p_dt.strftime("%A"),
        "Posting_Hour": p_dt.hour, "Reach": reach, "Ad_Spend": spend
    }])
    
    score = model.predict(input_df)[0]
    st.metric("Predicted Score", f"{score:.1f} / 100")
    
    if st.button("Generate AI Strategy"):
        if not api_key:
            st.error("Missing Gemini API key.")
        else:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"Act as Marketing Director. Analyze format: {content_type}, platform: {platform}, score: {score:.1f}. "
                "Provide exactly 3 ultra-concise, bulleted execution tips. Keep response under 50 words."
            )
            resp = client.models.generate_content(model='gemini-3.5-flash', contents=prompt)
            st.write(resp.text)