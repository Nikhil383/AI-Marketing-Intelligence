# src/app.py

import os
from datetime import date, datetime, time

import joblib
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from google import genai

from config import (
    MODEL_PATH,
    INDUSTRIES,
    PLATFORMS,
    CONTENT_TYPES,
    TOPICS
)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Talknlock Marketing Intelligence",
    layout="wide"
)

st.title(
    "Talknlock Marketing Intelligence"
)

st.caption(
    "AI-powered content recommendation system"
)


# ==========================================
# SESSION STATE
# ==========================================

if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

if "ranking" not in st.session_state:
    st.session_state.ranking = None

if "campaign" not in st.session_state:
    st.session_state.campaign = None


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_artifacts():

    return joblib.load(
        MODEL_PATH
    )


try:

    artifacts = load_artifacts()

    model = artifacts["model"]

    historical_performance = (
        artifacts[
            "historical_content_performance"
        ]
    )

except Exception:

    st.error(
        "Model not found. Run generate.py and train.py first."
    )

    st.stop()


# ==========================================
# INPUTS
# ==========================================

st.subheader(
    "Campaign Context"
)

col1, col2 = st.columns(2)


with col1:

    industry = st.selectbox(
        "Industry",
        INDUSTRIES
    )

    platform = st.selectbox(
        "Platform",
        PLATFORMS
    )

    topic = st.selectbox(
        "Topic",
        TOPICS
    )


with col2:

    posting_date = st.date_input(
        "Posting Date",
        value=date.today()
    )

    posting_time = st.time_input(
        "Posting Time",
        value=time(18, 0)
    )

    reach = st.number_input(
        "Target Reach",
        min_value=100,
        value=5000
    )

    spend = st.number_input(
        "Ad Spend ($)",
        min_value=0.0,
        value=100.0
    )


posting_datetime = datetime.combine(
    posting_date,
    posting_time
)


# ==========================================
# BUILD CANDIDATES
# ==========================================

def create_candidates():

    candidates = []

    for content_type in CONTENT_TYPES:

        candidates.append(
            {
                "Industry": industry,

                "Platform": platform,

                "Content_Type": content_type,

                "Content_Topic": topic,

                "Posting_Day":
                    posting_datetime.strftime(
                        "%A"
                    ),

                "Posting_Hour":
                    posting_datetime.hour,

                "Reach":
                    reach,

                "Ad_Spend":
                    spend
            }
        )

    return pd.DataFrame(
        candidates
    )


# ==========================================
# RECOMMEND CONTENT
# ==========================================

if st.button(
    "Recommend Best Content",
    type="primary"
):

    candidates = create_candidates()

    predictions = model.predict(
        candidates
    )

    candidates[
        "Predicted_Score"
    ] = predictions

    candidates[
        "Predicted_Score"
    ] = (
        candidates[
            "Predicted_Score"
        ]
        .clip(0, 100)
        .round(1)
    )

    ranking = (
        candidates
        .sort_values(
            "Predicted_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Add ranking
    ranking.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking) + 1
        )
    )

    # --------------------------------------
    # SAVE RESULT IN SESSION
    # --------------------------------------

    st.session_state.ranking = ranking

    st.session_state.recommendation = (
        ranking.iloc[0]["Content_Type"]
    )

    st.session_state.campaign = {
        "Industry": industry,
        "Platform": platform,
        "Topic": topic,
        "Posting_Day":
            posting_datetime.strftime("%A"),
        "Posting_Hour":
            posting_datetime.hour,
        "Reach": reach,
        "Ad_Spend": spend
    }


# ==========================================
# SHOW RECOMMENDATION
# ==========================================

if st.session_state.recommendation:

    recommendation = (
        st.session_state.recommendation
    )

    ranking = (
        st.session_state.ranking
    )

    st.divider()

    st.subheader(
        "Recommended Content"
    )

    st.success(
        f"Create a **{recommendation}** next."
    )

    best_score = ranking.iloc[0][
        "Predicted_Score"
    ]

    st.metric(
        "Expected Performance Score",
        f"{best_score:.1f} / 100"
    )

    # --------------------------------------
    # RANKING TABLE
    # --------------------------------------

    st.subheader(
        "Content Recommendation Ranking"
    )

    st.dataframe(
        ranking[
            [
                "Rank",
                "Content_Type",
                "Predicted_Score"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------
    # HISTORICAL PERFORMANCE
    # --------------------------------------

    st.subheader(
        "Historical Content Performance"
    )

    st.dataframe(
        historical_performance,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------
    # AI CONTENT GENERATION
    # --------------------------------------

    st.divider()

    if st.button(
        "Generate AI Content"
    ):

        if not api_key:

            st.error("GEMINI_API_KEY not found in .env")

        else:

            recommended_type = (
                st.session_state.recommendation
            )

            campaign = (
                st.session_state.campaign
            )

            client = genai.Client(
                api_key=api_key
            )

            prompt = f"""
You are a senior marketing strategist.

Create a marketing content idea based on the ML recommendation.

Campaign:
Industry: {campaign["Industry"]}
Platform: {campaign["Platform"]}
Topic: {campaign["Topic"]}
Posting Day: {campaign["Posting_Day"]}
Posting Hour: {campaign["Posting_Hour"]}
Target Reach: {campaign["Reach"]}
Ad Spend: ${campaign["Ad_Spend"]:.2f}

ML Recommendation:
Recommended Content Type: {recommended_type}

Generate:

1. Content title
2. Short content concept
3. Suggested hook
4. Call to action

The content type MUST be {recommended_type}.

Keep the answer concise and practical.
"""

            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
                st.subheader("AI-Generated Content")
                st.write(response.text)

            except Exception as e:

                st.error(f"Gemini request failed: {e}")