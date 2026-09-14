import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "online_news_popularity_model.pkl"

st.set_page_config(
    page_title="Online News Popularity Predictor",
    page_icon="📰",
    layout="wide",
)

@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.title("📰 Online News Popularity Prediction")
st.caption(
    "Predict the expected number of social-network shares using the trained "
    "regression model from the supplied Colab workflow."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run `python train_model.py` after placing "
        "`OnlineNewsPopularity.csv` in this folder."
    )
    st.stop()

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
meta = bundle["metadata"]
features = meta["feature_names"]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model")
    st.write("Algorithm: **Linear SVR**")
    st.write("Kernel: **linear**")
    st.write(f"Input features: **{len(features)}**")
    st.divider()
    st.write("The model predicts `shares`, the continuous target used in the notebook.")

# ---------- Batch prediction ----------
st.subheader("Batch prediction")
uploaded = st.file_uploader(
    "Upload a CSV containing the model's 58 input features",
    type=["csv"],
)

if uploaded is not None:
    try:
        batch = pd.read_csv(uploaded)
        missing = [c for c in features if c not in batch.columns]
        if missing:
            st.error(f"Missing {len(missing)} required columns.")
            st.code(", ".join(missing))
        else:
            X_batch = batch[features].copy()
            X_scaled = scaler.transform(X_batch)
            batch["predicted_shares"] = np.maximum(
                0, model.predict(X_scaled)
            ).round(0).astype(int)

            st.dataframe(batch, use_container_width=True)
            csv = batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                csv,
                "news_popularity_predictions.csv",
                "text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process the CSV: {exc}")

st.divider()

# ---------- Manual prediction ----------
st.subheader("Single article prediction")
st.info(
    "Enter the numerical article statistics used by the notebook. "
    "For convenience, fields are grouped into expandable sections."
)

# Groups based on the original Online News Popularity feature naming.
groups = {
    "Article structure": [
        "n_tokens_title", "n_tokens_content", "n_unique_tokens",
        "n_non_stop_words", "n_non_stop_unique_tokens",
        "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos",
        "average_token_length", "num_keywords",
    ],
    "Publication channel": [
        "data_channel_is_lifestyle", "data_channel_is_entertainment",
        "data_channel_is_bus", "data_channel_is_socmed",
        "data_channel_is_tech", "data_channel_is_world",
    ],
    "Keyword statistics": [
        "kw_min_min", "kw_max_min", "kw_avg_min",
        "kw_min_max", "kw_max_max", "kw_avg_max",
        "kw_min_avg", "kw_max_avg", "kw_avg_avg",
    ],
    "Reference shares": [
        "self_reference_min_shares", "self_reference_max_shares",
        "self_reference_avg_sharess",
    ],
    "Publication day": [
        "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
        "weekday_is_thursday", "weekday_is_friday",
        "weekday_is_saturday", "weekday_is_sunday", "is_weekend",
    ],
    "Topic / LDA": [
        "LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04",
    ],
    "Article sentiment": [
        "global_subjectivity", "global_sentiment_polarity",
        "global_rate_positive_words", "global_rate_negative_words",
        "rate_positive_words", "rate_negative_words",
        "avg_positive_polarity", "min_positive_polarity",
        "max_positive_polarity", "avg_negative_polarity",
        "min_negative_polarity", "max_negative_polarity",
        "title_subjectivity", "title_sentiment_polarity",
        "abs_title_subjectivity", "abs_title_sentiment_polarity",
    ],
}

# Any feature not assigned above is still exposed.
assigned = {x for vals in groups.values() for x in vals}
groups["Other"] = [x for x in features if x not in assigned]

values = {}

def add_input(col):
    default = float(meta["feature_medians"].get(col, 0.0))
    lo = float(meta["feature_mins"].get(col, default - 1))
    hi = float(meta["feature_maxs"].get(col, default + 1))

    # Avoid an unusable widget if a feature has constant training values.
    if not np.isfinite(lo): lo = default - 1
    if not np.isfinite(hi): hi = default + 1
    if lo == hi:
        lo -= 1
        hi += 1

    step = 1.0 if abs(hi - lo) >= 2 else 0.01

    values[col] = st.number_input(
        col,
        min_value=lo,
        max_value=hi,
        value=min(max(default, lo), hi),
        step=step,
        format="%.6f",
        help=f"Training range: {lo:.4g} to {hi:.4g}; default is training median.",
    )

for group_name, cols in groups.items():
    valid_cols = [c for c in cols if c in features]
    if not valid_cols:
        continue

    with st.expander(group_name, expanded=(group_name == "Article structure")):
        cols_ui = st.columns(3)
        for idx, col in enumerate(valid_cols):
            with cols_ui[idx % 3]:
                add_input(col)

if st.button("Predict News Popularity", type="primary", use_container_width=True):
    try:
        row = pd.DataFrame([[values[c] for c in features]], columns=features)
        row_scaled = scaler.transform(row)
        prediction = float(model.predict(row_scaled)[0])
        prediction = max(0.0, prediction)

        st.success(f"Estimated shares: **{prediction:,.0f}**")

        if prediction < 1000:
            label = "Low predicted popularity"
        elif prediction < 5000:
            label = "Moderate predicted popularity"
        elif prediction < 10000:
            label = "High predicted popularity"
        else:
            label = "Very high predicted popularity"

        st.metric("Predicted social shares", f"{prediction:,.0f}")
        st.caption(label)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
