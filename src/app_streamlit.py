# src/app_streamlit.py
import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px

st.set_page_config(page_title="Retail Forecast Dashboard", layout="wide")
st.title("Retail Sales — Forecast & Model Results")

default_outputs = "outputs"  # change if you used a different folder

st.sidebar.header("Data source")
output_dir = st.sidebar.text_input("Outputs directory", value=default_outputs)

def pick_csv(path_pattern):
    files = glob.glob(path_pattern)
    return files[0] if files else None

# Auto-load if present
daily_csv = pick_csv(os.path.join(output_dir, "daily_sales", "*.csv"))
lr_csv = pick_csv(os.path.join(output_dir, "lr_test_predictions", "*.csv"))
logr_csv = pick_csv(os.path.join(output_dir, "logr_test_predictions", "*.csv"))
future_csv = pick_csv(os.path.join(output_dir, "future_lr_predictions", "*.csv"))

uploaded_daily = st.file_uploader("Upload daily_sales CSV (optional)", type="csv")
uploaded_lr = st.file_uploader("Upload LR predictions CSV (optional)", type="csv")
uploaded_logr = st.file_uploader("Upload Logistic predictions CSV (optional)", type="csv")
uploaded_future = st.file_uploader("Upload future LR predictions CSV (optional)", type="csv")

def load_csv_choice(uploaded, auto_path):
    if uploaded:
        return pd.read_csv(uploaded, parse_dates=["date"])
    elif auto_path:
        return pd.read_csv(auto_path, parse_dates=["date"])
    else:
        return None

df_daily = load_csv_choice(uploaded_daily, daily_csv)
df_lr = load_csv_choice(uploaded_lr, lr_csv)
df_logr = load_csv_choice(uploaded_logr, logr_csv)
df_future = load_csv_choice(uploaded_future, future_csv)

if df_daily is None:
    st.warning("No daily_sales CSV found. Run pipeline first and place outputs in the outputs/ folder or upload CSVs here.")
else:
    st.subheader("Daily Revenue (sample)")
    st.dataframe(df_daily.sort_values("date").head(20))

if df_daily is not None and df_lr is not None:
    st.subheader("Actual vs Predicted (Linear Regression)")
    # Merge on date
    merged = pd.merge(df_daily[['date','daily_revenue']], df_lr[['date','prediction']], on='date', how='inner')
    fig = px.line(merged.sort_values("date"), x='date', y=['daily_revenue', 'prediction'], labels={'value':'Revenue','variable':'Series'})
    st.plotly_chart(fig, use_container_width=True)

if df_logr is not None:
    st.subheader("Logistic Regression: Predicted Growth Probability")
    # probability column often in form: "[0.7,0.3]" — convert to numeric
    def prob_extract(p):
        try:
            s = str(p)
            # Attempt to parse list-like strings
            if "[" in s and "," in s:
                parts = s.strip("[]").split(",")
                # probability of class 1 is last element (depending on lib); try second
                return float(parts[-1])
            return float(s)
        except:
            return None
    df_logr['prob_growth'] = df_logr['probability'].apply(prob_extract)
    fig2 = px.scatter(df_logr.sort_values("date"), x='date', y='prob_growth', color='prediction', labels={'prob_growth':'P(growth)'})
    st.plotly_chart(fig2, use_container_width=True)

if df_future is not None:
    st.subheader("Future Forecast (Linear Regression)")
    st.dataframe(df_future.sort_values("date").head(30))
    fig3 = px.line(df_future.sort_values("date"), x='date', y='prediction', labels={'prediction':'Predicted Revenue'})
    st.plotly_chart(fig3, use_container_width=True)
