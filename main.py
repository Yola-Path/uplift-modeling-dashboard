import streamlit as st
import pandas as pd
import logging
from utils import render_kpis, render_filters, render_uplift_chart, render_roi_chart, render_cohort_roi_chart, render_table, render_insight_footer

st.set_page_config(page_title="Uplift Modeling Dashboard", layout="wide")
st.title("Uplift Modeling and ROI Analysis Dashboard")

# Setup logging
logging.basicConfig(level=logging.INFO)

@st.cache_data
def load_data():
    df = pd.read_csv("uplift_dashboard_data.csv")
    metric_df = pd.read_csv("uplift_model_metrics.csv")
    return df, metric_df

df, model_metrics = load_data()

# Model selector
available_models = ['t_learner', 's_learner', 'causalml']
default_model = df['best_model'].iloc[0] if 'best_model' in df.columns else available_models[0]
model_choice = st.sidebar.selectbox("Choose Uplift Model", available_models, index=available_models.index(default_model))
st.sidebar.success(f"Currently viewing: **{model_choice}**")

# Apply filters
df_filtered = render_filters(df)

# Rename columns dynamically for selected model
rename_map = {
    'uplift_score': f'uplift_score_{model_choice}',
    'converted': f'converted_{model_choice}',
    'roi': f'roi_{model_choice}',
    'conversion_revenue': f'conversion_revenue_{model_choice}',
    'promo_cost': f'promo_cost_{model_choice}',
    'actual_ctr': f'actual_ctr_{model_choice}'
}
for k, v in rename_map.items():
    if v in df_filtered.columns:
        df_filtered[k] = df_filtered[v]

# Dashboard sections
render_kpis(df_filtered)
render_uplift_chart(df_filtered)
render_roi_chart(df_filtered)
render_cohort_roi_chart(df_filtered)
render_table(df_filtered)
render_insight_footer(df_filtered)

# Model comparison summary
st.markdown("## Model Performance Comparison")
st.dataframe(model_metrics.sort_values(by="qini", ascending=False).reset_index(drop=True))
