import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    return pd.read_csv("uplift_dashboard_data.csv")

def apply_filters(df):
    st.sidebar.header("🔍 Filter Panel")
    segment = st.sidebar.selectbox("User Segment", ['All'] + sorted(df['user_segment'].unique()))
    geo = st.sidebar.multiselect("Geo Region", options=df['geo'].unique(), default=df['geo'].unique())
    device = st.sidebar.multiselect("Device Type", options=df['device_type'].unique(), default=df['device_type'].unique())

    df_filtered = df.copy()
    if segment != 'All':
        df_filtered = df_filtered[df_filtered['user_segment'] == segment]
    df_filtered = df_filtered[df_filtered['geo'].isin(geo)]
    df_filtered = df_filtered[df_filtered['device_type'].isin(device)]

    return {'filtered': df_filtered}

def render_kpis(df):
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Uplift", f"{df['uplift_score'].mean():.4f}")
    k2.metric("Predicted CTR", f"{df['predicted_ctr'].mean():.2%}")
    k3.metric("Actual CTR", f"{df['actual_ctr'].mean():.2%}")

def render_overview(df):
    st.subheader("📁 User Distribution")
    c1, c2, c3 = st.columns(3)
    c1.bar_chart(df['user_segment'].value_counts())
    c2.bar_chart(df['geo'].value_counts())
    c3.bar_chart(df['device_type'].value_counts())

def render_charts(df):
    st.subheader("🧠 Model Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='user_segment', y='model_score', ax=ax)
    st.pyplot(fig)

    st.subheader("🎯 Uplift by Segment")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x='user_segment', y='uplift_score', ax=ax2, ci=None)
    st.pyplot(fig2)

    st.subheader("📈 Predicted vs Actual CTR")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x='predicted_ctr', y='actual_ctr', hue='user_segment', ax=ax3)
    ax3.plot([0, 0.25], [0, 0.25], '--', color='gray')
    st.pyplot(fig3)

def render_uplift_table(df):
    st.subheader("🚀 Top 10 Users by Uplift")
    st.dataframe(df.sort_values(by='uplift_score', ascending=False).head(10))
    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="filtered_data.csv")
