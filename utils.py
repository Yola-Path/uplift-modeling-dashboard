import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_filters(df):
    st.sidebar.header("Filter Users")
    segment = st.sidebar.multiselect("User Segment", df['user_segment'].unique(), default=list(df['user_segment'].unique()))
    geo = st.sidebar.multiselect("Region", df['geo'].unique(), default=list(df['geo'].unique()))
    device = st.sidebar.multiselect("Device Type", df['device_type'].unique(), default=list(df['device_type'].unique()))
    uplift_min = st.sidebar.slider("Min Uplift Score", float(df['uplift_score_model'].min()), float(df['uplift_score_model'].max()), float(df['uplift_score_model'].min()))

    df_filtered = df[
        (df['user_segment'].isin(segment)) &
        (df['geo'].isin(geo)) &
        (df['device_type'].isin(device)) &
        (df['uplift_score_model'] >= uplift_min)
    ]
    return df_filtered

def render_kpis(df):
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Uplift", f"{df['uplift_score_model'].mean():.4f}")
    col2.metric("Conversion Rate", f"{df['converted'].mean():.2%}")
    col3.metric("Avg ROI (Treated)", f"${df[df['treatment'] == 1]['roi'].mean():.2f}")

def render_uplift_chart(df):
    st.subheader("Uplift by User Segment")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x='user_segment', y='uplift_score_model', estimator='mean', ax=ax)
    ax.set_title("Average Uplift Score per Segment (Model-Based)")
    st.pyplot(fig)

def render_roi_chart(df):
    st.subheader("ROI by Segment (Treated Users)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df[df['treatment'] == 1], x='user_segment', y='roi', ax=ax)
    ax.set_title("Average ROI per Segment (Treated)")
    st.pyplot(fig)

def render_cohort_roi_chart(df):
    st.subheader("ROI by Segment and Treatment Group")
    df['treatment_group'] = df['treatment'].map({1: 'Treated', 0: 'Control'})
    group_roi = df.groupby(['user_segment', 'treatment_group'])['roi'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=group_roi, x='user_segment', y='roi', hue='treatment_group', ax=ax)
    ax.set_title("Average ROI by Segment and Treatment")
    st.pyplot(fig)

def render_table(df):
    st.subheader("Top Users by Uplift Score")
    top_users = df.sort_values("uplift_score_model", ascending=False).head(10)
    st.dataframe(top_users)

def render_insight_footer(df):
    st.markdown("---")
    st.markdown("**Recommendations:**")
    high_roi_segments = df[df['treatment'] == 1].groupby('user_segment')['roi'].mean().sort_values(ascending=False)
    for segment, roi in high_roi_segments.items():
        st.markdown(f"- Target **{segment}** users: Avg ROI = ${roi:.2f}")
