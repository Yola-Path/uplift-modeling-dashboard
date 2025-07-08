import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, apply_filters, render_kpis, render_overview, render_charts, render_uplift_table

st.set_page_config(page_title="Uplift Modeling Dashboard", layout="wide")
st.title("📊 Decision Dashboard: Uplift Modeling & Segment Insights")

df = load_data()
filters = apply_filters(df)
df_filtered = filters['filtered']

render_kpis(df_filtered)
render_overview(df_filtered)
render_charts(df_filtered)
render_uplift_table(df_filtered)
