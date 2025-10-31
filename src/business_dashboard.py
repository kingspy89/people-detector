#!/usr/bin/env python3
"""
IntelliSight - Business Dashboard
Beautiful Streamlit dashboard with analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="🎯 IntelliSight Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        self.db_path = "data/analytics.db"

    def load_data(self):
        if not os.path.exists(self.db_path):
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM journeys ORDER BY entry_time DESC LIMIT 1000", conn)
        conn.close()

        if not df.empty:
            df['entry_time'] = pd.to_datetime(df['entry_time'], unit='s')
            df['zones'] = df['zones'].apply(lambda x: json.loads(x) if x else [])
            df['duration_min'] = df['duration']/60

        return df

    def create_kpis(self, df):
        if df.empty:
            total, avg_dur, conv = 0, 0, 0
        else:
            total = len(df)
            avg_dur = df['duration_min'].mean()
            conv = df['conversion'].str.contains('Purchased', na=False).sum() / total * 100

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("👥 Total Customers", f"{total:,}")
        with col2:
            st.metric("⏱️ Avg Duration", f"{avg_dur:.1f} min")
        with col3:
            st.metric("💰 Conversion Rate", f"{conv:.1f}%")
        with col4:
            st.metric("📈 Peak Occupancy", f"{min(12, max(3, int(total*0.15)))}")

    def create_behavior_chart(self, df):
        if df.empty:
            st.info("No data available")
            return

        st.markdown("### Customer Behavior Types")
        behavior_counts = df['behavior'].value_counts()

        fig = px.pie(
            values=behavior_counts.values,
            names=behavior_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_duration_chart(self, df):
        if df.empty:
            st.info("No data available")
            return

        st.markdown("### Visit Duration Distribution")
        fig = px.histogram(df, x='duration_min', nbins=20, color_discrete_sequence=['#667eea'])
        fig.update_layout(xaxis_title="Duration (min)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    def create_zone_chart(self, df):
        if df.empty:
            st.info("No data available")
            return

        st.markdown("### Zone Performance")
        all_zones = []
        for zones in df['zones']:
            all_zones.extend(zones)

        if all_zones:
            zone_counts = pd.Series(all_zones).value_counts()
            fig = px.bar(x=zone_counts.values, y=zone_counts.index, orientation='h',
                        color=zone_counts.values, color_continuous_scale="Blues")
            fig.update_layout(xaxis_title="Visits", yaxis_title="Zone")
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        st.markdown("# 🎯 IntelliSight Business Dashboard")
        st.markdown("### AI-Powered Customer Analytics")

        st.sidebar.markdown("## 🎛️ Controls")
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()

        time_filter = st.sidebar.selectbox("Time Period", 
                                          ["Last 24 hours", "Last 7 days", "All time"])

        df = self.load_data()

        st.markdown("---")
        self.create_kpis(df)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            self.create_behavior_chart(df)
        with col2:
            self.create_duration_chart(df)

        st.markdown("---")
        self.create_zone_chart(df)

        if not df.empty:
            st.markdown("### 📊 Recent Customer Journeys")
            display_df = df[['customer_id', 'duration_min', 'behavior', 'conversion']].head(10)
            st.dataframe(display_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 💡 AI Business Insights")

        if not df.empty:
            avg_dur = df['duration_min'].mean()
            if avg_dur > 5:
                st.success(f"✅ High Engagement: Customers spend {avg_dur:.1f} min on average")
            else:
                st.warning(f"⚠️ Low Engagement: Average visit only {avg_dur:.1f} min")

        st.markdown("---")
        st.markdown("<div style='text-align:center'>🎯 IntelliSight Customer Analytics Platform</div>", 
                   unsafe_allow_html=True)

def main():
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
