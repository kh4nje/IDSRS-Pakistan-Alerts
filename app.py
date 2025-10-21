import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------------------------------------
# Load local threshold file automatically based on province
# ----------------------------------------------------------
def load_threshold_local(province):
    province_files = {
        "AJK": "AJK.csv",
        "Balochistan": "Balochistan.csv",
        "Gilgit Baltistan": "GB.csv",
        "Islamabad": "ICT.csv",
        "Sindh": "Sindh.xlsx",
        "KP": "seasonal_thresholds_kp.xlsx"
    }

    if province not in province_files:
        st.error(f"No threshold file found for {province}.")
        return None

    file_name = province_files[province]

    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        else:
            df = pd.read_excel(file_name)
        st.success(f"✅ Threshold file loaded for {province}: {file_name}")
        return df
    except Exception as e:
        st.error(f"Error loading threshold file: {e}")
        return None


# ----------------------------------------------------------
# Process alerts and detect threshold crossing
# ----------------------------------------------------------
def process_alerts(data_df, threshold_df, top_n=10, min_deviation=2):
    required_cols = ['province', 'district', 'tehsil', 'ucName', 'facilityname', 'disease', 'week', 'cases']

    # Ensure all required columns exist
    missing = [col for col in required_cols if col not in data_df.columns]
    if missing:
        st.error(f"Missing columns in uploaded data: {missing}")
        return pd.DataFrame()

    # Create a unique Facility ID
    data_df['Facility_ID'] = (
        data_df[['province', 'district', 'tehsil', 'ucName', 'facilityname']]
        .astype(str)
        .agg('-'.join, axis=1)
    )

    # Use only the last week's data
    latest_week = data_df['week'].max()
    data_df = data_df[data_df['week'] == latest_week]

    # Merge with thresholds
    merged = pd.merge(
        data_df,
        threshold_df,
        how='left',
        left_on=['province', 'disease'],
        right_on=['province', 'disease']
    )

    # Calculate deviation
    merged['deviation'] = merged['cases'] - merged['threshold']

    # Identify alerts only where threshold is crossed
    alerts = merged[merged['deviation'] > 0]

    # Rank by deviation
    alerts = alerts.sort_values(by='deviation', ascending=False).head(top_n)

    # Return only relevant columns
    return alerts[['province', 'district', 'tehsil', 'ucName', 'facilityname', 'disease', 'cases', 'threshold', 'deviation', 'week']]


# ----------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------
st.title("🦠 Disease Alert Detection (Weekly Threshold-Based)")
st.markdown("Upload **last week’s line list**, and the system will automatically load the threshold file for the selected province.")

# Select province (to load correct threshold)
province = st.selectbox(
    "Select Province:",
    ["AJK", "Balochistan", "Gilgit Baltistan", "Islamabad", "Sindh", "KP"]
)

# Load threshold automatically
threshold_df = load_threshold_local(province)

# Upload last week data
uploaded_file = st.file_uploader("Upload Last Week Data (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None and threshold_df is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data_df = pd.read_csv(uploaded_file)
        else:
            data_df = pd.read_excel(uploaded_file)

        st.write(f"✅ Data file loaded successfully. Total records: {len(data_df)}")

        # Process and show alerts
        alerts_df = process_alerts(data_df, threshold_df)

        if not alerts_df.empty:
            st.success("🚨 Threshold crossed! Showing priority disease alerts:")
            st.dataframe(alerts_df)
        else:
            st.info("✅ No diseases crossed the threshold this week.")
    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.warning("Please upload the last week data and select a province.")
