import streamlit as st
import pandas as pd
import numpy as np
import os
import re

# -----------------------
# Utility Functions
# -----------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def parse_week_number(period):
    """Extract week number (integer) from periodname like 'Week 23 2024' or '2024W23'."""
    match = re.search(r'\d{1,2}', str(period))
    return int(match.group()) if match else None

def determine_season(week):
    """Assign season based on week number."""
    if 6 <= week <= 18:
        return 'Spring'
    elif 19 <= week <= 31:
        return 'Summer'
    elif 32 <= week <= 44:
        return 'Autumn'
    else:
        return 'Winter'

def get_threshold_file(province):
    """Load existing province-specific threshold file."""
    filename = f"{province}_threshold.csv"
    if os.path.exists(filename):
        st.sidebar.info(f"✅ Loaded threshold file for {province}")
        return pd.read_csv(filename)
    else:
        st.sidebar.error(f"❌ Threshold file not found for {province}. Please add it manually first.")
        st.stop()

def validate_threshold_file(th_df):
    """Ensure threshold file has required columns."""
    required_cols = {'Facility_ID', 'Disease', 'Season', 'Mean', 'SD', 'Threshold_95', 'Threshold_99'}
    missing = required_cols - set(th_df.columns)
    if missing:
        st.error(f"Threshold file is missing columns: {', '.join(missing)}")
        return False
    return True

# -----------------------
# Main Processing Logic
# -----------------------
def process_alerts(data_df, threshold_df, province, top_n, min_deviation):
    """Main alert generation logic."""
    data_df = data_df.copy()
    threshold_df = threshold_df.copy()

    # Standardize column names
    data_df.columns = data_df.columns.str.strip()

    # Parse week and season
    data_df['Week_Number'] = data_df['periodname'].apply(parse_week_number)
    data_df['Season'] = data_df['Week_Number'].apply(determine_season)

    # Build Facility_ID
    data_df['Facility_ID'] = data_df[['province', 'district', 'tehsil', 'ucName', 'facilityname']].astype(str).agg('-'.join, axis=1)

    # Merge thresholds
    merged = pd.merge(
        data_df,
        threshold_df,
        on=['Facility_ID', 'Disease', 'Season'],
        how='left'
    )

    # Remove missing threshold rows
    merged = merged.dropna(subset=['Threshold_95', 'Threshold_99'], how='all')

    # Compute deviation
    merged['Deviation'] = merged['Current_Week'] - merged['Mean']

    # Detect threshold crossings
    merged['Crossed_99'] = merged['Current_Week'] > merged['Threshold_99']
    merged['Crossed_95'] = (merged['Current_Week'] > merged['Threshold_95']) & (~merged['Crossed_99'])

    # Include only records crossing any threshold
    alerts = merged[(merged['Crossed_95']) | (merged['Crossed_99'])].copy()

    # Compute deviation percentage
    alerts['Deviation_%'] = np.where(
        alerts['Mean'] > 0,
        ((alerts['Current_Week'] - alerts['Mean']) / alerts['Mean']) * 100,
        0
    ).round(1)

    # Apply minimum deviation filter
    alerts = alerts[alerts['Deviation_%'] >= min_deviation]

    # Sort alerts
    alerts = alerts.sort_values(['Crossed_99', 'Deviation_%'], ascending=[False, False])

    # Return top N
    return alerts.head(top_n)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Disease Alert Generator", layout="wide")

st.title("📊 Automated Disease Alert Generator")

st.sidebar.header("⚙️ Configuration")

province = st.sidebar.selectbox(
    "Select Province:",
    ["Punjab", "Sindh", "Balochistan", "Khyber Pakhtunkhwa"]
)

# Load saved threshold automatically
threshold_df = get_threshold_file(province)
if not validate_threshold_file(threshold_df):
    st.stop()

# Ask only for current week's data
uploaded_data = st.sidebar.file_uploader("⬆️ Upload Current Week Data CSV:", type=["csv"])
if not uploaded_data:
    st.info("Please upload the current week’s data file to generate alerts.")
    st.stop()

data_df = load_csv(uploaded_data)

# Sidebar filters
top_n = st.sidebar.slider("Show top N alerts:", 5, 50, 15)
min_deviation = st.sidebar.slider("Minimum deviation %:", 0, 200, 50, 10)

# Generate alerts
alerts_df = process_alerts(data_df, threshold_df, province, top_n, min_deviation)

# Display results
st.subheader("🚨 Disease Alerts")
st.dataframe(alerts_df)

if not alerts_df.empty:
    csv = alerts_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Alerts CSV", csv, "alerts.csv", "text/csv")
else:
    st.success("✅ No threshold crossings detected for this week.")
