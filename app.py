import pandas as pd
import streamlit as st
import numpy as np
import os
from io import BytesIO

@st.cache_data
def load_file(file):
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

def load_threshold_local(province):
    # Mapping of province names to threshold file names
    province_files = {
        "AJK": "AJK.csv",
        "Balochistan": "Balochistan.csv",
        "Gilgit Baltistan": "GB.csv",
        "Islamabad": "ICT.csv",
        "Sindh": "Sindh.xlsx",
        "KP": "seasonal_thresholds_kp.xlsx"
    }
    threshold_filename = province_files.get(province)
    if threshold_filename is None:
        st.error(f"No threshold file mapping found for {province}.")
        st.stop()
    try:
        if os.path.exists(threshold_filename):
            if threshold_filename.endswith('.xlsx'):
                df = pd.read_excel(threshold_filename)
            else:
                df = pd.read_csv(threshold_filename)
            if df.empty:
                raise ValueError("Local file is empty.")
            return df
        else:
            raise FileNotFoundError(f"Local file '{threshold_filename}' not found.")
    except Exception as e:
        st.error(f"Failed to load local threshold file '{threshold_filename}': {e}. Please ensure the file exists in the same folder as app.py.")
        st.stop()

# Streamlit app title and description
st.title("IDSRS Pakistan: Disease Outbreak Detection App")
st.markdown("**Simplified tool for early detection of disease outbreaks across all provinces using weekly surveillance data.**")

# Province selection
provinces = ["AJK", "Balochistan", "Gilgit Baltistan", "Islamabad", "Sindh", "KP"]
selected_province = st.selectbox("Select Province:", provinces, index=None)

if selected_province is None:
    st.warning("Please select a province to proceed.")
    st.stop()

# Load threshold file from local for selected province
threshold_df = None
progress_bar = st.progress(0)
status = st.empty()
status.text('Initializing...')
progress_bar.progress(10)

threshold_df = load_threshold_local(selected_province)
st.success(f"✅ Threshold file loaded for {selected_province}.")
progress_bar.progress(30)

# Upload new week file (weekly data)
new_file = st.file_uploader("📁 Upload this week's surveillance data (CSV or Excel)", type=['xlsx', 'csv'], help="Upload the weekly export from DHIS2 containing periodname, org levels, and disease columns.")

if new_file is None:
    st.info("👆 Please upload your weekly data file to generate alerts.")
    st.stop()

# Run button
if st.button("🚨 Generate Outbreak Alerts", type="primary"):
    if threshold_df is not None:
        status.text('Loading weekly data...')
        progress_bar.progress(40)
        new_df = load_file(new_file)
        st.success(f"✅ Weekly data loaded. Shape: {new_df.shape[0]} rows x {new_df.shape[1]} columns.")

        # Remove unnecessary columns
        columns_to_remove = ['periodid', 'periodcode', 'perioddescription', 'organisationunitid', 'organisationunitcode', 'organisationunitdescription']
        new_df = new_df.drop(columns=[col for col in columns_to_remove if col in new_df.columns])

        # Dynamic Org levels and Facility_ID (handles variations across provinces)
        org_level_cols = [col for col in new_df.columns if col.startswith('orgunitlevel')]
        org_level_cols.sort(key=lambda x: int(x.replace('orgunitlevel', '')))  # Sort by level
        org_cols = org_level_cols + ['organisationunitname']
        for col in org_cols:
            if col in new_df.columns:
                new_df[col] = new_df[col].fillna('Unknown').astype(str)
        if len(org_level_cols) > 0 and 'organisationunitname' in new_df.columns:
            new_df['Facility_ID'] = new_df[org_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            num_facilities = new_df['Facility_ID'].nunique()
            st.success(f"✅ Processed {num_facilities} health facilities.")
        else:
            st.error("❌ Missing organizational columns (e.g., orgunitlevel1-5 or organisationunitname). Check your data export.")
            st.stop()

        status.text('Parsing week and season...')
        progress_bar.progress(50)
        # Parse periodname (handles KP/Sindh formats)
        if 'periodname' in new_df.columns:
            new_df['periodname'] = new_df['periodname'].astype(str).str.strip()
            patterns = [
                r'Week (\d+) (\d{4})-\d{2}-\d{2} - \d{4}-\d{2}-\d{2}',  # KP/Sindh format
                r'(\d{4})W(\d{1,2})',  # Sindh W1 fallback
            ]
            best_extracted = None
            for pat in patterns:
                extracted = new_df['periodname'].str.extract(pat)
                if extracted.shape[1] == 2:
                    if 'Week' in pat:
                        extracted.columns = ['Week', 'Year']
                    else:
                        extracted.columns = ['Year', 'Week']
                    success = extracted.notna().all(axis=1).sum()
                    if success > 0:
                        best_extracted = extracted
                        break
            if best_extracted is not None:
                # Drop old Year/Week if exist
                for col in ['Year', 'Week']:
                    if col in new_df.columns:
                        new_df = new_df.drop(columns=[col])
                new_df = pd.concat([new_df, best_extracted], axis=1)
                new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
                new_df['Week'] = pd.to_numeric(new_df['Week'], errors='coerce')
                new_df = new_df.dropna(subset=['Year', 'Week'])
                if new_df.empty:
                    st.error("❌ No valid weeks parsed. Check 'periodname' format (e.g., 'Week 40 2025-10-01 - 2025-10-07').")
                    st.stop()
                new_week = new_df['Week'].iloc[0]
                st.success(f"✅ Parsed Week {new_week} of {new_df['Year'].iloc[0]}.")
            else:
                st.error("❌ No pattern matched 'periodname'. Ensure format is 'Week X YYYY-MM-DD - YYYY-MM-DD' or 'YYYYW#'.")
                st.stop()
        else:
            st.error("❌ No 'periodname' column found. Include it in your DHIS2 export.")
            st.stop()

        # Assign season
        def assign_season(week):
            if pd.isna(week):
                return 'Unknown'
            week = int(week)
            if 10 <= week <= 20:
                return 'Spring'
            elif 21 <= week <= 35:
                return 'Summer'
            elif 36 <= week <= 43:
                return 'Autumn'
            else:
                return 'Winter'

        new_df['Season'] = new_df['Week'].apply(assign_season)
        current_season = new_df['Season'].iloc[0]
        st.success(f"✅ Assigned Season: {current_season}")

        status.text('Processing diseases and generating alerts...')
        progress_bar.progress(70)
        # Disease columns and melt
        disease_cols = [col for col in new_df.columns if '(New Cases)' in col or '(New cases)' in col]
        if len(disease_cols) == 0:
            st.error("❌ No disease columns found (e.g., '(New Cases)'). Check your data export.")
            st.stop()
        # Fill NaNs with 0 for weekly data (as it's current reporting)
        new_df[disease_cols] = new_df[disease_cols].fillna(0).astype(int)
        if new_df.empty:
            st.error("❌ DataFrame is empty after parsing.")
            st.stop()
        long_new = pd.melt(new_df, id_vars=['Facility_ID', 'Season'], value_vars=disease_cols, var_name='Disease', value_name='Cases')
        long_new['Cases'] = long_new['Cases'].astype(int)
        num_diseases = len(disease_cols)
        st.success(f"✅ Melted data: {long_new.shape[0]} records across {num_diseases} diseases.")

        # Disease seasons mapping
        all_seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        disease_seasons = {
            # All seasonal diseases: thresholds for every season
            'Measles (New Cases)': all_seasons,
            'Chickenpox/ Varicella (New cases)': all_seasons,
            'Rubella (Congenital Rubella Syndrome (CRS)) (New Cases)': all_seasons,
            'Mumps (New Cases)': all_seasons,
            'Pertussis (New cases)': all_seasons,
            'Influenza-Like Illness (New Cases)': all_seasons,
            'Pneumonia/ALRI (Acute Lower Respiratory Infections) under 5 years (New Cases)': all_seasons,
            'Scabies (New Cases)': all_seasons,
            'Acute Diarrhea (Non-Cholera) (New Cases)': all_seasons,
            'Acute Watery Diarrhea (Suspected Cholera) (New Cases)': all_seasons,
            'Bloody Diarrhea (New Cases)': all_seasons,
            'Typhoid Fever (New Cases)': all_seasons,
            'Salmonellosis (New Cases)': all_seasons,
            'Viral Hepatitis (B, C & D) (New Cases)': all_seasons,
            'Brucellosis (New cases)': all_seasons,
            'Dog Bite (New Cases)': all_seasons,
            'Anthrax (New Cases)': all_seasons,
            'Chikungunya (New Cases)': all_seasons,
            'Malaria (New Cases)': all_seasons,
            'Dengue Fever (New Cases)': all_seasons,
            'Cutaneous Leishmaniasis (New Cases)': all_seasons,
            'Crimean Congo Hemorrhagic Fever (New Cases)': all_seasons,
            'Severe Acute Respiratory Infection (New Cases)': all_seasons,
            'Tuberculosis (New Cases)': all_seasons,
            'COVID-19 (New Cases)': all_seasons,
            'Diphtheria (Probable) (New Cases)': all_seasons,
            'Encephalitis (New Cases)': all_seasons,
            'Meningitis (New Cases)': all_seasons,
            # Add any additional KP-specific diseases here, e.g.:
            # 'New KP Disease (New Cases)': all_seasons,  # Placeholder; update based on your data head

            # Year-Round (unchanged, excluding Other-1/Other-2)
            'Acute Flaccid Paralysis (New Cases)': ['Year-Round'],
            'Botulism (New Cases)': ['Year-Round'],
            'Gonorrhea (New Cases)': ['Year-Round'],
            'HIV/AIDS (New Cases)': ['Year-Round'],
            'Leprosy (New Cases)': ['Year-Round'],
            'Nosocomial Infections (New Cases)': ['Year-Round'],
            'Syphilis (New Cases)': ['Year-Round'],
            'Visceral Leishmaniasis (New Cases)': ['Year-Round'],
            'Neonatal Tetanus (New Cases)': ['Year-Round']
        }

        # Map threshold season for each disease
        threshold_season_map = {}
        for disease, seasons in disease_seasons.items():
            if seasons == all_seasons:
                threshold_season_map[disease] = current_season
            elif seasons == ['Year-Round']:
                threshold_season_map[disease] = 'Year-Round'
        long_new['Threshold_Season'] = long_new['Disease'].map(threshold_season_map).fillna(current_season)

        # Merge with thresholds using Threshold_Season; rename to avoid suffix conflicts
        if 'Season' not in threshold_df.columns:
            st.error("❌ Threshold file missing 'Season' column. Regenerate thresholds.")
            st.stop()
        # Rename threshold_df's Season temporarily to avoid conflict
        threshold_df_renamed = threshold_df.rename(columns={'Season': 'Threshold_Season'})
        alerts = long_new.merge(
            threshold_df_renamed[['Facility_ID', 'Disease', 'Threshold_Season', 'Threshold_95', 'Threshold_99', 'Mean', 'SD']], 
            on=['Facility_ID', 'Disease', 'Threshold_Season'],
            how='left'
        )

        # Define high-priority (year-round zero-tolerance) diseases (alert on >=1 case, expected 0)
        high_priority_diseases = [
            "Crimean Congo Hemorrhagic Fever (New Cases)",
            "Anthrax (New Cases)",
            "Botulism (New Cases)",
            "Diphtheria (Probable) (New Cases)",
            "Neonatal Tetanus (New Cases)",
            "Acute Flaccid Paralysis (New Cases)"
        ]

        # Override thresholds for high-priority diseases (force zero-tolerance)
        is_high_priority = alerts['Disease'].isin(high_priority_diseases)
        alerts.loc[is_high_priority, 'Threshold_95'] = 0
        alerts.loc[is_high_priority, 'Threshold_99'] = 1
        alerts.loc[is_high_priority, 'Mean'] = 0
        alerts.loc[is_high_priority, 'SD'] = 0

        # Initialize Alert_Level to Normal
        alerts['Alert_Level'] = 'Normal'

        has_valid_threshold = alerts['Threshold_95'].notna()

        # For high-priority: Alert if Cases >=1 and valid threshold
        high_priority_mask = is_high_priority & (alerts['Cases'] >= 1) & has_valid_threshold
        alerts.loc[high_priority_mask & (alerts['Cases'] > alerts['Threshold_99']), 'Alert_Level'] = 'High Alert'
        alerts.loc[high_priority_mask & ~(alerts['Cases'] > alerts['Threshold_99']), 'Alert_Level'] = 'Alert'

        # For seasonal (non-high-priority) diseases: Use thresholds, but only if Cases >1
        seasonal_mask = ~is_high_priority & has_valid_threshold & (alerts['Cases'] > 1)
        alerts.loc[seasonal_mask & (alerts['Cases'] > alerts['Threshold_99']), 'Alert_Level'] = 'High Alert'
        alerts.loc[seasonal_mask & (alerts['Cases'] > alerts['Threshold_95']) & ~(alerts['Cases'] > alerts['Threshold_99']), 'Alert_Level'] = 'Alert'

        # Deviation: For high-priority, use Cases (expected 0); for seasonal, from relevant threshold
        alerts['Deviation'] = np.where(
            is_high_priority & (alerts['Alert_Level'] != 'Normal'),
            alerts['Cases'],
            np.where(
                alerts['Alert_Level'] == 'High Alert',
                alerts['Cases'] - alerts['Threshold_99'],
                np.where(
                    alerts['Alert_Level'] == 'Alert',
                    alerts['Cases'] - alerts['Threshold_95'],
                    0
                )
            )
        )

        # Filter: Only non-normal alerts with valid thresholds and positive deviation
        alerts = alerts[
            (alerts['Alert_Level'] != 'Normal') & 
            has_valid_threshold &
            (alerts['Deviation'] > 0)
        ].copy()
        alerts['District'] = alerts['Facility_ID'].str.split('_').str[2]
        alerts = alerts[['Facility_ID', 'District', 'Disease', 'Season', 'Cases', 'Mean', 'SD', 'Threshold_95', 'Threshold_99', 'Alert_Level', 'Deviation']]

        if alerts.empty:
            st.warning("No alerts detected this week—surveillance levels are normal. Monitor trends closely.")
            st.stop()

        # Sort by Deviation descending for simple prioritization
        alerts = alerts.sort_values('Deviation', ascending=False)

        status.text('Applying filters and summarizing...')
        progress_bar.progress(90)

        # Summary metrics
        total_alerts = len(alerts)
        high_alerts = len(alerts[alerts['Alert_Level'] == 'High Alert'])
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Alerts", total_alerts)
        with col_b:
            st.metric("High Alerts", high_alerts)

        st.success(f"**{total_alerts} total alerts** generated—all diseases above seasonal/year-round thresholds.")

        st.markdown("### 📊 Outbreak Alerts (Sorted by Deviation)")
        st.dataframe(alerts.style.format({
            'Cases': '{:.0f}', 'Mean': '{:.1f}', 'SD': '{:.1f}', 
            'Threshold_95': '{:.1f}', 'Threshold_99': '{:.1f}', 'Deviation': '{:.1f}'
        }))

        # Download button
        status.text('Preparing download...')
        progress_bar.progress(100)
        province_key = selected_province.lower().replace(" ", "_")
        csv_buffer = BytesIO()
        alerts.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label=f"💾 Download Alerts CSV (Week {new_week})",
            data=csv_data,
            file_name=f'alerts_{province_key}_week_{new_week}.csv',
            mime='text/csv',
            type="secondary"
        )

    else:
        st.warning("Upload weekly data to generate alerts.")

    progress_bar.empty()
    status.empty()

# Sidebar with instructions and tips
with st.sidebar:
    st.header("📖 Quick Guide")
    st.markdown("""
    1. **Select Province**: Choose from the dropdown.
    2. **Threshold File**: Ensure the matching file is in your app folder (e.g., `seasonal_thresholds_kp.xlsx` for KP).
    3. **Upload Data**: Weekly DHIS2 export with `periodname`, org levels, and `(New Cases)` columns.
    4. **Generate**: Click the button—all above-threshold diseases shown automatically.
    """)
    st.markdown("---")
    st.header("🛠️ How It Works")
    st.markdown("""
    - **Seasonal Diseases**: Alert if >1 case + > Threshold_95 (Alert) or > Threshold_99 (High Alert).
    - **Year-Round Diseases**: Use 'Year-Round' thresholds if available.
    - **High-Priority Diseases** (e.g., CCHF, Anthrax): Force alert on >=1 case (Alert on 1, High on >1); overrides thresholds.
    - **Deviation**: Cases for high-priority; from threshold for others.
    - **Filters**: Excludes 1-case seasonal alerts, zero/negative deviations, and invalid thresholds.
    - **Prioritization**: Sorted by deviation.
    - Fully automated—no tweaks needed.
    """)
    st.markdown("---")
    st.markdown("*Developed by Asad Khan* | *Simplified Multi-Province v1.3*")
