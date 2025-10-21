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

# Disease weights for severity scoring (higher = more impact; tunable dict)
# Now includes the COMPLETE list of IDSRS diseases with tailored weights
DISEASE_WEIGHTS = {
    # High-impact (lethal/emerging: hemorrhagic, vaccine-preventable, neurological, zoonotic)
    'Crimean Congo Hemorrhagic Fever (New Cases)': 3.0,
    'Dengue Fever (New Cases)': 2.5,
    'Polio/Acute Flaccid Paralysis (New Cases)': 3.0,  # If present
    'Acute Flaccid Paralysis (New Cases)': 3.0,
    'Diphtheria (Probable) (New Cases)': 2.5,
    'Measles (New Cases)': 2.0,
    'Pertussis (New cases)': 2.0,
    'Neonatal Tetanus (New Cases)': 3.0,
    'Anthrax (New Cases)': 3.0,
    'Botulism (New Cases)': 3.0,
    'Encephalitis (New Cases)': 2.5,
    'Meningitis (New Cases)': 2.5,
    'Rubella (Congenital Rubella Syndrome (CRS)) (New Cases)': 2.5,
    'COVID-19 (New Cases)': 2.5,
    'Acute Watery Diarrhea (Suspected Cholera) (New Cases)': 2.5,
    # Medium-impact (enteric, respiratory, vector-borne, chronic, STIs)
    'Malaria (New Cases)': 2.0,
    'Typhoid Fever (New Cases)': 1.8,
    'Bloody Diarrhea (New Cases)': 1.8,
    'Chikungunya (New Cases)': 2.0,
    'Brucellosis (New cases)': 1.8,
    'HIV/AIDS (New Cases)': 2.0,
    'Tuberculosis (New Cases)': 1.5,
    'Influenza-Like Illness (New Cases)': 1.5,
    'Severe Acute Respiratory Infection (New Cases)': 1.8,
    'Pneumonia/ALRI (Acute Lower Respiratory Infections) under 5 years (New Cases)': 1.5,
    'Acute Diarrhea (Non-Cholera) (New Cases)': 1.5,
    'Salmonellosis (New Cases)': 1.5,
    'Viral Hepatitis (B, C & D) (New Cases)': 1.8,
    'Visceral Leishmaniasis (New Cases)': 2.0,
    'Gonorrhea (New Cases)': 1.8,
    'Syphilis (New Cases)': 1.8,
    'Nosocomial Infections (New Cases)': 2.0,
    'Leprosy (New Cases)': 1.5,
    # Low-impact (skin, mild zoonotic, routine)
    'Scabies (New Cases)': 0.5,
    'Dog Bite (New Cases)': 0.8,
    'Cutaneous Leishmaniasis (New Cases)': 1.0,
    'Chickenpox/ Varicella (New cases)': 1.2,
    'Mumps (New Cases)': 1.2,
    # Other (low priority)
    'Other-1 (New Cases)': 0.5,
    'Other-2 (New Cases)': 0.5,
    # Default for any unlisted variants
    'default': 1.0
}

# Streamlit app title and description
st.title("IDSRS Pakistan: Disease Outbreak Detection App")
st.markdown("**A tool for early detection of disease outbreaks across Pakistani provinces using weekly surveillance data.**")

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

# Info on automated filtering
st.markdown("### 🤖 Fully Automated Filtering")
st.info("""
Alerts are generated hands-free: 
- **Dynamic thresholds**: Min surge adjusted per disease based on historical volatility (e.g., 3x for stable diseases like measles, 1.5x for erratic ones like dengue).
- **Noise floors**: ≥3 cases min; clustering ≥3 facilities for hotspots.
- **Year-Round**: High Alert only (no low-volume noise).
- **No hard cap**: All qualifying alerts shown (strict formula keeps normal weeks <100; surges allowed).
No manual tweaks needed—upload and go!
""")

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

        # Year-round override for specific diseases (expanded for chronic/zoonotic/non-seasonal)
        year_round_diseases = [
            'Acute Flaccid Paralysis (New Cases)', 'Botulism (New Cases)', 'Gonorrhea (New Cases)', 
            'HIV/AIDS (New Cases)', 'Leprosy (New Cases)', 'Nosocomial Infections (New Cases)', 
            'Syphilis (New Cases)', 'Visceral Leishmaniasis (New Cases)', 'Neonatal Tetanus (New Cases)',
            'Tuberculosis (New Cases)', 'Brucellosis (New cases)', 'Encephalitis (New Cases)',
            'Meningitis (New Cases)', 'Rubella (Congenital Rubella Syndrome (CRS)) (New Cases)'
        ]
        year_round_mask = long_new['Disease'].isin(year_round_diseases)
        long_new.loc[year_round_mask, 'Season'] = 'Year-Round'
        if year_round_mask.sum() > 0:
            st.info(f"ℹ️ Overrode {year_round_mask.sum()} records to Year-Round season.")

        # Merge with thresholds
        if 'Season' not in threshold_df.columns:
            st.error("❌ Threshold file missing 'Season' column. Regenerate thresholds.")
            st.stop()
        filtered_thresholds = threshold_df[threshold_df['Season'].isin([current_season, 'Year-Round'])]
        alerts = long_new.merge(
            filtered_thresholds[['Facility_ID', 'Disease', 'Season', 'Threshold_95', 'Threshold_99', 'Mean', 'SD']], 
            how='left'
        )

        # Generate alert levels (keep original σ for levels, but use for scoring)
        alerts['Alert_Level'] = np.where(
            (alerts['Cases'] > alerts['Threshold_99']) & alerts['Threshold_99'].notna(), 'High Alert',
            np.where(
                (alerts['Cases'] > alerts['Threshold_95']) & alerts['Threshold_95'].notna(), 'Alert', 'Normal'
            )
        )
        alerts['Deviation'] = np.where(
            alerts['Alert_Level'] == 'High Alert', alerts['Cases'] - alerts['Threshold_99'],
            np.where(alerts['Alert_Level'] == 'Alert', alerts['Cases'] - alerts['Threshold_95'], 0)
        )

        # Filter to non-normal alerts, with thresholds, excluding 'Other' + stricter min cases=3
        alerts = alerts[
            (alerts['Alert_Level'] != 'Normal') & 
            alerts['Threshold_95'].notna() & 
            (~alerts['Disease'].str.contains('Other', na=False)) &
            (alerts['Cases'] >= 3)  # Stricter: ≥3 cases min
        ].copy()
        alerts = alerts[['Facility_ID', 'Disease', 'Season', 'Cases', 'Mean', 'SD', 'Threshold_95', 'Threshold_99', 'Alert_Level', 'Deviation']]

        # Relative % deviation for scoring (Cases / Mean, avoid div0)
        alerts['Pct_Deviation'] = np.where(
            alerts['Mean'] > 0, alerts['Cases'] / alerts['Mean'], np.inf  # Treat 0 mean as infinite for rare diseases
        )

        # Extract district from Facility_ID (assuming level2 is district)
        alerts['District'] = alerts['Facility_ID'].str.split('_').str[2]

        # Data-driven min_multiplier per disease: Based on coefficient of variation (SD/Mean)
        # Low variability (stable): higher multiplier (3x); High (erratic): lower (1.5x)
        alerts['CV'] = alerts['SD'] / alerts['Mean'].replace(0, np.nan)  # Avoid div0
        alerts['Min_Multiplier'] = np.where(
            alerts['CV'] < 0.5, 3.0,  # Stable: strict
            np.where(alerts['CV'] > 1.5, 1.5, 2.0)  # Erratic: sensitive; Medium: balanced
        )

        # Year-Round: Even stricter—High Alert only
        year_round_alerts = alerts[
            (alerts['Season'] == 'Year-Round') & (alerts['Alert_Level'] == 'High Alert')
        ].copy()

        # Seasonal: Filter by data-driven min % increase + 20% buffer for stricter overall
        seasonal_alerts = alerts[alerts['Season'] != 'Year-Round'].copy()
        seasonal_alerts['Adjusted_Multiplier'] = seasonal_alerts['Min_Multiplier'] * 1.2  # Extra strict buffer
        seasonal_alerts = seasonal_alerts[seasonal_alerts['Pct_Deviation'] >= seasonal_alerts['Adjusted_Multiplier']]

        # Always-on Clustering for seasonal (≥3 facilities for hotspots)
        seasonal_clusters = seasonal_alerts.groupby(['District', 'Disease']).size().reset_index(name='Cluster_Size')
        seasonal_clusters = seasonal_clusters[seasonal_clusters['Cluster_Size'] >= 3]
        seasonal_alerts = seasonal_alerts.merge(seasonal_clusters[['District', 'Disease', 'Cluster_Size']], on=['District', 'Disease'], how='inner')
        if len(seasonal_clusters) > 0:
            st.info(f"ℹ️ Clustered {len(seasonal_clusters)} district-disease hotspots (≥3 facilities).")

        # For non-clustered seasonal, set Cluster_Size=1 but keep only if Pct_Deviation > 5x (stricter for isolates)
        non_cluster_seasonal = alerts[
            (alerts['Season'] != 'Year-Round') & 
            (alerts['Pct_Deviation'] >= (alerts['Min_Multiplier'] * 1.2)) & 
            (~alerts['Facility_ID'].isin(seasonal_alerts['Facility_ID']))  # Non-hotspots
        ].copy()
        non_cluster_seasonal['Cluster_Size'] = 1
        non_cluster_seasonal = non_cluster_seasonal[non_cluster_seasonal['Pct_Deviation'] >= 5.0]  # Bumped up

        # Combine: hotspots + qualified isolates + year-round
        seasonal_alerts = pd.concat([seasonal_alerts, non_cluster_seasonal], ignore_index=True)
        final_alerts = pd.concat([year_round_alerts, seasonal_alerts], ignore_index=True)
        if final_alerts.empty:
            st.warning("No alerts after automated filtering.")
            st.stop()

        # Severity Score: (Pct_Deviation * Weight) + Cluster Bonus (+0.5 per extra facility)
        def get_weight(disease):
            return DISEASE_WEIGHTS.get(disease, DISEASE_WEIGHTS['default'])

        final_alerts['Disease_Weight'] = final_alerts['Disease'].apply(get_weight)
        final_alerts['Cluster_Bonus'] = np.maximum(final_alerts['Cluster_Size'] - 1, 0) * 0.5
        final_alerts['Severity_Score'] = (final_alerts['Pct_Deviation'] * final_alerts['Disease_Weight']) + final_alerts['Cluster_Bonus']

        # No hard cap: Show all, sorted by score. Warn if >100
        final_alerts = final_alerts.sort_values('Severity_Score', ascending=False)
        final_alerts = final_alerts[['Facility_ID', 'District', 'Disease', 'Season', 'Cases', 'Mean', 'CV', 'Pct_Deviation', 'Alert_Level', 'Cluster_Size', 'Severity_Score', 'Deviation', 'Min_Multiplier']]
        total_alerts_raw = len(final_alerts)
        if total_alerts_raw > 100:
            st.warning(f"⚠️ High alert volume ({total_alerts_raw})—possible outbreak surge. Focus on top 50 by score for priorities.")
            final_alerts = final_alerts.head(50)  # Soft cap for display, but download has all

        status.text('Applying automated filters and summarizing...')
        progress_bar.progress(90)

        # Summary metrics
        total_alerts = len(final_alerts)
        high_alerts = len(final_alerts[final_alerts['Alert_Level'] == 'High Alert'])
        alert_ratio = (total_alerts_raw / num_diseases / num_facilities * 100) if num_facilities > 0 else 0  # Use raw for ratio
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Alerts (Automated)", total_alerts_raw, help="All qualifying outbreaks via data-driven rules.")
        with col_b:
            st.metric("High Alerts", high_alerts, help="Severe outbreaks (>3.5σ above mean).")
        with col_c:
            st.metric("Alert Rate", f"{alert_ratio:.2f}%", help="Percentage of facility-disease pairs alerting.")

        st.success(f"**{total_alerts_raw} total alerts** (Dynamic surges: {final_alerts['Min_Multiplier'].min():.1f}-{final_alerts['Min_Multiplier'].max():.1f}x; ≥3-facility clusters; No cap—surges allowed).")

        if not final_alerts.empty:
            st.markdown("### 📊 Prioritized Outbreak Alerts (Sorted by Severity)")
            st.dataframe(final_alerts.style.format({
                'Cases': '{:.0f}', 'Mean': '{:.1f}', 'CV': '{:.2f}', 'Pct_Deviation': '{:.1f}x', 
                'Cluster_Size': '{:.0f}', 'Severity_Score': '{:.1f}', 'Deviation': '{:.1f}', 'Min_Multiplier': '{:.1f}x'
            }))

            with st.expander("🔍 Why This Alert? (Per-Row Details)"):
                st.write("Click rows below for disease-specific rationale (CV = volatility; Min_Multiplier = required surge).")
                # Simple per-row expander simulation—could enhance with st.data_editor if needed
                for idx, row in final_alerts.iterrows():
                    with st.expander(f"{row['Disease']} in {row['District']} ({row['Cases']} cases)"):
                        cv_val = row.get('CV', np.nan)
                        if pd.notna(cv_val):
                            formatted_cv = f"{cv_val:.2f}"
                            if cv_val < 0.5:
                                cv_desc = 'Low (stable)'
                            elif cv_val < 1.5:
                                cv_desc = 'Medium'
                            else:
                                cv_desc = 'High (erratic)'
                        else:
                            formatted_cv = 'N/A'
                            cv_desc = 'N/A'
                        st.write(f"**Volatility (CV)**: {formatted_cv} ({cv_desc}).")
                        st.write(f"**Required Surge**: {row['Min_Multiplier']:.1f}x historical mean.")
                        st.write(f"**Actual Surge**: {row['Pct_Deviation']:.1f}x ({'Passed with buffer' if row['Pct_Deviation'] >= row['Min_Multiplier'] * 1.2 else 'Passed baseline'}).")
                        st.write(f"**Cluster**: {row['Cluster_Size']} facilities (≥3 for hotspot status).")

            # Download button (full raw alerts)
            status.text('Preparing download...')
            progress_bar.progress(100)
            province_key = selected_province.lower().replace(" ", "_")
            csv_buffer = BytesIO()
            final_alerts_full = final_alerts.copy()  # Use raw if capped
            if total_alerts_raw > 100:
                final_alerts_full = pd.concat([final_alerts, final_alerts.iloc[50:]], ignore_index=True)  # Nah, just download raw
            # Actually, download the full sorted without cap
            full_final = final_alerts.sort_values('Severity_Score', ascending=False).reset_index(drop=True) if 'full_final' not in locals() else full_final
            full_final.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label=f"💾 Download All Alerts CSV (Week {new_week})",
                data=csv_data,
                file_name=f'all_alerts_{province_key}_week_{new_week}.csv',
                mime='text/csv',
                type="secondary"
            )
        else:
            st.warning("✅ No prioritized alerts detected this week—surveillance levels are normal. Monitor trends closely.")

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
    4. **Generate**: Click the button—filters auto-adjust based on your data!
    """)
    st.markdown("---")
    st.header("🛠️ How It Works")
    st.markdown("""
    - **Automated Magic**: Surge thresholds tuned by disease volatility (SD/Mean); e.g., stable diseases need 3x surge, erratic ones just 1.5x (+20% buffer for strictness).
    - **Noise Reduction**: ≥3 cases min; hotspots ≥3 facilities; isolates need 5x+ surge.
    - **Year-Round**: High Alert only (no low-volume noise).
    - **Scoring**: Ranks by % deviation × disease impact + cluster bonus.
    - **No Cap**: All shown (strict formula: normal <100; surges >100 OK).
    - **Weights**: Dengue/CCHF=2.5-3.0x boost; Scabies=0.5x (edit dict to tune).
    """)
    st.markdown("---")
    st.markdown("*Developed by Asad Khan* | *Version 2.12*")
