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
        "Khyber Pakhtunkhwa": "seasonal_thresholds_kp.xlsx"
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

# Streamlit app title
st.title("IDSRS Pakistan, Disease Outbreak Detection App for Provinces")

# Province selection
provinces = ["AJK", "Balochistan", "Gilgit Baltistan", "Islamabad", "Sindh", "Khyber Pakhtunkhwa"]
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
st.success(f"Threshold file loaded for {selected_province} from local.")
progress_bar.progress(30)

# Upload new week file (weekly data)
new_file = st.file_uploader("Upload new week data (CSV or Excel)", type=['xlsx', 'csv'])

# Priority diseases (updated to match exact column names from provided list)
priority_diseases = [
    "Crimean Congo Hemorrhagic Fever (New Cases)",
    "Anthrax (New Cases)",
    "Botulism (New Cases)",
    "Diphtheria (Probable) (New Cases)",
    "Neonatal Tetanus (New Cases)",
    "Acute Flaccid Paralysis (New Cases)"
]
selected_priority_diseases = st.multiselect(
    "Select priority diseases to always include:",
    options=priority_diseases,
    default=priority_diseases
)

# Run button
if st.button("Generate Alerts"):
    if threshold_df is not None and new_file is not None:
        status.text('Loading new week data...')
        progress_bar.progress(40)
        new_df = load_file(new_file)
        st.write("New week data loaded. Shape:", new_df.shape)

        # Remove unnecessary columns
        columns_to_remove = ['periodid', 'periodcode', 'perioddescription', 'organisationunitid', 'organisationunitcode', 'organisationunitdescription']
        new_df = new_df.drop(columns=[col for col in columns_to_remove if col in new_df.columns])

        # Dynamic org levels and Facility_ID (handles up to level6 or more)
        org_level_cols = [col for col in new_df.columns if col.startswith('orgunitlevel') and col != 'organisationunitname']
        org_level_cols.sort(key=lambda x: int(x.replace('orgunitlevel', '')))  # Sort by level number
        org_cols = org_level_cols + ['organisationunitname']
        for col in org_cols:
            if col in new_df.columns:
                new_df[col] = new_df[col].fillna('Unknown').astype(str)
        if org_cols:
            # Concatenate dynamically
            new_df['Facility_ID'] = new_df[org_cols[0]]
            for col in org_cols[1:]:
                new_df['Facility_ID'] += '_' + new_df[col]
            st.write(f"Unique Facility_IDs: {new_df['Facility_ID'].nunique()}")
        else:
            st.error("No organization unit columns found for Facility_ID.")
            st.stop()

        status.text('Parsing week and season...')
        progress_bar.progress(60)
        # Parse periodname with expanded patterns for robustness
        if 'periodname' in new_df.columns:
            new_df['periodname'] = new_df['periodname'].astype(str).str.strip()
            patterns = [
                r'Week (\d+) (\d{4})-\d{2}-\d{2} - \d{4}-\d{2}-\d{2}',  # KP/Sindh primary
                r'(\d{4})W(\d{1,2})',  # YYYY WXX
                r'(\d{4})\s*W(?:eek)?\s*(\d{1,2})',  # YYYY Week XX
                r'Week\s+(\d{1,2})\s*,?\s*(\d{4})',  # Week XX, YYYY
                r'(\d{4})-(\d{2})-\d{2}'   # YYYY-MM-DD (use as year, derive week if needed)
            ]
            best_extracted = None
            success_rates = []
            for pat in patterns:
                extracted = new_df['periodname'].str.extract(pat)
                if extracted.shape[1] == 2:
                    # Assign columns based on pattern
                    if 'Week (\d+)' in pat or 'Week\s+(\d{1,2})' in pat:
                        extracted.columns = ['Week', 'Year']
                    else:
                        extracted.columns = ['Year', 'Week']
                    success = extracted.notna().all(axis=1).mean() * 100
                    success_rates.append(success)
                    if success >= 90:
                        best_extracted = extracted
                        break
            if best_extracted is None and success_rates:
                # Fallback to best partial
                best_idx = success_rates.index(max(success_rates))
                best_extracted = new_df['periodname'].str.extract(patterns[best_idx])
                if 'Week' in patterns[best_idx]:
                    best_extracted.columns = ['Week', 'Year']
                else:
                    best_extracted.columns = ['Year', 'Week']
                st.warning(f"Partial periodname match ({max(success_rates):.1f}%). Check data format.")

            if best_extracted is not None:
                new_df = pd.concat([new_df, best_extracted], axis=1)
                new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
                new_df['Week'] = pd.to_numeric(new_df['Week'], errors='coerce')
                new_df = new_df.dropna(subset=['Year', 'Week'])
                if new_df.empty:
                    st.error("No valid weeks parsed after dropna.")
                    st.stop()
                new_week = new_df['Week'].iloc[0]  # Assume single week
                st.write(f"Parsed Week: {new_week}")
            else:
                st.error("No pattern matched periodname. Check format (e.g., 'Week 40 2025-...').")
                st.stop()
        else:
            st.error("No 'periodname' column.")
            st.stop()

        # Season assignment
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
        st.write(f"Season: {new_df['Season'].iloc[0]}")

        status.text('Melting and merging data...')
        progress_bar.progress(70)
        # Disease columns and melt
        disease_cols = [col for col in new_df.columns if '(New Cases)' in col or '(New cases)' in col]
        disease_cols = [col for col in disease_cols if not col.startswith('Other-')]  # Exclude Other-*
        if len(disease_cols) == 0:
            st.error("No disease columns found.")
            st.stop()
        new_df[disease_cols] = new_df[disease_cols].fillna(0).astype(float)  # Fill NaN with 0, use float for consistency
        if new_df.empty:
            st.error("DataFrame is empty after parsing—cannot melt.")
            st.stop()
        long_new = pd.melt(new_df, id_vars=['Facility_ID', 'Season'], value_vars=disease_cols, var_name='Disease', value_name='Cases')
        long_new['Cases'] = long_new['Cases'].astype(float)
        st.write("Melted data shape:", long_new.shape)

        # Year-round override (updated to exact names from historical script and columns, no trailing '/')
        year_round_diseases = [
            'Acute Flaccid Paralysis (New Cases)', 'Botulism (New Cases)', 'Gonorrhea (New Cases)', 
            'HIV/AIDS (New Cases)', 'Leprosy (New Cases)', 'Nosocomial Infections (New Cases)', 
            'Syphilis (New Cases)', 'Visceral Leishmaniasis (New Cases)', 'Neonatal Tetanus (New Cases)'
        ]
        long_new.loc[long_new['Disease'].isin(year_round_diseases), 'Season'] = 'Year-Round'

        # Filter thresholds for speed (seasonal with seasonal, year-round with year-round via override)
        current_season = new_df['Season'].iloc[0]
        if 'Season' not in threshold_df.columns:
            st.error("Threshold file does not have 'Season' column. Please check the file structure.")
            st.stop()
        filtered_thresholds = threshold_df[(threshold_df['Season'] == current_season) | (threshold_df['Season'] == 'Year-Round')]
        alerts = long_new.merge(filtered_thresholds[['Facility_ID', 'Disease', 'Season', 'Threshold_95', 'Threshold_99', 'Mean', 'SD']], on=['Facility_ID', 'Disease', 'Season'], how='left')

        # Validate merge (warn if many unmatched)
        unmatched_pct = alerts['Threshold_95'].isna().mean() * 100
        if unmatched_pct > 20:
            st.warning(f"{unmatched_pct:.1f}% of rows have no matching threshold—check Facility_ID, Disease, or Season matches. Sample unmatched Diseases: {long_new['Disease'].unique()[:5]}")

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

        # Filter alerts
        alerts = alerts[(alerts['Alert_Level'] != 'Normal') & 
                        alerts['Threshold_95'].notna() & 
                        (~alerts['Disease'].str.contains('Other', na=False))].copy()
        alerts = alerts[['Facility_ID', 'Disease', 'Season', 'Cases', 'Mean', 'SD', 'Threshold_95', 'Threshold_99', 'Alert_Level', 'Deviation']]

        status.text('Filtering alerts...')
        progress_bar.progress(90)
        # Priority filtering
        priority_alerts = alerts[alerts['Disease'].isin(selected_priority_diseases)]
        non_priority_alerts = alerts[~alerts['Disease'].isin(selected_priority_diseases)]
        st.write(f"Priority alerts count: {len(priority_alerts)}, Non-priority alerts count: {len(non_priority_alerts)}")
        
        # Conditionally render sliders only if non-priority alerts exist
        if len(non_priority_alerts) > 0:
            col1, col2 = st.columns(2)
            with col1:
                top_n = st.slider("Top N Non-Priority Alerts", min_value=0, max_value=len(non_priority_alerts), value=min(100, len(non_priority_alerts)))
            with col2:
                max_dev = non_priority_alerts['Deviation'].max()
                min_dev = st.slider("Min Deviation for Non-Priority", min_value=0.0, max_value=float(max_dev), value=0.0)
        else:
            top_n = 0
            min_dev = 0.0
        
        filtered_non_priority = non_priority_alerts[non_priority_alerts['Deviation'] >= min_dev].nlargest(top_n, 'Deviation')
        final_alerts = pd.concat([priority_alerts, filtered_non_priority], ignore_index=True)
        final_alerts = final_alerts.sort_values('Deviation', ascending=False)

        st.write(f"Total alerts for {selected_province}: {len(final_alerts)} ({len(priority_alerts)} priority + {len(filtered_non_priority)} filtered)")

        if not final_alerts.empty:
            st.dataframe(final_alerts)

            # Download as CSV
            status.text('Preparing download...')
            progress_bar.progress(100)
            province_key = selected_province.lower().replace(" ", "_")
            csv = final_alerts.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download Alerts for {selected_province} Week {new_week} (CSV)",
                data=csv,
                file_name=f'alerts_{province_key}_week_{new_week}.csv',
                mime='text/csv'
            )
        else:
            st.warning("No alerts generated.")

    else:
        st.warning("Upload weekly data to generate alerts.")

    progress_bar.empty()
    status.empty()

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Select province.")
st.sidebar.write("2. Ensure the corresponding threshold file is in the same folder as app.py (e.g., AJK.csv for AJK, seasonal_thresholds_kp.csv for Khyber Pakhtunkhwa).")
st.sidebar.write("3. Upload weekly data (CSV/Excel).")
st.sidebar.write("4. Adjust filters and click 'Generate Alerts'.")
st.sidebar.write("5. View and download results (CSV).")
st.sidebar.write("Developer: Asad khan")





