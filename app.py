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

# Streamlit app title
st.title("Disease Outbreak Detection App for Provinces")
st.write("Select province, upload weekly data, and generate alerts using province-specific thresholds.")

# Province selection
provinces = ["AJK", "Balochistan", "Gilgit Baltistan", "Islamabad", "Sindh"]
selected_province = st.selectbox("Select Province:", provinces)

# File naming convention (standardize to lowercase with _ for spaces)
province_key = selected_province.lower().replace(" ", "_")
threshold_filename = f'seasonal_thresholds_{province_key}.csv'  # Always save/load as CSV

# Load or initialize threshold file for selected province
threshold_df = None
progress_bar = st.progress(0)
status = st.empty()
status.text('Initializing...')
progress_bar.progress(10)

if os.path.exists(threshold_filename):
    try:
        threshold_df = pd.read_csv(threshold_filename)
        if threshold_df.empty:
            raise pd.errors.EmptyDataError("Local file is empty.")
        st.success(f"Threshold file loaded for {selected_province} from local '{threshold_filename}'.")
        progress_bar.progress(30)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.warning(f"Local file '{threshold_filename}' is invalid/empty. Please re-upload.")
        os.remove(threshold_filename)  # Clean up bad file
        threshold_df = None
else:
    # Check for alternative filenames (e.g., capital letters or XLSX)
    alt_filenames = [
        f'seasonal_thresholds_{selected_province}.csv',  # Capital, no _
        f'seasonal_thresholds_{selected_province}.xlsx',
        f'seasonal_thresholds_{province_key}.xlsx',  # Lower + XLSX
        f'seasonal_thresholds_Gilgit_Baltistan.csv' if selected_province == "Gilgit Baltistan" else None,
        f'seasonal_thresholds_Islamabad.csv' if selected_province == "Islamabad" else None,
        f'seasonal_thresholds_Balochistan.csv' if selected_province == "Balochistan" else None
    ]
    alt_filenames = [f for f in alt_filenames if f is not None]
    loaded_from_alt = None
    for alt_fn in alt_filenames:
        if os.path.exists(alt_fn):
            try:
                if alt_fn.endswith('.xlsx'):
                    temp_df = pd.read_excel(alt_fn)
                else:
                    temp_df = pd.read_csv(alt_fn)
                if not temp_df.empty:
                    threshold_df = temp_df
                    loaded_from_alt = alt_fn
                    # Convert and save as standard CSV
                    threshold_df.to_csv(threshold_filename, index=False)
                    st.success(f"Threshold loaded from '{alt_fn}' and standardized to '{threshold_filename}'.")
                    progress_bar.progress(30)
                    break
            except Exception:
                continue  # Try next alt
    if threshold_df is None:
        initial_threshold_file = st.file_uploader(f"Upload initial threshold file for {selected_province} (CSV or XLSX)", type=['csv', 'xlsx'])
        if initial_threshold_file is not None:
            try:
                threshold_df = load_file(initial_threshold_file)
                if threshold_df.empty:
                    raise ValueError("Uploaded file is empty.")
                # Save to local CSV
                threshold_df.to_csv(threshold_filename, index=False)
                st.success(f"Initial threshold file for {selected_province} saved as '{threshold_filename}'.")
                progress_bar.progress(30)
            except Exception as e:
                st.error(f"Error loading threshold file: {e}")
                st.stop()
        else:
            st.warning(f"No threshold file for {selected_province}. Please upload to proceed.")
            st.stop()

# Upload new week file (weekly data)
new_file = st.file_uploader("Upload new week data (CSV or Excel)", type=['xlsx', 'csv'])

# Priority diseases
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

        # Org levels and Facility_ID
        org_cols = ['orgunitlevel1', 'orgunitlevel2', 'orgunitlevel3', 'orgunitlevel4', 'orgunitlevel5', 'organisationunitname']
        for col in org_cols:
            if col in new_df.columns:
                new_df[col] = new_df[col].fillna('Unknown').astype(str)
        if all(col in new_df.columns for col in org_cols):
            new_df['Facility_ID'] = (new_df['orgunitlevel1'] + '_' + new_df['orgunitlevel2'] + '_' + 
                                     new_df['orgunitlevel3'] + '_' + new_df['orgunitlevel4'] + '_' + 
                                     new_df['orgunitlevel5'] + '_' + new_df['organisationunitname'])
            st.write(f"Unique Facility_IDs: {new_df['Facility_ID'].nunique()}")
        else:
            st.error("Missing required org columns.")
            st.stop()

        status.text('Parsing week and season...')
        progress_bar.progress(50)
        # Parse periodname
        if 'periodname' in new_df.columns:
            new_df['periodname'] = new_df['periodname'].astype(str).str.strip()
            patterns = [
                r'(\d{4})W(\d{1,2})',
                r'Week (\d+) (\d{4})-\d{2}-\d{2} - \d{4}-\d{2}-\d{2}',
            ]
            best_extracted = None
            for pat in patterns:
                extracted = new_df['periodname'].str.extract(pat)
                if extracted.shape[1] == 2:
                    if 'Week' in pat:
                        extracted.columns = ['Week', 'Year']
                    else:
                        extracted.columns = ['Year', 'Week']
                    if extracted.notna().all(axis=1).sum() > 0:
                        best_extracted = extracted
                        break
            if best_extracted is not None:
                new_df = pd.concat([new_df, best_extracted], axis=1)
                new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
                new_df['Week'] = pd.to_numeric(new_df['Week'], errors='coerce')
                new_df = new_df.dropna(subset=['Year', 'Week'])
                new_week = new_df['Week'].iloc[0]
                st.write(f"Parsed Week: {new_week}")
            else:
                st.error("No pattern matched periodname.")
                st.stop()
        else:
            st.error("No 'periodname' column.")
            st.stop()

        # Season
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
        if len(disease_cols) == 0:
            st.error("No disease columns found.")
            st.stop()
        new_df[disease_cols] = new_df[disease_cols].fillna(0).astype(int)
        long_new = pd.melt(new_df, id_vars=['Facility_ID', 'Season'], value_vars=disease_cols, var_name='Disease', value_name='Cases', low_memory=False)
        long_new['Cases'] = long_new['Cases'].astype(int)
        st.write("Melted data shape:", long_new.shape)

        # Year-round override
        year_round_diseases = [
            'Acute Flaccid Paralysis (New Cases)', 'Botulism (New Cases)', 'Gonorrhea (New Cases)', 
            'HIV/AIDS (New Cases)', 'Leprosy (New Cases)', 'Nosocomial Infections (New Cases)', 
            'Syphilis (New Cases)', 'Visceral Leishmaniasis (New Cases)', 'Neonatal Tetanus (New Cases)'
        ]
        long_new.loc[long_new['Disease'].isin(year_round_diseases), 'Season'] = 'Year-Round'

        # Filter thresholds for speed
        current_season = new_df['Season'].iloc[0]
        filtered_thresholds = threshold_df[threshold_df['Season'].isin([current_season, 'Year-Round'])]
        alerts = long_new.merge(filtered_thresholds[['Facility_ID', 'Disease', 'Season', 'Threshold_95', 'Threshold_99', 'Mean', 'SD']], how='left')

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
                        (alerts['Deviation'] >= 1) &
                        (~alerts['Disease'].str.contains('Other'))].copy()
        alerts = alerts[['Facility_ID', 'Disease', 'Season', 'Cases', 'Mean', 'SD', 'Threshold_95', 'Threshold_99', 'Alert_Level', 'Deviation']]

        status.text('Filtering alerts...')
        progress_bar.progress(90)
        # Priority filtering
        priority_alerts = alerts[alerts['Disease'].isin(selected_priority_diseases)]
        non_priority_alerts = alerts[~alerts['Disease'].isin(selected_priority_diseases)]
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Top N Non-Priority Alerts", min_value=0, max_value=len(non_priority_alerts), value=min(50, len(non_priority_alerts)))
        with col2:
            min_dev = st.slider("Min Deviation for Non-Priority", min_value=1.0, max_value=non_priority_alerts['Deviation'].max() if len(non_priority_alerts) > 0 else 10.0, value=1.0)
        filtered_non_priority = non_priority_alerts[(non_priority_alerts['Deviation'] >= min_dev)].head(top_n)
        final_alerts = pd.concat([priority_alerts, filtered_non_priority], ignore_index=True)
        final_alerts = final_alerts.sort_values('Deviation', ascending=False)

        st.write(f"Total alerts for {selected_province}: {len(final_alerts)} ({len(priority_alerts)} priority + {len(filtered_non_priority)} filtered)")

        if not final_alerts.empty:
            st.dataframe(final_alerts)

            # Download
            status.text('Preparing download...')
            progress_bar.progress(100)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_alerts.to_excel(writer, index=False, sheet_name='Alerts')
            output.seek(0)
            st.download_button(
                label=f"Download Alerts for {selected_province} Week {new_week}",
                data=output,
                file_name=f'alerts_{province_key}_week_{new_week}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.warning("No alerts generated.")

    progress_bar.empty()
    status.empty()

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Select province.")
st.sidebar.write("2. Upload initial threshold CSV/XLSX for the province (saved as CSV locally).")
st.sidebar.write("3. Upload weekly data (CSV/Excel).")
st.sidebar.write("4. Adjust filters and click 'Generate Alerts'.")
st.sidebar.write("5. View and download results.")
st.sidebar.write("Note: Handles Other exclusion, year-round remapping, and priority inclusion.")
