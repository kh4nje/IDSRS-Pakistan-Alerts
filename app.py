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

def load_threshold_local():
    threshold_filename = 'AJK.csv'
    try:
        if os.path.exists(threshold_filename):
            df = pd.read_csv(threshold_filename)
            if df.empty:
                raise ValueError("Local file is empty.")
            return df
        else:
            raise FileNotFoundError(f"Local file '{threshold_filename}' not found.")
    except Exception as e:
        st.error(f"Failed to load local threshold file: {e}. Please ensure the file exists in the same folder as app.py.")
        st.stop()

# Streamlit app title
st.title("Disease Outbreak Detection App for AJK")
st.write("Upload weekly data and generate alerts using AJK-specific thresholds from local file.")

# Load threshold file from local for AJK
threshold_df = None
progress_bar = st.progress(0)
status = st.empty()
status.text('Initializing...')
progress_bar.progress(10)

threshold_df = load_threshold_local()
st.write("Threshold columns:", threshold_df.columns.tolist())
st.success(f"Threshold file loaded for AJK from local 'AJK.csv'.")
progress_bar.progress(30)

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
        # Parse periodname (prioritize KP-like "Week X YYYY..." pattern)
        if 'periodname' in new_df.columns:
            new_df['periodname'] = new_df['periodname'].astype(str).str.strip()
            patterns = [
                r'Week (\d+) (\d{4})-\d{2}-\d{2} - \d{4}-\d{2}-\d{2}',  # KP/Sindh format first
                r'(\d{4})W(\d{1,2})',  # W1 fallback
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
                        st.write(f"Matched pattern with {success} rows.")
                        break
            if best_extracted is not None:
                # Drop old Year/Week if exist to avoid conflict
                if 'Year' in new_df.columns:
                    new_df = new_df.drop(columns=['Year'])
                if 'Week' in new_df.columns:
                    new_df = new_df.drop(columns=['Week'])
                new_df = pd.concat([new_df, best_extracted], axis=1)
                new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
                new_df['Week'] = pd.to_numeric(new_df['Week'], errors='coerce')
                new_df = new_df.dropna(subset=['Year', 'Week'])
                if new_df.empty:
                    st.error("No valid weeks parsed after dropna.")
                    st.stop()
                new_week = new_df['Week'].iloc[0]
                st.write(f"Parsed Week: {new_week}")
            else:
                st.error("No pattern matched periodname. Check format (e.g., 'Week 40 2025-...').")
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
        # Check if DF is non-empty before melt
        if new_df.empty:
            st.error("DataFrame is empty after parsing—cannot melt.")
            st.stop()
        long_new = pd.melt(new_df, id_vars=['Facility_ID', 'Season'], value_vars=disease_cols, var_name='Disease', value_name='Cases')
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
        if 'Season' not in threshold_df.columns:
            st.error("Threshold file does not have 'Season' column. Please check the file structure.")
            st.stop()
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
                        (~alerts['Disease'].str.contains('Other', na=False))].copy()
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

        st.write(f"Total alerts for AJK: {len(final_alerts)} ({len(priority_alerts)} priority + {len(filtered_non_priority)} filtered)")

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
                label=f"Download Alerts for AJK Week {new_week}",
                data=output,
                file_name=f'alerts_ajk_week_{new_week}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.warning("No alerts generated.")

    else:
        st.warning("Upload weekly data to generate alerts.")

    progress_bar.empty()
    status.empty()

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Ensure AJK.csv is in the same folder as app.py.")
st.sidebar.write("2. Upload weekly data (CSV/Excel).")
st.sidebar.write("3. Adjust filters and click 'Generate Alerts'.")
st.sidebar.write("4. View and download results.")
st.sidebar.write("Note: Handles Other exclusion, year-round remapping, and priority inclusion.")
