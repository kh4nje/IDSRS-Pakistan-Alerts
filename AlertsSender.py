import streamlit as st
import pandas as pd
from twilio.rest import Client

# Streamlit app title
st.title("Disease Alert Sender App")

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Enter your Twilio credentials (Account SID and Auth Token).")
st.sidebar.write("2. Enter the sender WhatsApp number (e.g., whatsapp:+14155238886).")
st.sidebar.write("3. Enter the recipient WhatsApp number (e.g., whatsapp:+923001234567).")
st.sidebar.write("4. Upload the alert CSV file (e.g., alerts_khyber_pakhtunkhwa_week_XX.csv).")
st.sidebar.write("5. Click 'Send Alerts' to send WhatsApp messages using the template.")
st.sidebar.write("Note: Messages will be sent in the format 'DISEASE ALERT: {alert_details} Kindly Investigate ASAP'")
st.sidebar.write("Developer: Asad Khan")

# Input Twilio credentials
account_sid = st.text_input("Twilio Account SID", type="password")
auth_token = st.text_input("Twilio Auth Token", type="password")
from_whatsapp = st.text_input("Sender WhatsApp Number (Twilio sandbox, e.g., whatsapp:+14155238886)")
to_whatsapp = st.text_input("Recipient WhatsApp Number (e.g., whatsapp:+923001234567)")

# Upload alert file
alert_file = st.file_uploader("Upload Alert CSV", type=['csv'])

if st.button("Send Alerts"):
    if not all([account_sid, auth_token, from_whatsapp, to_whatsapp]) or alert_file is None:
        st.error("Please provide all Twilio credentials, WhatsApp numbers, and upload the alert file.")
    else:
        try:
            # Load the CSV
            df = pd.read_csv(alert_file)
            if df.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            # Initialize Twilio client
            client = Client(account_sid, auth_token)

            # Progress bar
            progress_bar = st.progress(0)
            status = st.empty()
            total_alerts = len(df)
            sent_count = 0

            for index, row in df.iterrows():
                # Format the alert details for {{1}}
                details = (
                    f"Disease: {row['Disease']}, "
                    f"Facility: {row['Facility_ID']}, "
                    f"Season: {row['Season']}, "
                    f"Cases: {row['Cases']}, "
                    f"Mean: {row['Mean']}, "
                    f"SD: {row['SD']}, "
                    f"Threshold_95: {row['Threshold_95']}, "
                    f"Threshold_99: {row['Threshold_99']}, "
                    f"Alert_Level: {row['Alert_Level']}, "
                    f"Deviation: {row['Deviation']}"
                )

                # Template body
                body = f"DISEASE ALERT:\n{details}\n\nKindly Investigate ASAP"

                # Send message
                message = client.messages.create(
                    from_=from_whatsapp,
                    body=body,
                    to=to_whatsapp
                )

                sent_count += 1
                progress = sent_count / total_alerts
                progress_bar.progress(progress)
                status.text(f"Sent {sent_count}/{total_alerts} alerts. Last SID: {message.sid}")

            st.success(f"All {total_alerts} alerts sent successfully!")

        except Exception as e:
            st.error(f"Error sending messages: {str(e)}")
