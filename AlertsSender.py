import streamlit as st
import pandas as pd
from twilio.rest import Client
import json

# Hardcoded Twilio Account SID, Template SID, and WhatsApp numbers
account_sid = 'ACe0cc33b5586f53c2fd861efdd7c6fef5'  # Updated with provided SID
template_sid = 'HX051daa8b0e94f978a0e3706a01dccc34'  # Updated with provided Template SID
from_whatsapp = 'whatsapp:+16093545812'       # Updated sender number
to_whatsapp = 'whatsapp:+923109511712'        # Updated recipient number (province headquarters)

# Streamlit app title
st.title("Disease Alert Sender App")

# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Enter your Twilio Auth Token below (it will be hidden).")
st.sidebar.write("2. Upload the alert CSV file (e.g., alerts_khyber_pakhtunkhwa_week_XX.csv).")
st.sidebar.write("3. Click 'Send Alerts' to send WhatsApp messages using the template.")
st.sidebar.write("Note: Messages will use the template 'DISEASE ALERT: {{1}} Kindly Investigate ASAP', with alert details in {{1}}.")
st.sidebar.write("Account SID, Template SID, and WhatsApp numbers are hardcoded—edit the code if needed.")
st.sidebar.write("Developer: Asad Khan")

# Input Twilio Auth Token manually
auth_token = st.text_input("Twilio Auth Token", type="password")

# Upload alert file
alert_file = st.file_uploader("Upload Alert CSV", type=['csv'])

if st.button("Send Alerts"):
    if not auth_token or alert_file is None:
        st.error("Please enter the Auth Token and upload the alert file.")
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

                # Content variables as JSON string
                content_variables = json.dumps({"1": details})

                # Send message using template
                message = client.messages.create(
                    from_=from_whatsapp,
                    content_sid=template_sid,
                    content_variables=content_variables,
                    to=to_whatsapp
                )

                sent_count += 1
                progress = sent_count / total_alerts
                progress_bar.progress(progress)
                status.text(f"Sent {sent_count}/{total_alerts} alerts. Last SID: {message.sid}")

            st.success(f"All {total_alerts} alerts sent successfully!")

        except Exception as e:
            st.error(f"Error sending messages: {str(e)}")
