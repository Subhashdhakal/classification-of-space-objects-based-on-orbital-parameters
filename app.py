import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model, label encoders, and scaler from pickle files
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to convert year to Unix timestamp
def year_to_timestamp(year):
    dt = pd.to_datetime(f'{year}-01-01')
    return int(dt.timestamp())

# Define the Streamlit app
def main():
    # App Title
    st.title("🚀 **Space Object Classification Tool**")
    
    # Subheader with information
    st.markdown("""
    Welcome to the **Space Object Classification Tool**. This app allows you to classify space objects 
    based on their orbital parameters. Fill in the details on the left, and we'll predict the object type 
    (e.g., **Debris**, **Payload**, **Rocket Body**).
    """)
    
    # Divider for sections
    st.divider()

    # Sidebar for Inputs
    st.sidebar.header("🔧 Input Parameters")

    # Dropdown list for SITE in the sidebar
    site_names = [
        'FRGUI', 'PKMTR', 'SRI', 'TSC', 'SEAL', 'AFETR', 'TTMTR', 'XSC', 'AFWTR',
        'TNSTA', 'KODAK', 'ERAS', 'KSCUT', 'WRAS', 'WLPIS', 'JSC', 'SVOB', 'OREN',
        'KYMTR', 'KWAJL', 'YUN', 'NSC', 'VOSTO', 'WSC', 'RLLC', 'YSLA', 'SMTS'
    ]
    site = st.sidebar.selectbox("Launch Site", site_names)

    # Input fields for other data
    inclination = st.sidebar.number_input("Inclination (degrees)", format="%.2f")
    launch_year = st.sidebar.number_input("Launch Year", min_value=1900, max_value=2100, value=2020)
    semimajor_axis = st.sidebar.number_input("Semimajor Axis (km)", format="%.2f")
    mean_motion = st.sidebar.number_input("Mean Motion (revolutions/day)", format="%.2f")
    period = st.sidebar.number_input("Orbital Period (minutes)", format="%.2f")
    apoapsis = st.sidebar.number_input("Apoapsis (km)", format="%.2f")
    periapsis = st.sidebar.number_input("Periapsis (km)", format="%.2f")
    eccentricity = st.sidebar.number_input("Eccentricity", format="%.4f")
    rcs_size = st.sidebar.selectbox("RCS Size", ["SMALL", "MEDIUM", "LARGE"])

    # Convert the year to Unix timestamp
    launch_date_timestamp = year_to_timestamp(launch_year)

    # Button for classification
    if st.sidebar.button("🚀 Classify Object"):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'SITE': [site],
            'INCLINATION': [inclination],
            'LAUNCH_DATE_TIMESTAMP': [launch_date_timestamp],
            'SEMIMAJOR_AXIS': [semimajor_axis],
            'MEAN_MOTION': [mean_motion],
            'PERIOD': [period],
            'APOAPSIS': [apoapsis],
            'PERIAPSIS': [periapsis],
            'ECCENTRICITY': [eccentricity],
            'RCS_SIZE': [rcs_size]
        })

        # Preprocess the input data
        for column in label_encoders.keys():
            if column in input_data.columns:
                le = label_encoders[column]
                input_data[column] = le.transform(input_data[column])

        # Scale the numerical features
        numerical_columns = [
            'INCLINATION', 'LAUNCH_DATE_TIMESTAMP', 'SEMIMAJOR_AXIS', 'MEAN_MOTION',
            'PERIOD', 'APOAPSIS', 'PERIAPSIS', 'ECCENTRICITY'
        ]
        input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

        # Predict with the model
        selected_features = [
            'SITE', 'INCLINATION', 'LAUNCH_DATE_TIMESTAMP', 'SEMIMAJOR_AXIS',
            'MEAN_MOTION', 'PERIOD', 'APOAPSIS', 'PERIAPSIS', 'ECCENTRICITY', 'RCS_SIZE'
        ]
        input_data_selected = input_data[selected_features]
        prediction = model.predict(input_data_selected)

        # Define a mapping from numeric labels to class names
        label_mapping = {0: 'DEBRIS', 1: 'PAYLOAD', 2: 'ROCKET BODY'}
        predicted_class = label_mapping[prediction[0]]

        # Display the prediction with a professional format
        st.markdown("### **Prediction Result**")
        st.success(f"**Predicted Object Type:** {predicted_class}")
        
        # Divider for sections
        st.divider()

        # Create a summary DataFrame
        summary_data = {
            'Feature': [
                'Launch Site', 'Inclination', 'Launch Year', 'Semimajor Axis',
                'Mean Motion', 'Orbital Period', 'Apoapsis', 'Periapsis',
                'Eccentricity', 'RCS Size'
            ],
            'Value': [
                site, inclination, launch_year, semimajor_axis,
                mean_motion, period, apoapsis, periapsis,
                eccentricity, rcs_size
            ]
        }

        summary_df = pd.DataFrame(summary_data)

        # Display the summary table in an elegant format
        st.markdown("### **Input Summary**")
        st.table(summary_df)

if __name__ == "__main__":
    main()
