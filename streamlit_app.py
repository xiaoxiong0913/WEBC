import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import warnings

# 处理版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the model and scaler
model_path = "log_reg_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Define risk cutoff
risk_cutoff = 0.1723

# Define the feature names in the specified order
feature_names = [
    'Glu(mmol/L)',
    'Hb(g/L)',
    'Scr(μmol/L)',
    'DBP(mmHg)',
    'Neu（10^9/L）',
    'reperfusion therapy (yes1,no0)',
    'ACEI/ARB(yes1,no0)',
    'atrial fibrillation(yes1,no0）',
    'gender（male0;female1）'
]

# Create the title for the web app
st.title(
    'Machine learning-based prediction of one-year mortality in patients with non-ST-segment elevation myocardial infarction combined with diabetes mellitus.')

# Introduction section
st.markdown("""
## Introduction
This web-based calculator was developed based on the Logistic Regression model with an AUC of 0.86 and a Brier score of 0.086. Users can obtain the 1-year risk of death for a given case by simply selecting the parameters and clicking on the 'Predict' button.
""")

# Create the input form
with st.form(key="unique_prediction_form"):
    glu = st.slider('GLU (mmol/L)', min_value=1.0, max_value=35.0, value=18.0, step=0.1, key='unique_Glu')
    hb = st.slider('Hb (g/L)', min_value=40.0, max_value=220.0, value=130.0, step=1.0, key='unique_Hb')
    scr = st.slider('Scr (μmol/L)', min_value=30.0, max_value=1200.0, value=100.0, step=1.0, key='unique_Scr')
    dbp = st.slider('DBP (mmHg)', min_value=40.0, max_value=180.0, value=80.0, step=1.0, key='unique_DBP')
    neu = st.slider('Neu (10^9/L)', min_value=1.0, max_value=25.0, value=5.0, step=0.1, key='unique_Neu')
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No',
                                       key='unique_Reperfusion_Therapy')
    acei_arb = st.selectbox('ACEI/ARB', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No',
                            key='unique_ACEI_ARB')
    atrial_fibrillation = st.selectbox('Atrial Fibrillation', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No',
                                       key='unique_Atrial_Fibrillation')
    gender = st.radio('Gender', ['male', 'female'], index=0, key='unique_Gender')

    predict_button = st.form_submit_button("Predict")

# Process form submission
if predict_button:
    data = {
        "Glu(mmol/L)": glu,
        "Hb(g/L)": hb,
        "Scr(μmol/L)": scr,
        "DBP(mmHg)": dbp,
        "Neu（10^9/L）": neu,
        "reperfusion therapy (yes1,no0)": reperfusion_therapy,
        "ACEI/ARB(yes1,no0)": acei_arb,
        "atrial fibrillation(yes1,no0）": atrial_fibrillation,
        "gender（male0;female1）": 0 if gender == 'male' else 1
    }

    try:
        # Convert input data to DataFrame using the exact feature names
        data_df = pd.DataFrame([data], columns=feature_names)

        # Scale the data using the loaded scaler
        data_scaled = scaler.transform(data_df)

        # Make a prediction
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # Getting the probability of class 1
        st.subheader("Prediction Result:")
        st.write(f'Prediction: {prediction * 100:.2f}%')  # Convert probability to percentage

        # Risk stratification and personalized recommendations
        if prediction >= risk_cutoff:
            st.markdown("<span style='color:red'>High risk: This patient is classified as a high-risk patient.</span>",
                        unsafe_allow_html=True)
            st.subheader("Personalized Recommendations:")
            # Gender-specific normal ranges
            gender_value = 'male' if data['gender（male0;female1）'] == 0 else 'female'

            # GLU
            if glu < 3.9:
                st.markdown(
                    f"<span style='color:red'>GLU (mmol/L): Your value is {glu}. It is lower than the normal range (3.9 - 6.1). Consider increasing it towards 3.9.</span>",
                    unsafe_allow_html=True)
            elif glu > 6.1:
                st.markdown(
                    f"<span style='color:red'>GLU (mmol/L): Your value is {glu}. It is higher than the normal range (3.9 - 6.1). Consider decreasing it towards 6.1.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"GLU (mmol/L): Your value is within the normal range (3.9 - 6.1).")

            # Hb
            if gender_value == 'male':
                hb_low = 120
                hb_high = 160
            else:
                hb_low = 110
                hb_high = 150

            if hb < hb_low:
                st.markdown(
                    f"<span style='color:red'>Hb (g/L): Your value is {hb}. It is lower than the normal range ({hb_low} - {hb_high}). Consider increasing it towards {hb_low}.</span>",
                    unsafe_allow_html=True)
            elif hb > hb_high:
                st.markdown(
                    f"<span style='color:red'>Hb (g/L): Your value is {hb}. It is higher than the normal range ({hb_low} - {hb_high}). Consider decreasing it towards {hb_high}.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"Hb (g/L): Your value is within the normal range ({hb_low} - {hb_high}).")

            # Scr
            if gender_value == 'male':
                scr_low = 53
                scr_high = 106
            else:
                scr_low = 44
                scr_high = 97

            if scr < scr_low:
                st.markdown(
                    f"<span style='color:red'>Scr (μmol/L): Your value is {scr}. It is lower than the normal range ({scr_low} - {scr_high}). Consider increasing it towards {scr_low}.</span>",
                    unsafe_allow_html=True)
            elif scr > scr_high:
                st.markdown(
                    f"<span style='color:red'>Scr (μmol/L): Your value is {scr}. It is higher than the normal range ({scr_low} - {scr_high}). Consider decreasing it towards {scr_high}.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"Scr (μmol/L): Your value is within the normal range ({scr_low} - {scr_high}).")

            # DBP
            if dbp < 60:
                st.markdown(
                    f"<span style='color:red'>DBP (mmHg): Your value is {dbp}. It is lower than the normal range (60 - 90). Consider increasing it towards 60.</span>",
                    unsafe_allow_html=True)
            elif dbp > 90:
                st.markdown(
                    f"<span style='color:red'>DBP (mmHg): Your value is {dbp}. It is higher than the normal range (60 - 90). Consider decreasing it towards 90.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"DBP (mmHg): Your value is within the normal range (60 - 90).")

            # Neu
            if neu < 1.8:
                st.markdown(
                    f"<span style='color:red'>Neu (10^9/L): Your value is {neu}. It is lower than the normal range (1.8 - 6.3). Consider increasing it towards 1.8.</span>",
                    unsafe_allow_html=True)
            elif neu > 6.3:
                st.markdown(
                    f"<span style='color:red'>Neu (10^9/L): Your value is {neu}. It is higher than the normal range (1.8 - 6.3). Consider decreasing it towards 6.3.</span>",
                    unsafe_allow_html=True)
            else:
                st.write(f"Neu (10^9/L): Your value is within the normal range (1.8 - 6.3).")

            # Reperfusion therapy
            if reperfusion_therapy == 0:
                st.write("Consider undergoing reperfusion therapy.")

            # ACEI/ARB
            if acei_arb == 0:
                st.write("Consider initiating ACEI/ARB therapy.")

            # Atrial fibrillation
            if atrial_fibrillation == 1:
                st.write("Ensure atrial fibrillation is being properly managed.")
        else:
            st.markdown("<span style='color:green'>Low risk: This patient is classified as a low-risk patient.</span>",
                        unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Error: {str(e)}')
