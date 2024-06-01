import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model_path = "log_reg_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

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
st.title('Machine learning-based prediction of one-year mortality in patients with non-ST-segment elevation myocardial infarction combined with diabetes mellitus.')

# Introduction section
st.markdown("""
## Introduction
This web-based calculator was developed based on the LogisticRegression model with an AUC of 0.86 and a Brier score of 0.086. Users can obtain the 1-year risk of death for a given case by simply selecting the parameters and clicking on the 'Predict' button.
""")

# Create the input form
with st.form("prediction_form"):
    glu = st.slider('GLU (mmol/L)', min_value=1, max_value=35, value=18, step=1, key='Glu(mmol/L)')
    hb = st.slider('Hb (g/L)', min_value=40, max_value=220, value=130, step=1, key='Hb(g/L)')
    scr = st.slider('Scr (μmol/L)', min_value=30, max_value=1200, value=100, step=10, key='Scr(μmol/L)')
    dbp = st.slider('DBP (mmHg)', min_value=40, max_value=180, value=80, step=1, key='DBP(mmHg)')
    neu = st.slider('Neu (10^9/L)', min_value=1, max_value=25, value=5, step=1, key='Neu（10^9/L）')
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='reperfusion therapy (yes1,no0)')
    acei_arb = st.selectbox('ACEI/ARB', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ACEI/ARB(yes1,no0)')
    atrial_fibrillation = st.selectbox('Atrial Fibrillation', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='atrial fibrillation(yes1,no0）')
    gender = st.radio('Gender', ['male', 'female'], format_func=lambda x: x.title(), index=0, key='gender（male0;female1）')

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
        st.write(f'Prediction: {prediction * 100:.2f}%')  # Convert probability to percentage
    except Exception as e:
        st.error(f'Error: {str(e)}')
