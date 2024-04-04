import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


with open("model/model3.pkl", "rb") as f:
    model = pickle.load(f)


def heart_attack_prediction(input_data):
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)

    df_train = pd.read_csv("data/heart.csv")

    scaler = StandardScaler()
    scaler.fit(df_train.iloc[:, :-1])
    std_data = scaler.transform(input_data_reshaped)

    print(std_data.shape)

    prediction = model.predict(std_data)

    if prediction[0] == 0:
        return "The person is not at risk of heart attack"
    else:
        return "The person is at risk of heart attack"


def main():
    st.title("Heart Attack Prediction Web App")

    # Getting input data from user
    age = st.number_input("Age", min_value=20, max_value=80, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    )
    resting_blood_pressure = st.number_input(
        "Resting Blood Pressure", min_value=30, value=None, placeholder="in mmHg"
    )
    cholesterol = st.number_input(
        "Serum Cholestoral", min_value=0, value=None, placeholder="in mg/dl"
    )
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    resting_ecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)",
            "Showing probable or definite left ventricular hypertrophy by Estes’ criteria",
        ],
    )
    max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved", min_value=0)
    exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    st_depression = st.number_input(
        "ST Depression Induced by Exercise Relative to Rest", min_value=0.0, step=0.1
    )
    slope = st.selectbox(
        "Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"]
    )
    number_of_major_vessels = st.number_input(
        "Number of Major Vessels", min_value=0, max_value=3
    )
    thalassemia = st.selectbox(
        "Thalassemia",
        [
            "Normal blood flow",
            "Abnormal blood flow during exercise",
            "Low blood flow during both rest and exercise",
            "No thallium visible in parts of the heart",
        ],
    )

    diagnosis = ""

    # Gender
    if gender == "Male":
        gender = 1
    else:
        gender = 0

    # Chest Pain Type
    if chest_pain_type == "Typical Angina":
        chest_pain_type = 1
    elif chest_pain_type == "Atypical Angina":
        chest_pain_type = 2
    elif chest_pain_type == "Non-Anginal Pain":
        chest_pain_type = 3
    else:
        chest_pain_type = 4

    # Fasting Blood Sugar
    if fasting_blood_sugar == "Yes":
        fasting_blood_sugar = 1
    else:
        fasting_blood_sugar = 0

    # Resting ECG
    if resting_ecg == "Normal":
        resting_ecg = 0
    elif (
        resting_ecg
        == "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)"
    ):
        resting_ecg = 1
    elif (
        resting_ecg
        == "Showing probable or definite left ventricular hypertrophy by Estes’ criteria"
    ):
        resting_ecg = 2

    # Exercise Induced Angina
    if exercise_induced_angina == "Yes":
        exercise_induced_angina = 1
    else:
        exercise_induced_angina = 0

    # Slope
    if slope == "Upsloping":
        slope = 1
    elif slope == "Flat":
        slope = 2
    else:
        slope = 3

    # Thalassemia
    if thalassemia == "Normal blood flow":
        thalassemia = 0
    elif thalassemia == "Abnormal blood flow during exercise":
        thalassemia = 1
    elif thalassemia == "Low blood flow during both rest and exercise":
        thalassemia = 2
    else:
        thalassemia = 3

    if st.button("Predict"):
        diagnosis = heart_attack_prediction(
            [
                age,
                gender,
                chest_pain_type,
                resting_blood_pressure,
                cholesterol,
                fasting_blood_sugar,
                resting_ecg,
                max_heart_rate_achieved,
                exercise_induced_angina,
                st_depression,
                slope,
                number_of_major_vessels,
                thalassemia,
            ]
        )

    if diagnosis == "The person is at risk of heart attack":
        st.warning(diagnosis)
    elif diagnosis == "The person is not at risk of heart attack":
        st.success(diagnosis)


if __name__ == "__main__":
    main()
