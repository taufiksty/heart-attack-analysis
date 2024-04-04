import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    st.title("Dataset Heart Attack Analysis")

    st.markdown(
        """This dataset was taken from the Kaggle website. 
        Heart attack analysis & prediction dataset is a dataset for heart attack classification. 
        This dataset is a heart attack dataset that has the highest votes among the others, as many as 3600+ users have voted."""
    )

    st.markdown(
        "[See the dataset 🚀](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data)"
    )

    st.markdown(
        """This dataset has 300+ rows and 14 columns or features. These columns include:"""
    )

    st.write("1. age: Age of the patient")
    st.write("2. sex : Sex of the patient")
    st.write(
        "3. cp : Chest Pain type chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic)"
    )
    st.write("4. trtbps : resting blood pressure (in mm Hg)")
    st.write("5. chol — serum cholestoral in mg/dl")
    st.write("6. fbs — (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")
    st.write("7. restecg — resting electrocardiographic results")
    st.write("8. thalachh — maximum heart rate achieved")
    st.write("9. exng — exercise induced angina (1 = yes; 0 = no)")
    st.write("10. oldpeak — ST depression induced by exercise relative to rest")
    st.write("11. slp — the slope of the peak exercise ST segment")
    st.write("12. caa — number of major vessels (0–3) colored by flourosopy")
    st.write("13. thall: thallassemia")
    st.write(
        "14. output — (the predicted attribute) — diagnosis of heart disease (angiographic disease status)"
    )

    df = pd.read_csv("data/heart.csv")
    sns.set_style("whitegrid")

    # Heart Disease Gender Wise
    st.header("Heart Disease Gender Wise")
    fig, _ = plt.subplots(figsize=(6, 4))
    sns.countplot(x="output", data=df, hue="sex")
    plt.legend(["Female", "Male"])
    plt.title("Heart Disease Gender Wise")
    plt.xticks([0, 1], ["No Heart Disease", "Heart Disease"])
    plt.xlabel("")
    st.pyplot(fig)

    # Heart Disease By Chain Pain Type
    st.header("Heart Disease By Chain Pain Type")
    fig, _ = plt.subplots(figsize=(6, 4))
    f = sns.countplot(x="cp", data=df, hue="output")
    plt.legend(["No Disease", "Disease"])
    f.set_title("Heart Disease Presence Distribution")
    f.set_xticks([0, 1, 2, 3])
    f.set_xticklabels(
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    plt.xlabel("")
    st.pyplot(fig)

    # Age vs Max Heart Rate for Heart Disease
    st.header("Age vs Max Heart Rate for Heart Disease")
    fig, _ = plt.subplots(figsize=(8, 6))
    plt.scatter(x=df.age[df.output == 1], y=df.thalachh[df.output == 1], c="salmon")
    plt.scatter(x=df.age[df.output == 0], y=df.thalachh[df.output == 0], c="lightblue")
    plt.title("Heart Disease in Function of Age and Max Heart Rate")
    plt.xlabel("Age")
    plt.ylabel("Max Heart Rate")
    plt.legend(["Disease", "No Disease"])
    st.pyplot(fig)

    # Heart Disease Presence Distribution
    st.header("Heart Disease Presence Distribution")
    fig, _ = plt.subplots(figsize=(6, 4))
    f = sns.countplot(x="output", data=df)
    f.set_title("Heart Disease Presence Distribution")
    f.set_xticks([0, 1])
    f.set_xticklabels(["No Disease", "Disease"])
    f.set_xlabel("")
    st.pyplot(fig)

    # Correlation (Heat Map)
    st.header("Correlation (Heat Map)")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(data=corr, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    bottom, top = ax.get_xlim()
    ax.set_ylim(bottom=bottom + 0.5, top=top - 0.5)
    st.pyplot(fig)

    # Plot age category wise patient who diagnosed with heart disease
    bins = list(np.arange(25, 85, 5))
    df["age_category"] = pd.cut(df["age"], bins=bins)

    # Get the count of positive heart disease diagnosed age category wise
    count_of_positive_heart_disease_diagnosed_by_age_category = (
        df[df["output"] == 1].groupby("age_category", observed=True)["age"].count()
    )
    st.header("Age Distribution of Patients with Positive Heart Diagnosis")
    fig, _ = plt.subplots(figsize=(8, 6))
    count_of_positive_heart_disease_diagnosed_by_age_category.plot(
        kind="bar", color="red"
    )
    plt.title("Age Distribution of Patients with Positive Heart Diagnosis")
    st.pyplot(fig)

    # Get the count of negative heart disease diagnosed age category wise
    count_of_negative_heart_disease_diagnosed_by_age_category = (
        df[df["output"] == 0].groupby("age_category", observed=True)["age"].count()
    )
    st.header("Age Distribution of Patients with Negative Heart Diagnosis")
    fig, _ = plt.subplots(figsize=(8, 6))
    count_of_negative_heart_disease_diagnosed_by_age_category.plot(
        kind="bar", color="blue"
    )
    plt.title("Age Distribution of Patients with Negative Heart Diagnosis")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
