import streamlit as st


def main():
    st.title("Heart Attack Analysis Web App")

    st.markdown(
        """The Heart Attack Analysis Web App is a comprehensive platform that 
        combines data collection and predictive analytics to assess the likelihood 
        of heart attacks. Through user-provided medical data, the application 
        performs rigorous preprocessing to ensure data consistency and compatibility 
        with its predictive model. Leveraging machine learning algorithms trained on 
        historical heart health data, the app generates personalized predictions regarding 
        the risk of a heart attack. By seamlessly integrating data input and 
        predictive analysis, the web app empowers users to proactively monitor their 
        heart health and take informed actions to mitigate potential risks."""
    )


if __name__ == "__main__":
    main()
