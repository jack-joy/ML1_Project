# Imports
import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("NBA Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
# CHANGE THE OPTIONS WITHIN THE VIEW ONCE CODE COMPLETED
option = st.sidebar.selectbox("Choose a view:", ["EDA", "Model Results"])

# LOADING DATASET TEMPLATE
df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [10, 4, 6, 8]
})

# POTENTIAL DASHBOARD VIEW TEMPLATE
if option == "EDA":
    st.subheader("Exploratory Data Analysis")
    fig = px.scatter(df, x="x", y="y", title="Example Scatter Plot")
    st.plotly_chart(fig)

elif option == "Model Results":
    st.subheader("Model Performance")
    st.metric("Accuracy", "0.92")
    st.metric("Log Loss", "0.21")

# To run this dashboard, use the terminal command:
# streamlit run nba_model_app.py
# then the dashboard will automatically open in your web browser.