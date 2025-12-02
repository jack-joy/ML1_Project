# Imports
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------------------------------------------------------------------------------------
# Title
# -------------------------------------------------------------------------------------------------------------------------------------
st.title("NBA Dashboard")

# -------------------------------------------------------------------------------------------------------------------------------------
#Side Bar
# -------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.title("NBA Dashboard")
tab = st.sidebar.radio("Navigation", ["README", "Data Table", "Exploratory Data Analysis", 'Models'])

# -------------------------------------------------------------------------------------------------------------------------------------
# README
# -------------------------------------------------------------------------------------------------------------------------------------
if tab == "README":
    st.title("**Machine Learning 1 Final Project**: NBA General Manager Trade Simulation")
    st.write("""
    **Members:** Eddie, Chase, Adam, Neel, Timothy, Jack, Harrison

### **OVERVIEW**:
Using all of the models we have used this semester, we will analyze NBA player data from the 2024-2025 seasons to answer various research questions. We clean and transform the data, explore it through descriptive statistics and visualizations, and build multiple predictive models depending on the prediction type. Finally, we deploy a Streamlit app to showcase our findings in an interactive way.

### **RESEARCH QUESTIONS & OBJECTIVES**:
1. Can we accurately predict player salary, all-star nominations, and other accomplishment features?
2. Assess which players are undervalued/overvalued in order to build a new team and predict transfers.
3. Can we predict the categorical variable of whether a player will be an all-star based on their season statistics?
4. KNN: Can we classify players into different archetypes based on their playing style and performance metrics?
5. Create a trade analysis model based on projected evaluated salaries + other evaluative metrics.
6. Predict win/loss for next season based on current roster and player statistics.


### **MODELS**:
1. Multiple Linear Regression (Add Polynomial?) -- Adam 
2. Logistic Regression -- Tim
3. KNN: K-Nearest Neighbors -- Chase
4. K-Means Clustering -- Eddie
    - Clustering players into different archetypes based on performance metrics
    - Clustering players based on their valuation to identify undervalued/overvalued players
5. PCA Model -- 
6. MLP Neural Network --
7. Model Extension -- Harrison

### **PROJECT/FILE STRUCTURE**


### **INSTRUCTIONS FOR VIEWERS - HOW TO RUN**
Our x file does y
Our z file does a
...

### **DATA SOURCES**: 
1. *NBA API:* https://github.com/swar/nba_api
2. ESPN Salary Data -- Scraped from https://www.espn.com/nba/salaries
3. `2012-2023 NBA Stats.csv
             
### ** Viewing Data**
1. Run get_data.py
2. Run scrape_salaries.py
3. Run data_cleaning.py
    """)

# -------------------------------------------------------------------------------------------------------------------------------------
# Data Table
# -------------------------------------------------------------------------------------------------------------------------------------
#if tab == "Data Table":
    
# -------------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------------------------------------------------------------------------------------------
#if tab == "Exploratory Data Analysis":
    
# -------------------------------------------------------------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------------------------------------------------------------
if tab == "Models":
    st.title("Model Training & Prediction")
    st.subheader("Select Model")
    model_choice = st.radio("Choose a model:", 
                            ["Multiple Linear Regression", "Logistic Regression", "K-Means", "KNN", "PCA", "MLP Neural Network"])

# Multiple Linear Regression
    if model_choice == "Multiple Linear Regression":
        st.write("Multiple Linear Regression")

# Logistic Regression
    if model_choice == "Logistic Regression":
        st.write("Logistic Regression")

# K-Means
    if model_choice == "K-Means":
        st.write("K-Means")

# KNN
    if model_choice == "KNN":
        st.write("KNN")

# PCA
    if model_choice == "PCA":
        st.write("PCA")

# MLP Neural Network
    if model_choice == "MLP Neural Network":
        st.write("MLP Neural Network")

# To run this dashboard, use the terminal command:
# streamlit run nba_model_app.py
# then the dashboard will automatically open in your web browser.