# **Machine Learning 1 Final Project: NBA General Manager Trade Simulation**

**Members:** Eddie, Chase, Adam, Neel, Timothy, Jack, Harrison

=======
## **Overview**
Using all of the models we have used this semester, we analyze NBA player data from the 2024–2025 seasons to answer a set of research questions. We clean and transform the data, explore it with descriptive statistics and visualizations, and build multiple predictive models depending on the task. Finally, we deploy a Streamlit app to showcase our findings interactively.

---
>>>>>>> 64e7703968b10405aefa7852841b3bdc1fa9111b

## **Research Questions & Objectives**
1. Can we accurately predict player salary, all-star nominations, and other accomplishment features?  
2. Can we classify whether a player will be an all-star using season statistics?  
3. Can we cluster players based on performance metrics and valuation to identify archetypes or undervalued players?  
4. Can we classify players into different salary tiers using per game performance metrics?  
5. Build a trade analysis model based on projected evaluated salaries and other evaluative metrics.  
6. Predict next season’s win/loss record based on current roster and player statistics.

---

## **Models**
1. Multiple Linear Regression (Polynomial extensions optional)  
2. Logistic Regression  
3. K-Nearest Neighbors (KNN)  
4. K-Means Clustering  
   - Clustering players into performance archetypes  
   - Clustering by valuation to identify overvalued/undervalued players  
5. Principal Component Analysis (PCA)  
6. MLP Neural Network — Trade Analysis

---

## **App Structure**
1. **Page 1:** README  
2. **Page 2:** Interactive data table  
3. **Page 3:** Exploratory Data Analysis (EDA)  
4. **Page 4:** Statistical model pages

---

## **Instructions for Viewers — How to Run**
1. Create the Conda environment: `conda env create -f environment.yml`
2. Activate Environment: `conda activate nba_ml_project`
3. Run Data Processing Scripts: `python scrape_salaries.py` and `python get_clean_data.py`
4. Run Streamlit App: `streamlit run nba_model_app.py`

--

## **Data Sources**
1. NBA API: https://github.com/swar/nba_api
2. ESPN Salary Data: https://www.espn.com/nba/salaries
3. 2012–2023 NBA Stats.csv
