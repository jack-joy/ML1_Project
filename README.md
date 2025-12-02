# **Machine Learning 1 Final Project**: NBA General Manager Trade Simulation
#### **Members** Eddie, Chase, Adam, Neel, Timothy, Jack, Harrison

---------------------------------------------------------------------------------------------------------------------

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
6. MLP Neural Network -- Jack 
7. Model Extension -- Harrison

### **PROJECT/FILE STRUCTURE**

1. README.md
2. data_cleaning.py
3. data_cleaning.ipynb
4. get_data.py
5. get_data.ipynb
6. k_means_model.py
7. k_means_model.ipynb
8. knn_model.py

### **DATA SOURCES**: 
1. *NBA API:* https://github.com/swar/nba_api
2. ESPN Salary Data -- Scraped from https://www.espn.com/nba/salaries
3. `2012-2023 NBA Stats.csv`

### ** Viewing Data**
1. Run get_data.py
2. Run scrape_salaries.py
3. Run data_cleaning.py