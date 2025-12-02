# K Means Clustering For the NBA Data
# We will use SKLean module primarily
# First will conduct some exploratory analysis
# Then will create a more modularlized approach that can be called by the Streamlit app


######################################################################################
# Import Libraries
# Data Manipulation
import numpy as np
import pandas as pd
# SKLearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns



######################################################################################
# FUNCTION BASED MODEL
def load_and_clean_data():
    '''Function to load NBA data with salaries'''
    df = pd.read_csv('DATA/nba_data_with_salaries.csv')
    feature_map = {
    'PTS': 'Points',
    'FGA_base' : 'Shots Attempted',
    'FGM_base' : 'Shots Made',
    'AST': 'Assists',
    'REB': 'Rebounds',
    'DREB': 'Defensive Rebounds',
    'STL': 'Steals',
    'OFF_RATING': 'Offensive Rating',
    'DEF_RATING': 'Defensive Rating',
    'OREB_PCT': 'Offensive Rebound Percentage',
    'DREB_PCT': 'Defensive Rebound Percentage',
    'TS_PCT': 'True Shooting Percentage',
    'USG_PCT': 'Usage Percentage', 
    'FG3A' : '3 Pointer Attempts',
    'FG3M' : '3 Pointer Made',
    'FG3_PCT' : '3 Pointer Percentage',
    'SALARY': 'Salary',
    'NET_RATING': 'Net Rating'
    }
    df.rename(columns=feature_map, inplace=True)
    return df

# Pipeline
def create_pipeline(n_clusters):
    '''Function to create SKLearn Pipeline for KMeans Clustering'''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, 
                          init='k-means++',
                          n_init=10,
                          random_state=42))
    ])
    return pipeline

# Test K Vals with Silhouette Score and Elbow Method
def test_k_values(parameters, range_k = range(2,12)):
    '''Function to test different K values for KMeans'''
    silhouette_scores = []
    wcss = []
    for k in range_k:
        pipeline = create_pipeline(n_clusters=k)
        pipeline.fit(parameters)
        labels = pipeline['kmeans'].labels_
        inertia = pipeline['kmeans'].inertia_
        wcss.append(inertia)
        scaled_x = pipeline['scaler'].transform(parameters)
        sil_score = silhouette_score(scaled_x, labels)
        silhouette_scores.append(sil_score)
    optimal_k = range_k[np.argmax(silhouette_scores)]
    print(f'Optimal K based on Silhouette Score: {optimal_k}')
    return silhouette_scores, wcss, optimal_k

# Graphing Functions
def graph_elbow_silhouette(wcss_scores, sil_scores, range_k=range(2,12)):
    """Graph Elbow and Silhouette Scores and return both figures."""
    # --- Elbow Plot ---
    fig1, ax1 = plt.subplots(figsize=(12,5))
    sns.lineplot(x=range_k, y=wcss_scores, marker='o', ax=ax1)
    ax1.set_title("Elbow Method For Optimal K", fontweight="bold")
    ax1.set_xlabel("Number of Clusters K")
    ax1.set_ylabel("Within Cluster Sum of Squares (WCSS)")
    ax1.set_xticks(range_k)
    # --- Silhouette Plot ---
    fig2, ax2 = plt.subplots(figsize=(12,5))
    sns.lineplot(x=range_k, y=sil_scores, marker='o', ax=ax2)
    ax2.set_title("Silhouette Scores For Optimal K", fontweight="bold")
    ax2.set_xlabel("Number of Clusters K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_xticks(range_k)
    # Returns Plots    
    return fig1, fig2




# FUNCTION CALLS AND MODEL EXECUTION
FEATURES = {
    'Player Archetype' : ['Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 'True Shooting Percentage', 
                          'Usage Percentage', 'Defensive Rating', 'Offensive Rating', 
                          '3 Pointer Attempts', '3 Pointer Made', 'Steals'],
    'Player Valuation' : ['Salary', 'Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 
                          'Steals', '3 Pointer Made'],
    'Custom Model' : [] # User input features (dropdown of all features)
} # User will choose one of these options in the app
K_VALUE_OVERRIDE = np.nan
MODEL_TYPE = 'Player Archetype'

X = df[FEATURES[MODEL_TYPE]]
###########################################################################
# ML PIPELINE AND MODEL



# Function Calls to Get Optimal K
sil_score, wcss, optimal_k = test_k_values(X)
graph_elbow_silhouette(wcss, sil_score)

# Final Model (with chosen K to override)
if K_VALUE_OVERRIDE is not np.nan:
    print(f'Overriding K to: {K_VALUE_OVERRIDE}')
    pipeline = create_pipeline(K_VALUE_OVERRIDE)
else: 
    print(f'Using Optimal K of: {optimal_k}')
    pipeline = create_pipeline(optimal_k)
pipeline.fit(X)
labels = pipeline['kmeans'].labels_
df['Cluster'] = labels

#############################################################################
# Assess Results and Analyze Clusters

# Styled DataFrame Summary
summary = df.groupby(['Cluster'])[FEATURES[MODEL_TYPE]].mean().reset_index()
summary.style.format("{:,.2f}", subset=summary.columns[2:])

# Plot
options = summary.columns.tolist()
X_AXIS = 'Points'
Y_AXIS = 'Assists'
COLOR = 'Cluster'
ALPHA = .8

plt.subplots(figsize=(12,6))
sns.scatterplot(data=df, x=X_AXIS, y=Y_AXIS,
                hue=COLOR, palette='Set2', 
                alpha=ALPHA)
plt.title(f'K-Means Clustering: {MODEL_TYPE}', fontweight='bold')
plt.xlabel(f'{X_AXIS} Per Game')
plt.ylabel(f'{Y_AXIS} Per Game')
plt.show()