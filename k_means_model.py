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
# Load Data
nba = pd.read_csv('DATA/nba_data_with_salaries.csv')

######################################################################################
# We will try to group players in a few different ways:
    # Player Archetypes -- shooters, defenders, rim defenders, etc.
    # Offensive Style (Efficiency vs Volume Scorers, Playmakers, etc.)
    # Defensive Style (Rim Protectors, Perimeter Defenders, Non-Impact..., etc.)
    # Salary Based: Well Paid, Overpaid, Underpaid, etc.
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
subset = nba[list(feature_map.keys())]

# Feature Selection:
FEATURES = {
    'Player Archetype' : ['PTS', 'AST', 'REB', 'DREB', 'TS_PCT', 'USG_PCT', 'DEF_RATING', 
                          'OFF_RATING', 'FG3A', 'FG3M', 'STL'],
    'Player Valuation' : ['SALARY', 'PTS', 'AST', 'REB', 'DREB', 'STL', 'FG3M']
} # User will choose one of these options in the app
K_VALUES = {
    'Player Archetype' : 4, 
    'Player Valuation' : 4
}
# Label Interpretation:
LABEL_INTERPRET = {
    'Player Archetype' : {
        0 : 'Benchwarmer',
        1 : 'Defender',
        2 : 'Frontcourt Player',
        3 : 'Volume Scorer'
        },
    'Player Valuation' : {
        0 : 'Well Paid All-Rounder',
        1 : 'Well Paid Benchwarmer',
        2 : 'Overpaid Star',
        3 : 'Underpaid Defender'
}}

# MODEL TYPE
MODEL_TYPE = 'Player Valuation' # Options: 'Player Archetype', 'Player Valuation'
X = nba[FEATURES[MODEL_TYPE]]

###########################################################################
# ML PIPELINE AND MODEL
# Pipeline
def create_pipeline(n_clusters=4):
    '''Function to create SKLearn Pipeline for KMeans Clustering'''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=n_clusters, 
                          init='k-means++',
                          n_init=10,
                          random_state=42))
    ])
    return pipeline

# Choose Best K using Silhouette Score and Elbow Method
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

def graph_elbow_silhouette(wcss_scores, sil_scores, range_k = range(2,12)):
    '''Function to graph Eblow and Silhouette Scores'''
    # Elbow First
    plt.subplots(figsize=(12,5))
    sns.lineplot(x=range_k, y=wcss_scores, marker='o')
    plt.title('Elbow Method For Optimal K', fontweight='bold')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.xticks(range_k)
    plt.show()
    # Silhouette Next
        # 1 = well defined clusters
        # .5 = reasonable clusters
        # 0 = overlapping clusters
        # -1 = incorrect clustering
    plt.subplots(figsize=(12,5))
    sns.lineplot(x=range_k, y=sil_scores, marker='o')
    plt.title('Silhouette Scores For Optimal K', fontweight='bold')
    plt.xlabel('Number of Clusters K')
    plt.ylabel('Silhouette Score')
    plt.xticks(range_k)
    plt.show()

# Function Calls to Get Optimal K
sil_score, wcss, optimal_k = test_k_values(X)
graph_elbow_silhouette(wcss, sil_score)

# Final Model (with chosen K to override)
if MODEL_TYPE in K_VALUES.keys():
    pipeline = create_pipeline(K_VALUES[MODEL_TYPE])
else: 
    pipeline = create_pipeline(optimal_k)
pipeline.fit(X)
labels = pipeline['kmeans'].labels_

subset['Cluster'] = labels
if MODEL_TYPE in LABEL_INTERPRET.keys():
    subset['Cluster Name'] = subset['Cluster'].map(LABEL_INTERPRET[MODEL_TYPE])
else:
    subset['Cluster Name'] = 0
subset.rename(columns=feature_map, inplace=True)

#############################################################################
# Assess Results and Analyze Clusters
# Analyze Clusters
def summarize_clusters(subset):
    '''Function to summarize clusters'''
    summary = subset.groupby(['Cluster', 'Cluster Name']).mean().reset_index()
    return summary

summary = summarize_clusters(subset)
summary.style.format("{:,.2f}", subset=summary.columns[2:])


# Plots based on Model Type:
def graph_results(MODEL_TYPE, subset):
    '''Function to graph results based on model type'''
    ALPHA = .85
    if MODEL_TYPE == 'Player Archetype':
        # Scatter of 3 Point Attempts vs 3 Point Made Colored by Cluster
        plt.subplots(figsize=(12,6))
        sns.scatterplot(data=subset, x='3 Pointer Attempts', y='3 Pointer Made', 
                        hue='Cluster Name', palette='Set2', 
                        alpha=ALPHA)
        plt.title('Player Clusters: 3 Point Attempts vs Made', 
                fontweight='bold')
        plt.xlabel('3 Point Attempts Per Game')
        plt.ylabel('3 Point Made Per Game')
        plt.show()
        # Scatter of Assists vs Points Colored by Cluster
        plt.subplots(figsize=(12,6))
        sns.scatterplot(data=subset, x='Assists', y='Points', 
                        hue='Cluster Name', palette='Set2', 
                        alpha=ALPHA)
        plt.title('Player Clusters: Assists vs Points', fontweight='bold')
        plt.xlabel('Assists Per Game')
        plt.ylabel('Points Per Game')
        plt.show()
    elif MODEL_TYPE == 'Player Valuation':
        # Scattery of Salery vs Points
        plt.subplots(figsize=(12,6))
        sns.scatterplot(data=subset, x='Points', y='Salary', 
                        hue='Cluster Name', palette='Set2', 
                        alpha=ALPHA)
        plt.title('Player Clusters: Salary vs Points', fontweight='bold')
        plt.xlabel('Points Per Game')
        plt.ylabel('Salary')
        plt.show()
    else:
        print('No Graphs Available for This Model Type')

graph_results(MODEL_TYPE, subset)

######################################################################################
# Main Function
if __name__ == '__main__':
    pass
