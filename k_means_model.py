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

len(nba.columns.tolist())
sorted(nba.columns.tolist())


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
    'DEF_RATING': 'Defensive Rating',
    'OREB_PCT': 'Offensive Rebound Percentage',
    'DREB_PCT': 'Defensive Rebound Percentage',
    'TS_PCT': 'True Shooting Percentage',
    'USG_PCT': 'Usage Percentage', 
    'FG3A' : '3 Point Attempts',
    'FG3_PCT' : '3 Point Percentage',
    'SALARY': 'Salary',
    'NET_RATING': 'Net Rating'
}
nba = nba[[key for key in feature_map.keys()]] 

###########################################################################
# ML PIPELINE AND MODEL
FEATURES = ['PTS', 'AST', 'REB', 'TS_PCT', 'USG_PCT']
X = nba[FEATURES]

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, 
                      init='k-means++',
                      n_init=10,
                      random_state=42))
])

# Choose Best K using Silhouette Score and Elbow Method
K_VALS = range(2,12)
silhouette_scores = []
wcss = []

for k in K_VALS:
    pipeline.set_params(kmeans__n_clusters=k)
    pipeline.fit(X)
    labels = pipeline['kmeans'].labels_
    intertia = pipeline['kmeans'].inertia_
    wcss.append(intertia)
    sil_score = silhouette_score(X, labels)
    silhouette_scores.append(sil_score)

# Plot Elbow: Based on this, it looks like optimal K is 5
plt.subplots(figsize=(12,5))
sns.lineplot(x=K_VALS, y=wcss, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters K')
plt.ylabel('Within Cluster Sum of Squares (WCSS)')
plt.xticks(K_VALS)
plt.show()

# Plot Silhouette Scores
plt.subplots(figsize=(12,5))
sns.lineplot(x=K_VALS, y=silhouette_scores, marker='o')
plt.title('Silhouette Scores For Optimal K')
plt.xlabel('Number of Clusters K')
plt.ylabel('Silhouette Score')
plt.xticks(K_VALS)
plt.show()

# Get Optimal K
optimal_k = K_VALS[np.argmax(silhouette_scores)]
print(f'Optimal K Based on Silhouette Score: {optimal_k}')

# Final Model
pipeline.set_params(kmeans__n_clusters=optimal_k)
pipeline.fit(X)
labels = pipeline['kmeans'].labels_
nba['cluster'] = labels



#############################################################################
# Assess Results and Analyze Clusters
# Analyze Clusters
cluster_summary = nba.groupby('cluster').mean().reset_index()
cluster_summary.style.format("{:,.2f}", subset=cluster_summary.columns[1:])

# Scatter of Salery vs Points Colored by Cluster
plt.subplots(figsize=(12,6))
sns.scatterplot(data=nba, x='PTS', y='SALARY', hue='cluster', palette='Set2')
plt.title('Player Clusters: Salary vs Points')
plt.xlabel('Points Per Game')
plt.ylabel('Salary')
plt.show()

# Scatter of DREB vs OREB Colored by Cluster
plt.subplots(figsize=(12,6))
sns.scatterplot(data=nba, x='DREB', y='OREB_PCT', hue='cluster', palette='Set2')
plt.title('Player Clusters: Defensive Rebounds vs Offensive Rebound Percentage')
plt.xlabel('Defensive Rebounds Per Game')
plt.ylabel('Offensive Rebound Percentage')
plt.show()
