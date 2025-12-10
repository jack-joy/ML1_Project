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
    'NET_RATING': 'Net Rating', 
    'BLK': 'Blocks'
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
    fig1, ax1 = plt.subplots(figsize=(12,5), dpi=300)
    sns.lineplot(x=range_k, y=wcss_scores, marker='o', ax=ax1)
    ax1.set_title("Elbow Method For Optimal K", fontweight="bold")
    ax1.set_xlabel("Number of Clusters K")
    ax1.set_ylabel("Within Cluster Sum of Squares (WCSS)")
    ax1.set_xticks(range_k)
    # --- Silhouette Plot ---
    fig2, ax2 = plt.subplots(figsize=(12,5), dpi=300)
    sns.lineplot(x=range_k, y=sil_scores, marker='o', ax=ax2)
    ax2.set_title("Silhouette Scores For Optimal K", fontweight="bold")
    ax2.set_xlabel("Number of Clusters K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_xticks(range_k)
    # Returns Plots    
    return fig1, fig2

# Final Model Function
def final_model_and_labels(data, n_clusters):
    '''Function to create final KMeans model and return labels'''
    pipeline = create_pipeline(n_clusters)
    pipeline.fit(data)
    labels = pipeline['kmeans'].labels_
    return pipeline, labels

# Final Graph Function
def graph_clusters(data, x_axis, y_axis, color, model_type, alpha=0.8):
    '''Function to graph clusters'''
    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    sns.scatterplot(
        data=data, 
        x=x_axis, 
        y=y_axis,
        hue=color, 
        palette='Set2', 
        alpha=alpha
        )
    ax.set_title(f'K-Means Clustering: {model_type}', fontweight='bold')
    ax.set_xlabel(f'{x_axis}')
    ax.set_ylabel(f'{y_axis}')
    return fig




#######################################################################################
# FUNCTION CALLS AND MODEL EXECUTION
# This prevents the code from running if this file is imported as a module
if __name__ == "__main__":
    df = load_and_clean_data()
    attribute_options = sorted(df.columns.tolist())

    # Feature & Model Selection
    FEATURES = {
        'Player Archetype' : ['Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 'True Shooting Percentage', 
                            'Usage Percentage', 'Defensive Rating', 'Offensive Rating', 
                            '3 Pointer Attempts', '3 Pointer Made', 'Steals'],
        'Player Valuation' : ['Salary', 'Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 
                            'Steals', '3 Pointer Made'],
        'Custom Model' : [] # User input features (dropdown of all features)
    } # User will choose one of these options in the app
    K_VALUE = 'Auto' # User can choose Auto or specify K integer value
    MODEL_TYPE = 'Player Archetype'

    X = df[FEATURES[MODEL_TYPE]]

    # Test Different K Values
    sil_scores, wcssm, opt_k = test_k_values(X)
    elbow_graph, sil_graph = graph_elbow_silhouette(wcssm, sil_scores)
    elbow_graph.show()
    sil_graph.show()

    # Build Final Model
    k_value = opt_k if K_VALUE == 'Auto' else int(K_VALUE)
    pipeline, labels = final_model_and_labels(X, k_value)
    df['Cluster'] = labels

    # Final Assessment
    # Styled DataFrame Summary
    summary = df.groupby(['Cluster'])[FEATURES[MODEL_TYPE]].mean().reset_index()
    summary.style.format("{:,.1f}", subset=summary.columns[1:])

    # Plot
    options = summary.columns.tolist()
    X_AXIS = 'Points'
    Y_AXIS = 'Assists'
    COLOR = 'Cluster'
    ALPHA = .8

    cluster_graph = graph_clusters(df, X_AXIS, Y_AXIS, COLOR, MODEL_TYPE, ALPHA)
    cluster_graph.show()