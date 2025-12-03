"""
NBA LEBRON Prediction Model
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# LOAD DATA
# ============================================================================

def select_player_row(group):
    """Handle players who played for multiple teams"""
    if len(group) == 1:
        return group
    else:
        multi_team_rows = group[group['Team'].isin(['2TM', '3TM', '4TM'])]
        if len(multi_team_rows) > 0:
            return multi_team_rows
        else:
            return group.head(1)

# Load current season per game stats
url_pg = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html#per_game_stats"
nba_pg = pd.read_html(url_pg, header=0)[0]
nba_pg = nba_pg[nba_pg['Player'] != 'Player']
nba_pg = nba_pg.groupby('Player').apply(select_player_row).reset_index(drop=True)
nba_pg = nba_pg.drop(columns=['Rk', 'Age', 'GS', 'MP', 'Team'], errors='ignore')

# Load current season advanced stats
url_adv = "https://www.basketball-reference.com/leagues/NBA_2025_advanced.html#advanced"
nba_value = pd.read_html(url_adv, header=0)[0]
nba_value = nba_value[nba_value['Player'] != 'Player']
nba_value = nba_value.groupby('Player').apply(select_player_row).reset_index(drop=True)
nba_value = nba_value.drop(columns=['Rk', 'Age', 'G', 'GS', 'MP', 'Team', 'Pos', 'Awards'], errors='ignore')

# Load LEBRON data
nba_lebron = pd.read_csv('/Users/adamchow/Library/CloudStorage/Box-Box/UVA/MSDS/Fall 2025/DS 6021/Final Project/ML1_Project/LEBRON Data - Sheet1.csv') # may need to move sheet to same directory
nba_lebron = nba_lebron.rename(columns={'Rank': 'LEBRON_Rank'})

# Merge all data
nba = nba_pg.merge(nba_value, on='Player', how='inner')\
            .merge(nba_lebron, on='Player', how='inner')

print(f"Data loaded: {nba.shape}")

# ============================================================================
# TRAIN MODEL
# ============================================================================

# Define features
final_features = ['3P%', '3PAr', 'eFG%', 'FT%', 'AST', 'TRB', 'STL', 'BLK', 
                  'USG%', 'WS/48', 'VORP', 'Age', 'Minutes']

# Prepare data
model_data = nba[final_features + ['LEBRON']].dropna()
X = model_data[final_features]
y = model_data['LEBRON']

print(f"Final dataset shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train OLS model
print("\n=== TRAINING OLS MODEL ===")
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train_const).fit()

# Make predictions
y_train_pred = model.predict(X_train_const)
y_test_pred = model.predict(X_test_const)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Print results
print("\n=== MODEL RESULTS ===")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

print("\n=== MODEL SUMMARY ===")
print(model.summary())