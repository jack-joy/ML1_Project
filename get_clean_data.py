from nba_api.stats.endpoints import LeagueDashPlayerStats
import time
from unidecode import unidecode
import pandas as pd
import os

# Made DATA FOLDER
if not os.path.exists('DATA'):
    os.makedirs('DATA')

#base per-game stats
stats_base = LeagueDashPlayerStats(
    season='2024-25',
    per_mode_detailed='PerGame'
).get_data_frames()[0]
time.sleep(1)

#advanced stats
stats_adv = LeagueDashPlayerStats(
    season='2024-25',
    per_mode_detailed='PerGame',
    measure_type_detailed_defense='Advanced'
).get_data_frames()[0]
time.sleep(1)

stats_base['PLAYER_NAME'] = stats_base['PLAYER_NAME'].apply(unidecode)
stats_adv['PLAYER_NAME'] = stats_adv['PLAYER_NAME'].apply(unidecode)

#merge
nba_df = stats_base.merge(
    stats_adv,
    on='PLAYER_ID',
    suffixes=('_base', '_adv')
)

#drop duplicates
if 'PLAYER_NAME_adv' in nba_df.columns:
    nba_df.drop(columns=['PLAYER_NAME_adv'], inplace=True)
    nba_df.rename(columns={'PLAYER_NAME_base': 'PLAYER_NAME'}, inplace=True)

nba_df.head()

# Confirm the data
print(nba_df.head())

# Get the directory where this script is located
project_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(project_dir, "DATA/player_stats.csv")
nba_df.to_csv(output_path, index=False)

print(f"Saved data to: {output_path}")


###############################################################
# DATA CLEANING AND MERGING WITH SALARIES
# Load Data
nba = pd.read_csv('DATA/player_stats.csv')
salaries = pd.read_csv('DATA/24-25_salaries.csv')

nba['helper'] = nba['PLAYER_NAME'].str.lower()\
    .str.replace('.', '', regex=False)\
        .str.replace("'", '', regex=False)\
            .str.replace(' ', '', regex=False)
            
salaries['helper'] = salaries['PLAYER_NAME'].str.lower()\
    .str.replace('.', '', regex=False)\
        .str.replace("'", '', regex=False)\
            .str.replace(' ', '', regex=False)

merged = pd.merge(
    nba, 
    salaries[['helper', 'SALARY']],
    on='helper',
    how='outer', 
    validate='one_to_one', 
    indicator=True
)

merged = merged[merged['_merge'] == 'both']
merged = merged.drop(columns=['helper', '_merge'])

merged['SALARY'] = merged['SALARY']\
    .astype(str).str.replace('$', '', regex=False)\
        .str.replace(',', '', regex=False)
merged['SALARY'] = pd.to_numeric(merged['SALARY'], errors='coerce')

merged.to_csv('DATA/nba_data_with_salaries.csv', index=False)
print('\n\nFinalized Data Cleaning!')