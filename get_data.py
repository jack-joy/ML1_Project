from nba_api.stats.endpoints import LeagueDashPlayerStats
import time
from unidecode import unidecode
import pandas as pd

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
output_path = os.path.join(project_dir, "player_stats.csv")
nba_df.to_csv(output_path, index=False)

print(f"Saved data to: {output_path}")

# Save to CSV
nba_df.to_csv('DATA/nba_base_data.csv', index=False)