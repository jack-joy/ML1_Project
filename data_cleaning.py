import pandas as pd
import numpy as np

# Load Data
nba = pd.read_csv('DATA/nba_base_data.csv')
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