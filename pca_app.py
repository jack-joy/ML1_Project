import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import LeagueDashPlayerStats
from unidecode import unidecode
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
import time

# -------------------------------------
# STREAMLIT PAGE
# -------------------------------------
st.title("NBA PCA Explorer (2024–25)")

st.write("""
This app fetches 2024–25 NBA data, merges salaries, runs PCA,  
and displays the resulting components for clustering or KNN.
""")

# -------------------------------------
# FUNCTIONS
# -------------------------------------

def get_nba_stats():
    stats_base = LeagueDashPlayerStats(
        season='2024-25',
        per_mode_detailed='PerGame'
    ).get_data_frames()[0]
    time.sleep(1)

    stats_adv = LeagueDashPlayerStats(
        season='2024-25',
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    time.sleep(1)

    stats_base["PLAYER_NAME"] = stats_base["PLAYER_NAME"].apply(unidecode)
    stats_adv["PLAYER_NAME"] = stats_adv["PLAYER_NAME"].apply(unidecode)

    nba_df = stats_base.merge(stats_adv, on="PLAYER_ID", suffixes=("_base", "_adv"))

    if "PLAYER_NAME_adv" in nba_df.columns:
        nba_df.drop(columns=["PLAYER_NAME_adv"], inplace=True)
        nba_df.rename(columns={"PLAYER_NAME_base": "PLAYER_NAME"}, inplace=True)

    return nba_df


def scrape_salaries():
    all_rows = []
    for i in range(1, 14):
        url = f"https://www.espn.com/nba/salaries/_/year/2025/page/{i}"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html, "html.parser")

        names = [a.text.strip()
                 for a in soup.select("td:has(a)")[::2]]

        salaries = [td.text.strip()
                    for td in soup.select('td[style*="text-align:right"]:not([width])')]

        df = pd.DataFrame({"PLAYER_NAME": names, "SALARY": salaries})
        all_rows.append(df)

    df_salaries = pd.concat(all_rows, ignore_index=True)

    # salary cleaning
    df_salaries["SALARY"] = (
        df_salaries["SALARY"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df_salaries["SALARY"] = pd.to_numeric(df_salaries["SALARY"], errors="coerce")

    return df_salaries


def merge_stats_salaries(stats, salaries):
    stats["helper"] = (
        stats["PLAYER_NAME"].str.lower()
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    salaries["helper"] = (
        salaries["PLAYER_NAME"].str.lower()
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    merged = pd.merge(
        stats,
        salaries[["helper", "SALARY"]],
        on="helper",
        how="left"
    )

    merged.drop(columns=["helper"], inplace=True)
    return merged


def run_pca(df):
    # filters
    df_model = df[df["MIN_base"] >= 15].copy()

    pca_features = [
        'PTS', 'FGA_base', 'FG3A', 'FTM', 'FTA',
        'TS_PCT', 'EFG_PCT',
        'OREB', 'DREB', 'OREB_PCT', 'DREB_PCT',
        'AST', 'AST_TO',
        'TOV', 'PF',
        'STL', 'BLK', 'DEF_RATING',
        'USG_PCT', 'PACE',
        'PLUS_MINUS'
    ]

    df_model = df_model.dropna(subset=pca_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[pca_features])

    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(5)])
    pca_df["PLAYER_NAME"] = df_model["PLAYER_NAME"].values

    return pca, pca_df


# -------------------------------------
# INTERFACE
# -------------------------------------

if st.button("Fetch & Process Data (Takes ~10 seconds)"):
    with st.spinner("Loading NBA base & advanced stats..."):
        stats = get_nba_stats()

    with st.spinner("Scraping salary data..."):
        salaries = scrape_salaries()

    with st.spinner("Merging datasets..."):
        df_merged = merge_stats_salaries(stats, salaries)

    st.success("Data successfully merged!")
    st.write(df_merged.head())

    with st.spinner("Running PCA..."):
        pca, pca_df = run_pca(df_merged)

    st.success("PCA completed!")

    st.write("### PCA Components (PC1–PC5)")
    st.dataframe(pca_df)

    st.write("### Explained Variance")
    st.bar_chart(pca.explained_variance_ratio_)

    st.download_button(
        "Download PCA CSV",
        pca_df.to_csv(index=False),
        "pca_components.csv",
        "text/csv"
    )
