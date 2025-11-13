import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

def spider(url):
  useragent = f'ds6021rt/0.0 (yht7nf@virginia.edu) python-requests/{requests.__version__}'
  headers = {'User-Agent': useragent, 'From': 'yht7nf@virginia.edu'}

  r = requests.get(url, headers=headers)
  mysoup = BeautifulSoup(r.text, 'html.parser')

  player_list = mysoup.select('td:has(a)')
  player_list = player_list[::2]
  players = []
  for i in range(len(player_list)):
    name = player_list[i].find("a").get_text(strip=True)
    players.append(name)
  salaries = mysoup.select('td[style*="text-align:right"]:not([width])')
  salaries = [td.get_text(strip=True) for td in salaries]

  df = pd.DataFrame({
    "PLAYER_NAME": players,
    "SALARY": salaries
  })
  return df

if __name__ == "__main__":
  new_df = pd.DataFrame()

  for i in range(1, 14):
    url = f"https://www.espn.com/nba/salaries/_/year/2025/page/{i}"
    df = spider(url)
    new_df = pd.concat([new_df, df], ignore_index=True)

  new_df.to_csv("DATA/24-25_salaries.csv", index=False)