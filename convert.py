import pandas as pd
import numpy as np
import csv
import tqdm

if __name__ == "__main__":
    maps = {
            'de_dust2' : 0,
            'de_inferno' : 1,
            'de_overpass' : 2,
            'de_mirage' : 3,
            'de_nuke' : 4,
            'de_cache' : 5,
            'de_train' : 6,
            'de_vertigo' : 7
            }
    round_winner = { 'T':0, 'CT':1 }
    with open("csgo_round_snapshots.csv", "r") as f:
        df = pd.read_csv(f)
        n = len(df)
        map_i = df.columns.get_loc('map')
        win_i = df.columns.get_loc('round_winner')
        bomb_plant_i = df.columns.get_loc('bomb_planted')
        for i in tqdm.tqdm(range(0,n)):
            df.iloc[i,map_i] = maps[df.iloc[i,map_i]]
            df.iloc[i,win_i] = round_winner[df.iloc[i,win_i]]
            df.iloc[i, bomb_plant_i] = int(df.iloc[i,bomb_plant_i])
    with open("csgo_clean.csv", "w") as f:
        df.to_csv(f)

