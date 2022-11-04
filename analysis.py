import matplotlib.pyplot as plt
from datahandler import *

def get_class_dist(df):
    ct_win = df.loc[df['round_winner'] == 1].shape[0]
    t_win = df.shape[0] - ct_win
    return ct_win, t_win

if __name__ == "__main__":   
    df = load_df('./csgo_clean.csv')
    ct, t =get_class_dist(df)
    print((ct/(t+ct)), (t/(ct+t)))