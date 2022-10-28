import sklearn as sk
import numpy as np
import pandas as pd
import os
import pathlib
from normalizers import *

class DataLoader:
    """
    A class for loading and preprocessing a CSV file containing the CSGO dataset found here:
    """
    def __init__(self):
        self._df = None
        self._training_set = None
        self._test_set = None
        self._val_set = None

    def save_splits_to_path(self,dirpath,dirname, f_name):
        dirpath = os.path.join(dirpath, dirname)
        if not os.isdir(dirpath):
            os.mkdir(dirpath)
        names = ['_train.csv','_val.csv','_test.csv']
        sets = self.get_split_sets()
        for name,set in zip(names,sets):
            f_name+= name
            self._save_df_to_path(dirpath, f_name)
        return
    def get_training_set(self):
        try:
            assert not self._training_set.empty
            return self._training_set
        except:
            raise Exception("Splits haven't been set; run set_split")
    def get_val_set(self):
        try:
            assert not self._val_set.empty
            return self._val_set
        except:
            raise Exception("Splits haven't been set; run set_split")
    def get_test_set(self):
        try:
            assert not self._test_set.empty
            return self._test_set
        except:
            raise Exception("Splits haven't been set; run set_split")

    def get_full_set(self):
        try:
            assert not self._df.empty
            return self._df
        except:
            raise Exception("Splits haven't been set; run load_df_from_path()")
    def get_split_sets(self):
        train = self.get_training_set()
        val = self.get_val_set()
        test = self.get_test_set()
        return train, val, test
    def drop_numbers(self):
        self._df.select_dtypes(exclude=np.number)
    def numbers_only(self):
        self._df = self._df.select_dtypes(include=['number','bool'])
        self._df = self._df.astype(float)

    def load_df_from_path(self,f_name,dirpath='.'):
        path = os.path.join(dirpath, f_name)
        delim = "," if '.csv' in path else "\t"
        with open(path, "r") as f:
            df = pd.read_csv(f, delimiter=delim)
            
            self._df = df
        return 
    def _save_df_to_path(self,dirpath, f_name):
        path = os.path.join(dirpath, f_name)
        with open(path, "w") as f:
            self._df.to_csv(f)
        return 

    def set_split(self, train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True):
        assert (train_split + val_split + test_split) == 1
        n = self._df.shape[0]
        train_n = int(train_split * n)
        val_n = int(val_split * n)
        test_n = int(test_split * n)
        df = self._df.sample(frac=1) if shuffle else self._df
        train = df.iloc[:train_n].copy()
        val = df.iloc[train_n:train_n + val_n].copy()
        test = df.iloc[train_n + val_n:].copy()
        print(f"TRAIN/VAL/TEST: {train_n}/{val_n}/{test_n}")
        self._training_set = train
        self._val_set = val
        self._test_set = test
        return
    
    def normalize(self, method='z-score'):
        norm_f_dict = {
            'z-score' : df_z_score
            }
        norm_f = norm_f_dict[method]
        self._df = norm_f(self._df)
        return


class Dataset:
    def __init__(self):
        self._df = None
    def get_column_names(self):
        return self._df.columns
    def get_column(self, name):
        return self.df.iloc[name]

    def get_df(self):
        return self._df
    def get_by_feature(self, feature,set=None):
        df = self.get_df() if set is None else self._set_dict[set]
        return df.iloc[feature]

            
