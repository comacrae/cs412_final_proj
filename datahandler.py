import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,StandardScaler

def load_df(path):
    with open(path, "r") as f:
        return pd.read_csv(f)

def get_features(df,features):
    return df[features]

def load_feature_list(path):
    with open(path,"r") as f:
        return f.read().splitlines()

def scale(X_train,X_test,use_mas=False):
    scaler = MaxAbsScaler() if use_mas else StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def get_splits(df_path, features=None,features_path=None,split=[0.8,0.2],max_abs=False):
    df = load_df(df_path)
    if features_path is not None:
        features = load_feature_list(features_path)
    if features is not None:
        df = get_features(df, features)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split[0],train_size=split[1])
    X_train, X_test = scale(X_train, X_test)
    return X_train, X_test, y_train, y_test,df

