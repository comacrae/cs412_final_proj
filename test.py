import pandas as pd
from sklearn.neural_network import MLPClassifier
from dataloader import DataLoader
from sklearn.neural_network._multilayer_perceptron import *
from sklearn.metrics import f1_score


def get_features(df,feature_names):
    features = pd.DataFrame()
    for name in feature_names:
            col = df.loc(name)
            n = features.shape[1]
            features.insert(n,name,col)
    return features
dl = DataLoader()
dl.load_df_from_path('csgo_clean.csv')
df = dl.get_full_set()
features = get_features(df,["round"])
print(features)


def split(X, y, train=0.8):
    nX = X.shape[0]
    nY = y.shape[0]
    train_n = int(0.8 *  n)
    X = X.values.tolist()
    y = np.ravel(y.values)
    trainX = X[:train_n]
    trainy = y[:train_n]
    testX = X[train_n:]
    testy = y[train_n:]
    return trainX, trainy, testX, testy
trainX, trainy, testX, testy = split(features,labels)
print(testy.shape)

#train,val,test= dl.get_split_sets()
mlp = MLPClassifier()
mlp.fit(trainX, trainy)
predy = mlp.predict(testX)
print(f1_score(testy, predy))
