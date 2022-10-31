from datahandler import get_splits
from models import *
from utils import set_seed
from sklearn.metrics import f1_score
from tqdm import tqdm
seeds = range(0,10)
f1 = []
for seed in tqdm(seeds):
    X_train, X_test, y_train, y_test, df = get_splits("./csgo_clean.csv",None,"./features.txt")
    assert X_train.shape[0] + X_test.shape[0] == df.shape[0] and y_train.shape[0] + y_test.shape[0] == df.shape[0]
    set_seed(3)
    lgr = get_lgr()
    lgr = train(lgr,X_train,y_train)
    y_pred = lgr.predict(X_test)
    f1.append(f1_score(y_test,y_pred))
print(sum(f1)/len(f1))
    
