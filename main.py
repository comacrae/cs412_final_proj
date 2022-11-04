from datahandler import get_splits
from models import *
from utils import set_seed
from sklearn.metrics import f1_score
from tqdm import tqdm
seeds = [0]#range(0,10)
features = ['all','engineered','economy','weapons','players']
for feature in features:
    f1 = []
    for seed in tqdm(seeds):
        X_train, X_test, y_train, y_test, df = get_splits("./csgo_clean.csv",None,f"./{feature}_features.txt",max_abs=True)
        assert X_train.shape[0] + X_test.shape[0] == df.shape[0] and y_train.shape[0] + y_test.shape[0] == df.shape[0]
        set_seed(seed)
        lgr = get_lgr()
        lgr = train(lgr,X_train,y_train)
        y_pred = lgr.predict(X_test)
        f1.append(f1_score(y_test,y_pred))
    print(f'f1 for {feature} features: {sum(f1)/len(f1)} seed={seed}')
    
