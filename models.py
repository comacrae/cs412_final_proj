from sklearn.linear_model import LogisticRegressionCV as LR

def train(m,X,y):
    return m.fit(X,y)

def get_lgr(cv=10):
    return LR(cv=cv)