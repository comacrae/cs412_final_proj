
def std_dev(X):
    n = len(X)
    m = mean(X)
    diff_sq = [ (x - m)**2 for x in X]
    return (sum(diff_sq) / n) ** 0.5
def mean(X):
    n = len(X)
    return sum(X) / n
def z_score(x, u, o):
    return (x - u)/ o
def sample_z_score(X):
    u = mean(X)
    o = std_dev(X)
    return [((x - u) / o) for x in X ]
def sample_min_max(X):
    min = min(X)
    max = max(X)
    return [((x - min)/(max - min)) for x in X]
def min_max(x, min, max):
    return (x - min)/(max - min)
def abs_max(x, max):
    return x/max
def sample_abs_max(X):
    max = abs(max(X)) 
    return [(x/max) for x in X]
def df_z_score(df):
    return (df - df.mean(numeric_only=True) )/ df.std(numeric_only=True)