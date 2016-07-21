import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import scale
import sklearn.datasets as boston

data = boston.load_boston()
# data = np.loadtxt('wine.data', delimiter=",")

X, y = data.data, data.target
kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)

X = scale(X)

result_list = list()
array_tipa = np.linspace(1., 10., 200)
for k in array_tipa:
    kNN = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=k)
    kNN.fit(X, y)
    find_quality = cross_val_score(estimator=kNN, X=X, y=y, cv=kf, scoring='mean_squared_error')
    m = find_quality.mean()
    result_list.append(m)

print("%0.2f" % (max(result_list)))
print(result_list.index(max(result_list)) + 1)

# max_quality(scale_X, y, kf)
