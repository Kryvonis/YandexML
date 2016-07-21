import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import scale


def max_quality(X, y, kf):
    result_list = list()
    for k in range(1, 51):
        kNN = KNeighborsClassifier(n_neighbors=k)
        kNN.fit(X, y)
        find_quality = cross_val_score(estimator=kNN, X=X, y=y, cv=kf, scoring='accuracy')
        m = find_quality.mean()
        result_list.append(m)

    print("%0.2f" % (max(result_list)))
    print(result_list.index(max(result_list)) + 1)


data = np.loadtxt('wine.data', delimiter=",")
X, y = data[:, 1:14], data[:, 0]

kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)
max_quality(X, y, kf)

scale_X = scale(X)
max_quality(scale_X, y, kf)
