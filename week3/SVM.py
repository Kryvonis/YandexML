from sklearn import svm
import numpy as np

data_test = np.loadtxt('svm-data.csv', delimiter=",")
X, y = data_test[:, 1:3], data_test[:, 0]
clf = svm.SVC(C=100000,random_state=241,kernel='linear')
clf.fit(X,y)

print(clf.support_)