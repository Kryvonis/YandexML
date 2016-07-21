from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# scaler = StandardScaler()
# X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
# X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([0, 1, 0])
# clf = Perceptron(random_state=241)
# clf.fit(X, y)
# predictions = clf.predict(X)
# print(predictions)



data_test = np.loadtxt('perceptron-test.csv', delimiter=",")
X_test, y_test = data_test[:, 1:3], data_test[:, 0]

# clf_test = Perceptron(random_state=241)
# clf_test.fit(X_test,y_test)

data_train = np.loadtxt('perceptron-train.csv', delimiter=",")
X_train, y_train = data_train[:, 1:3], data_train[:, 0]

clf_train = Perceptron(random_state=241)
clf_train.fit(X_train, y_train)
y_predict = clf_train.predict(X_test)
# clf_train.

accuracy = accuracy_score(y_test, y_predict)
print("without scale %s" % accuracy)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# clf_train = Perceptron(random_state=241)
clf_train.fit(X_train_scaled, y_train)

y_predict = clf_train.predict(X_test_scaled)
# clf_train.
accuracy_scale = accuracy_score(y_test, y_predict)
print("with scale %s" % accuracy_scale)


