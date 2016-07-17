import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

sex_key = {"female": 0, "male": 1}

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

num_sex = {"female": 0, "male": 1}
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
training_data = data.drop(data.columns[[2, 5, 6, 7, 9, 10]], axis=1, )
training_data = training_data.dropna()
training_data['Sex'] = training_data['Sex'].apply(num_sex.get).astype(bool)

classifier = training_data.drop(training_data.columns[[1, 4, 3, 2]], axis=1, )


training_data = training_data.drop(training_data.columns[[0]], axis=1, )


print(len(classifier))
print(len(training_data))
print(classifier.head())
clf = DecisionTreeClassifier(random_state=241)
clf.fit(training_data, classifier)

importances = clf.feature_importances_

print(training_data.head())
print(classifier.head())
print(importances)
