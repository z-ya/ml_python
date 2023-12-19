import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

credit_data = pd.read_csv("Datasets/credit_data.csv")

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

X = np.array(features).reshape(-1,3)
y = np.array(target)

X = preprocessing.MinMaxScaler().fit_transform(X)
features_train, features_test, target_train, target_test = train_test_split(X,y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=32)
fitted_model = model.fit(features_train, target_train)
predictions = fitted_model.predict(features_test)

cross_valid_scores = []
for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring = "accuracy")
    cross_valid_scores.append(scores.mean())
print('Optimal k: ', np.argmax(cross_valid_scores))


print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))


