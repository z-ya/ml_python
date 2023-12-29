import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def is_tasty(quality):
    if quality>=7:
        return 1
    else:
        return 0

data = pd.read_csv("Datasets/wine.csv", sep=";")
data['tasty'] = data["quality"].apply(is_tasty)
features = data.iloc[:,0:11]
targets = data["tasty"]

X = np.array(features).reshape(-1,11)
y = np.array(targets)

X = preprocessing.MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

param_dist = {
    'n_estimators':[10,50,200],
    'learning_rate':[0.01, 0.05,0.3, 1]
}

grid_search = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_dist, cv=10)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

