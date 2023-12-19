import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

iris_data = datasets.load_iris()

features = iris_data.data
target = iris_data.target

param_grid = {'max_depth' : np.arange(1,10)}

features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.2)

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(features_train, target_train)
print('Best parametr with GridSearch: ', tree.best_params_)

grid_predictions = tree.predict(features_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))


#model = DecisionTreeClassifier(criterion='entropy')
#predicted = cross_validate(model, features, target, cv = 10)

#print(np.mean(predicted['test_score']))




