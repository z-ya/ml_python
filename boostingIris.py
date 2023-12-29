from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()

features = iris_data.data
target = iris_data.target

features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)
model_fitted = model.fit(features_train, target_train)
predictions = model_fitted.predict(features_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
