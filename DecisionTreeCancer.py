import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import datasets

cancer_data = datasets.load_breast_cancer()

#print(cancer_data.data)

features = cancer_data.data
labels = cancer_data.target

#print(features.shape)
#print(lables.shape)

features_train, features_test, target_train, target_test = train_test_split(features,labels, test_size=0.3)

model = DecisionTreeClassifier(criterion= 'entropy', max_depth=3)

predicted = cross_validate(model, features, labels, cv = 10)
print(np.mean(predicted['test_score']))
