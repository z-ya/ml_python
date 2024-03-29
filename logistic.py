import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

credit_data = pd.read_csv("Datasets/credit_data.csv")

print(credit_data.head())
print(credit_data.describe())
print(credit_data.corr())

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(features_train, target_train)
predictions = model.fit.predict(features_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
print(model.fit.coef_)
print(model.fit.intercept_)

