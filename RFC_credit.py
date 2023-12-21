import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("Datasets/credit_data.csv")

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

X = np.array(features).reshape(-1,3)
y = np.array(target)

model = RandomForestClassifier()
predicted = cross_validate(model, X, y, cv=10)

print(np.mean(predicted['test_score']))