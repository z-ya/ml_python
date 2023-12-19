import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("Datasets/credit_data.csv")

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

X = np.array(features).reshape(-1,3)
y = np.array(target).reshape(-1,1)

model = LogisticRegression()
predicted = cross_validate(model, X, y,cv=5)

print(predicted["test_score"])

