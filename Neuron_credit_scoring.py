from keras import Sequential
from keras import layers
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

credit_data = pd.read_csv("Datasets/credit_data.csv")
features = credit_data[["income", "age", "loan"]]
y = np.array(credit_data.default).reshape(-1,1)
#print(y)

encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(layers.Dense(10, input_dim=3, activation="sigmoid"))
model.add(layers.Dense(2, activation="softmax"))

optimizer = optimizers.Adam(learning_rate = 0.001)

model.compile(loss = "categorical_crossentropy",
              optimizer = optimizer,
              metrics = ["accuracy"])

model.fit(train_features, train_targets, epochs = 1000, verbose = 2)
results = model.evaluate(test_features, test_targets)

print("results")
print(results)
