from keras import Sequential
from keras import layers
import numpy as np

x = np.array([[0, 0], [0, 1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])
print(x.shape)

model = Sequential()
model.add(layers.Dense(4, input_dim=2, activation="sigmoid"))
model.add(layers.Dense(1, input_dim=4, activation="sigmoid"))
print(model.weights)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["binary_accuracy"])
model.fit(x, y, epochs = 10000, verbose = 2)
print("Predictions after the training ")
print(model.predict(x))