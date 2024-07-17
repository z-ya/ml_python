import numpy as np
from keras import Sequential
from keras import layers

training_data = np.array([[0, 0], [0, 1], [1,0], [1,1]], "float32")
target_data = np.array([0][1][1][0],"float32")

model = Sequential()
model.add(layers.Dense(16, input_dim=2, activation="relu"))
model.add(layers.Dense(16, input_dim=16, activation="relu"))
model.add(layers.Dense(16, input_dim=16, activation="relu"))
model.add(layers.Dense(16, input_dim=16, activation="relu"))
model.add(layers.Dense(16, input_dim=16, activation="relu"))
model.add(layers.Dense(16, input_dim=16, activation="relu"))
model.add(layers.Dense(16, input_dim=16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss = "mean_squared_error",
              optimizer = "adam",
              metrics = ["binary_accuracy"])
