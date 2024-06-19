from keras import Sequential
from keras import layers
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris_data = load_iris()
features = iris_data.data
labels= iris_data.target.reshape(-1,1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(layers.Dense(10, input_dim=4, activation="sigmoid"))
model.add(layers.Dense(3, activation="softmax"))

optimizer = optimizers.Adam(learning_rate = 0.001)
model.compile(loss = "categorical_crossentropy",
              optimizer = optimizer,
              metrics = ["accuracy"])

model.fit(train_features, train_targets, epochs = 1000, batch_size = 20, verbose = 2)
results = model.evaluate(test_features, test_targets)

print("results")
print(model.predict(test_features))
print(results)



