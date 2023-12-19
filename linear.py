import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

house_data = pd.read_csv("Datasets/house_prices.csv")
print(house_data)
size = house_data["sqft_living"]
price = house_data["price"]
print(size)
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)
print(x)

model = LinearRegression()
model.fit(x,y)

regression_model_mse = mean_squared_error(x,y)

print("mse: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x,y))

print(model.coef_[0])
print(model.intercept_[0])

plt.scatter(x,y, color = "green")
plt.plot(x, model.predict(x), color= "black")
plt.title("Liner regression")
plt.xlabel("size")
plt.ylabel("price")
plt.show()

print("Predictions by the model: ", model.predict([[2000]]))


