import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

# Select one feature
X = df["AveRooms"].values
y = df["Price"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Gradient Descent
# -------------------------
w = 0
b = 0
lr = 0.01
epochs = 1000
cost = []
n = len(X_train)
for i in range(epochs):
    y_pred = w * X_train + b
    error = y_pred - y_train
    c = np.mean(error ** 2) / 2
    cost.append(c)
    dw = np.mean(error * X_train)
    db = np.mean(error)
    w = w - lr * dw
    b = b - lr * db

    if i % 100 == 0:
        print("Epoch:", i, "Cost:", c)
# Gradient Descent prediction
y_gd = w * X_test + b

# Gradient Descent results
print()
print("Gradient Descent")
print("----------------")
print("Weight:", w)
print("Bias:", b)
print("MSE:", mean_squared_error(y_test, y_gd))
print("R2 Score:", r2_score(y_test, y_gd))


# -------------------------
# Normal Equation
# -------------------------

x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

w2 = np.sum(
    (X_train - x_mean) * (y_train - y_mean)
) / np.sum(
    (X_train - x_mean) ** 2
)

b2 = y_mean - w2 * x_mean


# Normal Equation prediction

y_ne = w2 * X_test + b2


# Normal Equation results

print()
print("Normal Equation")
print("---------------")
print("Weight:", w2)
print("Bias:", b2)
print("MSE:", mean_squared_error(y_test, y_ne))
print("R2 Score:", r2_score(y_test, y_ne))


# -------------------------
# Cost Convergence Graph
# -------------------------

plt.plot(cost)

plt.title("Gradient Descent Cost Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.grid()
plt.show()


# -------------------------
# Regression Line
# -------------------------

x_line = np.sort(X_test)
y_line = w2 * x_line + b2

plt.scatter(X_test, y_test, label="Actual Data")

plt.plot(x_line, y_line, label="Regression Line")

plt.title("Linear Regression")
plt.xlabel("Average Rooms")
plt.ylabel("House Price")

plt.legend()
plt.grid()

plt.show()
