import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
a = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")

print(a.head())
print(a.isnull().sum())

# 2. Remove missing values
a = a[["displacement", "mpg"]].dropna()

x = a[["displacement"]]
y = a["mpg"]

print(x.head())
print(y.head())

# 3. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 4. Linear Regression
m1 = LinearRegression()
m1.fit(x_train, y_train)
p1 = m1.predict(x_test)

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, p1))
print("R2:", r2_score(y_test, p1))

# 5. Polynomial Regression
for d in range(2, 6):
    poly = PolynomialFeatures(degree=d)

    x_train1 = poly.fit_transform(x_train)
    x_test1 = poly.transform(x_test)

    m2 = LinearRegression()
    m2.fit(x_train1, y_train)
    p2 = m2.predict(x_test1)

    print("\nDegree:", d)
    print("MSE:", mean_squared_error(y_test, p2))
    print("R2:", r2_score(y_test, p2))




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
a = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")

# 2. Remove missing values
a = a[["displacement", "mpg"]].dropna()

x = a[["displacement"]]
y = a["mpg"]

# 3. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 4. Linear Regression
m1 = LinearRegression()
m1.fit(x_train, y_train)
p1 = m1.predict(x_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, p1))
print("R2:", r2_score(y_test, p1))

# 5. Polynomial Regression - Degree 2
poly2 = PolynomialFeatures(degree=2)
x_train2 = poly2.fit_transform(x_train)
x_test2 = poly2.transform(x_test)

m2 = LinearRegression()
m2.fit(x_train2, y_train)
p2 = m2.predict(x_test2)

print("\nPolynomial Regression Degree 2")
print("MSE:", mean_squared_error(y_test, p2))
print("R2:", r2_score(y_test, p2))

# 6. Polynomial Regression - Degree 3
poly3 = PolynomialFeatures(degree=3)
x_train3 = poly3.fit_transform(x_train)
x_test3 = poly3.transform(x_test)

m3 = LinearRegression()
m3.fit(x_train3, y_train)
p3 = m3.predict(x_test3)

print("\nPolynomial Regression Degree 3")
print("MSE:", mean_squared_error(y_test, p3))
print("R2:", r2_score(y_test, p3))

# 7. Polynomial Regression - Degree 4
poly4 = PolynomialFeatures(degree=4)
x_train4 = poly4.fit_transform(x_train)
x_test4 = poly4.transform(x_test)

m4 = LinearRegression()
m4.fit(x_train4, y_train)
p4 = m4.predict(x_test4)

print("\nPolynomial Regression Degree 4")
print("MSE:", mean_squared_error(y_test, p4))
print("R2:", r2_score(y_test, p4))

# 8. Polynomial Regression - Degree 5
poly5 = PolynomialFeatures(degree=5)
x_train5 = poly5.fit_transform(x_train)
x_test5 = poly5.transform(x_test)

m5 = LinearRegression()
m5.fit(x_train5, y_train)
p5 = m5.predict(x_test5)

print("\nPolynomial Regression Degree 5")
print("MSE:", mean_squared_error(y_test, p5))
print("R2:", r2_score(y_test, p5))




