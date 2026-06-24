# ==============================
# Session 02 – Linear Regression
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------- Load Dataset ------------------

data = pd.read_csv(".......\\datasets\\salary.csv")

# Remove missing values
data = data.dropna()

# Features and Target
X = data[["YearsExperience"]]
y = data["Salary"]

# -------------- Train Model --------------------

model = LinearRegression()

model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# -------------- Visualization ------------------

plt.figure(figsize=(8,6))

plt.scatter(X, y)

plt.plot(X, y_pred)

plt.title("Salary Prediction using Linear Regression")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()

# --------------- Metrics ------------------

score = r2_score(y, y_pred)

print("\nCoefficient (Slope):")
print(model.coef_[0])

print("\nIntercept:")
print(model.intercept_)

print("\nR² Score:")
print(score)

print("\nSample Predictions:")

print(pd.DataFrame({
    "Actual": y.head(),
    "Predicted": y_pred[:5]
}))