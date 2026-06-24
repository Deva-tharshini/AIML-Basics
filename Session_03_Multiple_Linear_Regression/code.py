# ========================================
# Session 03 – Multiple Linear Regression
# ========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --------------- Load Dataset ------------------

data = pd.read_csv("......\\datasets\\startups.csv")

# Remove missing values
data = data.dropna()

# -------------- Features and Target -----------------

X = data[
    [
        "R&D Spend",
        "Administration",
        "Marketing Spend"
    ]
]

y = data["Profit"]

# ---------------- Train Model -------------------

model = LinearRegression()

model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# ------------------------------
# Visualization
# Projection using R&D Spend
# ------------------------------

plt.figure(figsize=(8,6))

plt.scatter(data["R&D Spend"],y)

plt.plot(data["R&D Spend"],y_pred)

plt.title("Multiple Linear Regression")

plt.xlabel("R&D Spend")

plt.ylabel("Profit")

plt.show()

# ---------------- Metrics -----------------

score = r2_score(y,y_pred)

print("\nCoefficients:")

print(model.coef_)

print("\nIntercept:")

print(model.intercept_)

print("\nR² Score:")

print(score)

print("\nSample Predictions:")

print(
    pd.DataFrame({
        "Actual": y.head(),
        "Predicted": y_pred[:5]
    })
)