# =================================
# Session 04 – Logistic Regression
# =================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------- Load Dataset ---------------------

data = pd.read_csv("........\\datasets\\breast_cancer.csv")

# Remove unwanted column
data = data.drop(columns=["Unnamed: 32"],errors="ignore")

# Convert target
data["diagnosis"] = (data["diagnosis"].map({"M": 1,"B": 0}))

# Remove missing values
data = data.dropna()

# ------------------ Features and Target -----------------

X = data[["radius_mean"]]

y = data["diagnosis"]

# -------------------- Train Model -----------------

model = LogisticRegression()

model.fit(X,y)

# Prediction
y_pred = model.predict(X)

# Accuracy
score = accuracy_score(y,y_pred)

# ---------------- Sigmoid Curve --------------------

X_values = np.linspace(X.min(),X.max(),300).reshape(-1, 1)

y_prob = (model.predict_proba(X_values))[:, 1]

# ----------------- Visualization --------------------

plt.figure(figsize=(8,6))

plt.scatter(X,y)

plt.plot(X_values,y_prob)

plt.title("Logistic Regression")

plt.xlabel("Radius Mean")

plt.ylabel("Diagnosis")

plt.show()

# ------------------------------
# Results
# ------------------------------

print("\nCoefficient:")

print(model.coef_[0][0])

print("\nIntercept:")

print(model.intercept_[0])

print("\nAccuracy:")

print(score)

print("\nSample Predictions:")

print(pd.DataFrame({"Actual": y.head(),"Predicted": y_pred[:5]}))