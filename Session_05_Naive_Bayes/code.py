# ==============================
# Session 05 – Naive Bayes
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------- Load Dataset ------------------

data = pd.read_csv("...........\\datasets\\iris.csv")
data = data.dropna()

# Convert species if needed

species_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

if data["Species"].dtype == "object":
    data["Species"] = data["Species"].map(species_map)

# ---------------- Features ------------------

X = data[["SepalLengthCm", "SepalWidthCm"]]
y = data["Species"]

# --------------- Split dataset -------------------

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------- Train model -------------------

model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# -------------- Confusion Matrix ------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True)

plt.title("Confusion Matrix")
plt.show()

# -------------- Decision Boundary -----------------

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1

y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train
)

plt.title("Decision Boundary")
plt.show()

print("\nConfusion Matrix:")
print(cm)