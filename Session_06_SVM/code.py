# ============================================
# Session 06 – Support Vector Machine (SVM)
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------- Load dataset -----------------

data = pd.read_csv("..............\\datasets\\iris.csv")
data = data.dropna()

# Convert species if needed
species_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

if data["Species"].dtype == "object":
    data["Species"] = data["Species"].map(species_map)

# Use 2 features
X = data[["SepalLengthCm", "SepalWidthCm"]]
y = data["Species"]

# ------------------- Split --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Train multiple kernels -------------------

kernels = ["linear", "poly", "rbf", "sigmoid"]

accuracies = {}

for k in kernels:
    model = SVC(kernel=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracies[k] = accuracy_score(
        y_test,
        y_pred
    )

    print(f"{k} Accuracy:", accuracies[k])

# Best kernel
best_kernel = max(
    accuracies,
    key=accuracies.get
)

print("\nBest Kernel:", best_kernel)

# ---------------- Final model -----------------------

model = SVC(kernel=best_kernel)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# ----------------- Confusion Matrix --------------------

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(cm,annot=True)

plt.title("Confusion Matrix")

plt.show()

# --------------- Decision Boundary ---------------------

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1

y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,alpha=0.3)

plt.scatter(X_train[:, 0],X_train[:, 1],c=y_train)

plt.title(f"SVM Decision Boundary ({best_kernel})")

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.show()

print("\nBest Accuracy:")
print(accuracies[best_kernel])