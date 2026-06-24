# ===============================================
# Session 08 – Random Forest
# ================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ------------------ Load dataset ------------------

data = pd.read_csv("............\\datasets\\titanic.csv")

# Select columns
data = data[["Survived","Pclass","Age","Sex","Fare"]]

# Convert categorical values
data["Sex"] = data["Sex"].map({"male": 0,"female": 1})

# Remove missing values
data = data.dropna()

# Features and target
X = data[["Pclass","Age","Sex","Fare"]]

y = data["Survived"]

# ----------------- Split -----------------

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# ----------------- Train model -----------------

model = RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:", accuracy)

# ----------------- Confusion Matrix -----------------

cm = confusion_matrix(y_test,y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# ----------------- Classification Report -----------------

print("\nClassification Report:\n")

print(classification_report(y_test,y_pred))

# ----------------- Heatmap -----------------

plt.figure(figsize=(6, 5))

sns.heatmap(cm,annot=True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()