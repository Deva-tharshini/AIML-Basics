# =============================
# Session 07 – Decision Tree
# =============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ------------------- Load dataset  ------------------------

data = pd.read_csv("...............\\datasets\\titanic.csv")

# Select required columns
data = data[["Survived","Pclass","Age","Sex","Fare"]]

# Convert categorical values
data["Sex"] = data["Sex"].map({"male": 0,"female": 1})

# Remove missing values
data = data.dropna()

# Features and target
X = data[["Pclass","Age","Sex","Fare"]]

y = data["Survived"]

# Split
X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2,random_state=42))

# ------------------- Train model  ------------------

model = DecisionTreeClassifier(max_depth=4,random_state=42)

model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:", accuracy)

# ---------------- Classification Report  ------------------

print("\nClassification Report:\n")

print(classification_report(y_test,y_pred))

# --------------- Confusion Matrix ---------------------

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm,annot=True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ---------------- Tree Visualization ------------------

plt.figure(figsize=(14,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Survived","Survived"],filled=True
    )

plt.title("Decision Tree")

plt.show()