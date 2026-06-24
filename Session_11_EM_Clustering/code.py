# ==============================
# Session 11 – EM Algorithm
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# --------------- Load dataset -----------------

data = pd.read_csv(".............\\datasets\\mall_customers.csv")

# Select columns
data = data[["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]

# Convert categorical
data["Gender"] = data["Gender"].map({"Male": 0,"Female": 1})

# Remove missing values
data = data.dropna()

# Features
X = data[["Annual Income (k$)","Spending Score (1-100)"]]

# Scale
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------- EM Clustering ----------------

model = GaussianMixture(n_components=5,random_state=42)

clusters = model.fit_predict(X_scaled)

# Score
score = silhouette_score(X_scaled,clusters)

print("Silhouette Score:")

print(score)

# Save cluster labels
data["Cluster"] = clusters

# ----------------- Correlation Heatmap -----------------

plt.figure(figsize=(8, 6))

sns.heatmap(data.corr(),annot=True,cmap="coolwarm")

plt.title("Feature Correlation")

plt.show()

# ------------------- Cluster Graph -------------------

plt.figure(figsize=(8, 6))

plt.scatter(X_scaled[:, 0],X_scaled[:, 1],c=clusters)

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.title("EM Clustering")

plt.show()