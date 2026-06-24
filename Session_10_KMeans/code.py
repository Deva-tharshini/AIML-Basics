# =================================
# Session 10 – K-Means Clustering
# =================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------- Load dataset --------------------

data = pd.read_csv("...........\\datasets\\mall_customers.csv")

# Select columns
data = data[["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]]

# Convert categorical
data["Gender"] = data["Gender"].map({"Male": 0,"Female": 1})

# Remove missing values
data = data.dropna()

# Features for clustering
X = data[["Annual Income (k$)","Spending Score (1-100)"]]

# Scale
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ------------------- Elbow Method -------------------

inertia = []

K_range = range(1, 11)

for k in K_range:

    model = KMeans(n_clusters=k,random_state=42,n_init=10)

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(K_range,inertia,marker="o")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()

# ------------------ KMeans ----------------
 
kmeans = KMeans(n_clusters=5,random_state=42,n_init=10)

clusters = kmeans.fit_predict(X_scaled)

# Score
score = silhouette_score(X_scaled,clusters)

print("Silhouette Score:")

print(score)

# ------------------ Cluster Graph ---------------

plt.figure(figsize=(8, 6))

plt.scatter(X_scaled[:, 0],X_scaled[:, 1],c=clusters)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",s=200
)

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.title("K-Means Clustering")

plt.show()