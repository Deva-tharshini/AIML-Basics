# ===================================
#  Session 12 – Bayesian Networks
# ===================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ------------------------- Load Dataset -------------------------

data = pd.read_csv("..............\\datasets\\heart_disease.csv")

# Select required columns
data = data[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak"]]

# Remove missing values
data = data.dropna()

# ------------------------- Convert continuous columns -------------------------

continuous = ["age","trestbps","chol","thalach","oldpeak"]

for col in continuous:

    data[col] = pd.cut(
        data[col],
        bins=3,
        labels=[0, 1, 2]
    )

# Convert all to integer
data = data.astype(int)

# ------------------------- Create Bayesian Network -------------------------

model = BayesianNetwork([
    ("age", "exang"),
    ("sex", "exang"),
    ("cp", "exang"),
    ("trestbps", "exang"),
    ("chol", "exang"),
    ("fbs", "exang"),
    ("restecg", "exang"),
    ("thalach", "exang"),
    ("oldpeak", "exang")
])

# ------------------------- Train Model -------------------------

model.fit(data,estimator=MaximumLikelihoodEstimator)

print("Model Valid:",model.check_model())

# ------------------------- Inference -------------------------

infer = VariableElimination(model)

result = infer.query(variables=["exang"],evidence={"sex": 1,"cp": 1})

print("\nProbability of Exercise Induced Angina:\n")

print(result)

# ------------------------- Correlation Heatmap -------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(data.corr(),annot=True,cmap="coolwarm")

plt.title("Heart Disease Correlation")

plt.show()

# ------------------------- Distribution Plot -------------------------

plt.figure(figsize=(6, 5))

data["exang"].value_counts().plot(kind="bar")

plt.title("Exercise Induced Angina Distribution")

plt.xlabel("Exang")

plt.ylabel("Count")

plt.show()