# ==============================================
# Session 01 – Exploratory Data Analysis (EDA)
# ==============================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Load Dataset ---------------

df = pd.read_csv("........\\datasets\\iris.csv")

#----------- Display Dataset -------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ---------- Visualization 1: Histogram --------------

df["SepalLengthCm"].hist()

plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")

plt.show()

# ---------- Visualization 2: Scatter Plot ----------------

sns.scatterplot(
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species",
    data=df
)

plt.title("Sepal Length vs Petal Length")

plt.show()

# --------------- Visualization 3: Heatmap --------------

plt.figure(figsize=(8, 6))

numeric_df = df.drop(columns=["Id"]).select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# -------------- Visualization 4: Boxplot -----------------

plt.figure(figsize=(10, 6))

sns.boxplot(data=df.drop(columns=["Id"]))

plt.title("Boxplot for Feature Distribution")

plt.show()

print("\nEDA Completed Successfully")