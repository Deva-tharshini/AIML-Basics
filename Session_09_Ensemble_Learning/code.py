# ===========================================
# Session 09 – Ensemble (Bagging & Boosting)
# ===========================================

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    BaggingRegressor,
    AdaBoostRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error
)
 
# ------------------ Load dataset ---------------
data = pd.read_csv("..............\\datasets\\titanic.csv")

# Select columns
data = data[["Survived","Pclass","Age","Sex","Fare"]]

# Convert categorical
data["Sex"] = data["Sex"].map({"male": 0,"female": 1})

# Remove missing values
data = data.dropna()

# ----------------- Classification -------------------

X = data[["Pclass","Age","Sex","Fare"]]

y = data["Survived"]

X_train, X_test, y_train, y_test = (
    train_test_split(X,y,test_size=0.2,random_state=42)
    )

# ------------------- Bagging -------------------------

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=20,
    random_state=42
)

bag_model.fit(X_train,y_train)

bag_pred = bag_model.predict(X_test)

print("Bagging Classification Accuracy:")

print(accuracy_score(y_test,bag_pred))

# ----------------- Boosting -------------------------

boost_model = AdaBoostClassifier(
    n_estimators=20,
    random_state=42
)

boost_model.fit(X_train,y_train)

boost_pred = boost_model.predict(X_test)

print("\nBoosting Classification Accuracy:")

print(accuracy_score(y_test,boost_pred))

# ----------------- Regression -------------------

X_reg = data[["Pclass","Age","Sex"]]

y_reg = data["Fare"]

Xr_train, Xr_test, yr_train, yr_test = (
    train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)
    )

# --------------- Bagging Regression ---------------

bag_reg = BaggingRegressor(n_estimators=20,random_state=4)

bag_reg.fit(Xr_train,yr_train)

bag_pred = bag_reg.predict(Xr_test)

print("\nBagging Regression MSE:")

print(mean_squared_error(yr_test,bag_pred))

# --------------- Boosting Regression ---------------

boost_reg = AdaBoostRegressor(n_estimators=20,random_state=42)

boost_reg.fit(Xr_train,yr_train)

boost_pred = boost_reg.predict(Xr_test)

print("\nBoosting Regression MSE:")

print(mean_squared_error(yr_test,boost_pred))