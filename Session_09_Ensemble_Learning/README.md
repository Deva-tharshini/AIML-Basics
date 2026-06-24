# Session 09 – Ensemble (Bagging & Boosting)

## Objective

To improve prediction performance by combining multiple models using Ensemble Learning techniques for both classification and regression.

## Dataset Used

titanic.csv

## Classification Task

Target Variable:

* Survived

Features:

* Pclass
* Age
* Sex
* Fare

Models:

* Bagging Classifier
* AdaBoost Classifier

## Regression Task

Target Variable:

* Fare

Features:

* Pclass
* Age
* Sex

Models:

* Bagging Regressor
* AdaBoost Regressor

## Operations Performed

* Imported dataset using Pandas
* Selected required features
* Converted categorical values into numeric values
* Removed missing values
* Split dataset into training and testing sets
* Trained Bagging Classification model
* Trained Boosting Classification model
* Trained Bagging Regression model
* Trained Boosting Regression model
* Compared performance metrics

## Model Outputs

Classification:

* Bagging Classification Accuracy
* Boosting Classification Accuracy

Regression:

* Bagging Regression Mean Squared Error (MSE)
* Boosting Regression Mean Squared Error (MSE)

## Real-world Applications

* Fraud detection
* Credit risk prediction
* Customer behavior analysis
* Recommendation systems
* Business forecasting

## Output Files

* console_output.png

## Learning Outcome

Learned how Ensemble Learning combines multiple models to improve stability, reduce overfitting, and increase predictive performance for both classification and regression tasks.

## Technologies Used

* Python
* Pandas
* Scikit-learn
