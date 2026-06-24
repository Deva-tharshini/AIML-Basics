# Session 06 – Support Vector Machine (SVM)

## Objective

To build a Support Vector Machine (SVM) classification model and identify the best-performing kernel for classifying Iris flower species.

## Dataset Used

iris.csv

## Features

Independent Variables (X):

* sepal_length
* sepal_width

Dependent Variable (y):

* species

## Operations Performed

* Imported dataset using Pandas
* Removed missing values
* Converted categorical values into numeric values
* Split dataset into training and testing sets
* Standardized feature values
* Trained SVM using multiple kernels:

  * Linear
  * Polynomial (Poly)
  * RBF
  * Sigmoid
* Compared kernel accuracy
* Selected best-performing kernel
* Generated predictions
* Calculated model accuracy
* Generated Confusion Matrix
* Visualized Decision Boundary

## Model Outputs

* Accuracy for each kernel
* Best Kernel Selection
* Confusion Matrix
* Decision Boundary Visualization

## Real-world Applications

* Image classification
* Face recognition
* Medical diagnosis
* Bioinformatics
* Text classification

## Output Files

* confusion_matrix.png
* decision_boundary.png
* console_output.png

## Learning Outcome

Learned how SVM separates classes using decision boundaries and how kernel selection affects classification performance.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
