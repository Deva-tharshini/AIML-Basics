# Session 12 – Bayesian Networks

## Objective

To build a Bayesian Network model and perform probabilistic inference using the Heart Disease dataset to analyze relationships between medical attributes.

## Dataset Used

heart_disease.csv

## Features

Features Used for Bayesian Inference:

* age
* sex
* cp (Chest Pain Type)
* trestbps (Resting Blood Pressure)
* chol (Cholesterol)
* fbs (Fasting Blood Sugar)
* restecg (Resting ECG Result)
* thalach (Maximum Heart Rate)
* oldpeak (ST Depression)

Target Variable:

* exang (Exercise Induced Angina)

## Operations Performed

* Imported dataset using Pandas
* Selected required columns
* Removed missing values
* Converted continuous values into categorical bins
* Built Bayesian Network structure
* Trained model using Maximum Likelihood Estimation
* Applied Variable Elimination for inference
* Generated probability predictions
* Created correlation heatmap
* Visualized target distribution

## Model Outputs

* Probability of Exercise Induced Angina
* Correlation Heatmap
* Distribution Visualization

Sample Output:

* exang(0) → 60.96%
* exang(1) → 39.04%

## Real-world Applications

* Medical diagnosis support
* Disease risk analysis
* Healthcare prediction systems
* Clinical decision support
* Probabilistic reasoning systems

## Output Files

* heatmap.png
* distribution.png
* console_output.png

## Learning Outcome

Learned how Bayesian Networks represent dependencies between variables, estimate probabilities, and perform inference using observed evidence.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* pgmpy