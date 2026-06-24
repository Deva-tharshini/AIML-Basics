# Session 11 – EM Algorithm

## Objective

To perform customer segmentation using the Expectation Maximization (EM) Algorithm and group customers based on purchasing behavior.

## Dataset Used

mall_customers.csv

## Features

Features Used for Clustering:

* Annual Income (k$)
* Spending Score (1–100)

Additional Dataset Columns:

* CustomerID
* Gender
* Age

## Operations Performed

* Imported dataset using Pandas
* Selected required columns
* Converted categorical values into numeric values
* Removed missing values
* Standardized feature values
* Applied Expectation Maximization using Gaussian Mixture Model (GMM)
* Assigned cluster labels
* Calculated Silhouette Score
* Generated Correlation Heatmap
* Visualized customer clusters

## Model Outputs

* Silhouette Score
* Correlation Heatmap
* Cluster Visualization

## Real-world Applications

* Customer segmentation
* Personalized marketing
* Customer behavior analysis
* Product recommendation
* Retail analytics
* Business decision making

## Output Files

* heatmap.png
* clustering_graph.png
* console_output.png

## Learning Outcome

Learned how the EM Algorithm groups data probabilistically using Gaussian distributions and how clustering helps discover patterns in unlabeled data.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn