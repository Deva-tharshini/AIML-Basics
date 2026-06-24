# Session 10 – K-Means Clustering

## Objective

To group customers into clusters based on purchasing behavior using the K-Means clustering algorithm.

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
* Selected required features
* Converted categorical values into numeric values
* Removed missing values
* Standardized feature values
* Applied Elbow Method to determine optimal clusters
* Trained K-Means clustering model
* Assigned cluster labels
* Calculated Silhouette Score
* Visualized customer clusters

## Model Outputs

* Elbow Graph
* Cluster Visualization
* Silhouette Score

## Real-world Applications

* Customer segmentation
* Personalized marketing
* Product recommendation
* Retail analytics
* Business decision making

## Output Files

* elbow_graph.png
* clustering_graph.png
* console_output.png

## Learning Outcome

Learned how K-Means groups similar data points into clusters and how cluster evaluation methods help determine meaningful segmentation.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
