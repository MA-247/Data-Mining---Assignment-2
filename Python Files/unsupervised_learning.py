# -*- coding: utf-8 -*-
"""Unsupervised Learning.ipynb



# Task 1 - 4

# Importing Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

"""# Loading Dataset"""

data = pd.read_csv('Mall_Customers.csv')

print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nSummary Statistics:\n", data.describe())

"""#Pre-processing

Missing Values
"""

# Checking for missing values
print("\nMissing Values:\n", data.isnull().sum())

"""Dropping unnecessary features"""

# Dropping 'CustomerID' column as it is not needed for analysis or model training
data.drop(columns=['CustomerID'], inplace=True)

"""Label encoding"""

#Fixing misspelling
data.rename(columns={'Genre': 'Gender'}, inplace=True)

# Label Encoding
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

"""Feature Scaling"""

# Feature Scaling
scaler = StandardScaler()
data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(
    data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

print("\nPreprocessed Data Head:\n", data.head())

"""# Unsupervised Learning"""

X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

"""K Mean Clustering"""

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X, data['Cluster'])
print(f'Silhouette Score for K-Means: {silhouette_avg:.2f}')

print("\nData with Cluster Labels:\n", data.head())

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram for Optimal Clusters')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Applying Agglomerative Clustering with the optimal number of clusters (e.g., 5 from dendrogram)
agg_cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
data['Agglomerative_Cluster'] = agg_cluster.fit_predict(X)

# Calculate Silhouette Score for Agglomerative Clustering
silhouette_avg_agg = silhouette_score(X, data['Agglomerative_Cluster'])
print(f'Silhouette Score for Agglomerative Clustering: {silhouette_avg_agg:.2f}')

# Displaying the first few rows with the new cluster labels
print("\nData with Agglomerative Cluster Labels:\n", data.head())

"""# Silhouette Score

K Mean: 0.36
Agglomerative Clustering: 0.44

# Feature Selection

Using Recursive Feature Elimination (RFE)
"""

X = data.drop(columns=['Cluster'])
y = data['Cluster']

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
X_selected = rfe.fit_transform(X, y)

print("Selected Features (Based on RFE):", rfe.support_)
print("Feature Ranking:", rfe.ranking_)

"""New CSV file with the selected features only"""

selected_features = X.columns[rfe.support_]

data_selected = data[selected_features]

data_selected.to_csv('selected_features.csv', index=False)

print("New CSV file with selected features saved as 'selected_features.csv'.")

"""# Supervised Leanring

Before Moving to Supervised Learning, we have to add a target class for the classifiers to predict as the orignal dataset does not have such feature, we will add it using clustering.
"""

data = pd.read_csv('selected_features.csv')

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

agg_cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
data['Agglomerative_Cluster'] = agg_cluster.fit_predict(X)

print(data.head())

"""Now, we have "Customer Segmeent" as our target for classification"""
