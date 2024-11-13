# -*- coding: utf-8 -*-

"""
# Task 6

Dataset Used: Iris Dataset from scikit-learn

# Importing Libraries
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

"""# Loading Dataset"""

iris = load_iris()
X = iris.data  # Input features
y = iris.target  # Output labels

"""# Manually Extracting Features

Normalizing the data and extracting statistical features
"""

mean_features_repeated = np.repeat(mean_features.reshape(1, -1), X_scaled.shape[0], axis=0)
std_features_repeated = np.repeat(std_features.reshape(1, -1), X_scaled.shape[0], axis=0)

X_manual = np.hstack((X_scaled, mean_features_repeated, std_features_repeated))

X_train, X_test, y_train, y_test = train_test_split(X_manual, y, test_size=0.3, random_state=42)

"""# Classification

Logistic Regression classifier
"""

classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

accuracy_manual = accuracy_score(y_test, y_pred)
print(f'Accuracy with manual feature extraction: {accuracy_manual * 100}%')

"""Result: Hundred percent accuracy achieved by manual feature extraction."""
