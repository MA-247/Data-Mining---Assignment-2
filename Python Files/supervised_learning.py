# -*- coding: utf-8 -*-
"""Supervised Learning.ipynb


# Task 5
"""

from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

"""# Supervised Leanring

Before Moving to Supervised Learning, we have to add a target class for the classifiers to predict as the orignal dataset does not have such feature, we will add it using clustering.
"""

data = pd.read_csv('selected_features.csv')

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

agg_cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
data['CustomerSegment'] = agg_cluster.fit_predict(X)

print(data.head())

"""Now, we have "Customer Segment" as our target for classification"""

# Assume 'CustomerSegment' is the target column
X = data.drop(columns=['CustomerSegment'])
y = data['CustomerSegment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

"""Random Forest Classifier"""

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""SVM"""

# 2. Support Vector Machine (SVM)
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM - Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

"""KNN"""

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN - Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

"""Gradient Boosting"""

grad_boost = GradientBoostingClassifier(random_state=42)
grad_boost.fit(X_train, y_train)
y_pred_grad_boost = grad_boost.predict(X_test)
print("Gradient Boosting - Accuracy:", accuracy_score(y_test, y_pred_grad_boost))
print("Classification Report:\n", classification_report(y_test, y_pred_grad_boost))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_grad_boost))

"""XGBoost"""

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost - Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))









