import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import os

# Load the dataset
data = pd.read_csv('c:/Users/shawn/Documents/Forest Data/covertype_train.csv')

# Choose a subset of features for both models
selected_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology']

# Extract features and labels
X = data[selected_features].values
Y = data['class'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the KNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Calculate the accuracy of the KNN model on the test set
knn_accuracy = neigh.score(X_test, y_test)
print("KNN Accuracy:", knn_accuracy)

# Create and train a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions using the linear regression model
predictions = linear_model.predict(X_test)

# Evaluate the linear regression model using Mean Squared Error (MSE)
mse = np.mean((predictions - y_test) ** 2)
print("Linear Regression MSE:", mse)

# Calculate R-squared (accuracy) for the linear regression model
r_squared = linear_model.score(X_test, y_test)
print("Linear Regression R-squared (Accuracy):", r_squared)

# Visualize the linear regression predictions against the true labels
plt.scatter(X_test[:, 0], y_test, label='True Labels')
plt.plot(X_test[:, 0], predictions, color='red', label='Linear Regression Predictions')
plt.xlabel('Elevation')
plt.ylabel('Class')
plt.legend()
plt.show()
