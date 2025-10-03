import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])  # Quadratic growth

# Create Decision Tree Regressor
tree_model = DecisionTreeRegressor(max_depth=3)  # Limit depth to avoid overfitting
tree_model.fit(X, y)

# Predictions
y_pred = tree_model.predict(X)

# Predict for a new value
X_test = np.array([[10]])
print("Predicted value for X=10:", tree_model.predict(X_test)[0])

# Visualization
X_grid = np.arange(1, 10.1, 0.1).reshape(-1, 1)  # Smooth curve
y_grid_pred = tree_model.predict(X_grid)

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X_grid, y_grid_pred, color='red', label="Decision Tree Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
