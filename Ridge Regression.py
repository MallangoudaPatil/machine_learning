import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([3, 5, 7, 10, 15, 21, 28, 36, 45])  # Non-linear growth

# Create Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # alpha = λ (regularization strength)
ridge_model.fit(X, y)

# Predictions
y_pred = ridge_model.predict(X)

# Coefficient & intercept
print("Coefficient:", ridge_model.coef_[0])
print("Intercept:", ridge_model.intercept_)

# Predict for new value
print("Predicted for X=10:", ridge_model.predict([[10]])[0])

# Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Ridge Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
