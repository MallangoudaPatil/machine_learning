import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18]) + np.random.randn(9) * 0.5  # Add noise

# Create Bayesian Ridge Regression model
bayes_model = BayesianRidge()
bayes_model.fit(X, y)

# Predictions
y_pred, y_std = bayes_model.predict(X, return_std=True)  # Also get standard deviation

# Coefficient & intercept
print("Coefficient:", bayes_model.coef_[0])
print("Intercept:", bayes_model.intercept_)

# Predict for new value
X_test = np.array([[10]])
y_test, y_test_std = bayes_model.predict(X_test, return_std=True)
print("Predicted for X=10:", y_test[0], "±", y_test_std[0])

# Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Bayesian Regression Line")
plt.fill_between(X.ravel(), y_pred - y_std, y_pred + y_std, color='pink', alpha=0.3, label="Uncertainty")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
