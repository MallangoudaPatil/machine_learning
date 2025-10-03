import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([3, 5, 7, 10, 15, 21, 28, 36, 45])  # Non-linear growth

# Create Lasso Regression model
lasso_model = Lasso(alpha=0.5)  # alpha = λ (regularization strength)
lasso_model.fit(X, y)

# Predictions
y_pred = lasso_model.predict(X)

# Coefficient & intercept
print("Coefficient:", lasso_model.coef_[0])
print("Intercept:", lasso_model.intercept_)

# Predict for new value
print("Predicted for X=10:", lasso_model.predict([[10]])[0])

# Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Lasso Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
