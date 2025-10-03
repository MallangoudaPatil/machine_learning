import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([2, 6, 14, 28, 45, 68, 96, 130, 170])  # Non-linear growth

# Transform input features into polynomial features
poly = PolynomialFeatures(degree=2)   # quadratic (x^2)
X_poly = poly.fit_transform(X)

# Fit Linear Regression on transformed data
model = LinearRegression()
model.fit(X_poly, y)

# Predictions
y_pred = model.predict(X_poly)

# Print coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict for 10 hours of study
X_test = poly.transform([[10]])
print("Predicted score for 10 hours:", model.predict(X_test)[0])

# Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Polynomial Regression Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()
