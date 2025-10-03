# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
# X = hours studied, y = exam scores
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([20, 40, 60, 80, 100])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Print slope (coefficient) and intercept
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict score for 6 hours of study
print("Predicted score for 6 hours:", model.predict([[6]])[0])

# Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()
