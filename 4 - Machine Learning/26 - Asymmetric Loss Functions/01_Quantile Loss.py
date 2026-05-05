'''
What you’ll observe
1. The quantile regression model will usually predict higher values when q=0.9
2. That’s intentional → it avoids underprediction
3. Its MAE may not be better, but its quantile loss will be lower
'''
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error

# ----------------------------
# 1. Load dataset (no download needed)
# ----------------------------
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 2. Model 1: Standard Regression (MSE)
# ----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# ----------------------------
# 3. Model 2: Quantile Regression
# ----------------------------
# quantile = 0.9 → penalizes underprediction more
qr = QuantileRegressor(quantile=0.9, alpha=0.0)
qr.fit(X_train, y_train)

y_pred_qr = qr.predict(X_test)

# ----------------------------
# 4. Custom Quantile Loss Function
# ----------------------------
def quantile_loss(y_true, y_pred, q):
    errors = y_true - y_pred
    return np.mean(np.maximum(q * errors, (q - 1) * errors))

# ----------------------------
# 5. Evaluation
# ----------------------------
print("=== Linear Regression (MSE) ===")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("Quantile Loss (q=0.9):", quantile_loss(y_test, y_pred_lr, 0.9))

print("=== Quantile Regression (q=0.9) ===")
print("MAE:", mean_absolute_error(y_test, y_pred_qr))
print("Quantile Loss (q=0.9):", quantile_loss(y_test, y_pred_qr, 0.9))

# ----------------------------
# 6. Compare predictions
# ----------------------------
print("Sample predictions (first 5):")
for i in range(5):
    print(f"Actual: {y_test[i]:.2f} | LR: {y_pred_lr[i]:.2f} | QR: {y_pred_qr[i]:.2f}")