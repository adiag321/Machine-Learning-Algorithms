import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ----------------------------
# 1. Load data
# ----------------------------
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 2. Train quantile models
# ----------------------------
# Lower quantile (10%)
model_q10 = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.1,   # lower bound
    n_estimators=100
)

# Median (50%)
model_q50 = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.5,
    n_estimators=100
)

# Upper quantile (90%)
model_q90 = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.9,   # upper bound
    n_estimators=100
)

# Train
model_q10.fit(X_train, y_train)
model_q50.fit(X_train, y_train)
model_q90.fit(X_train, y_train)

# ----------------------------
# 3. Predictions
# ----------------------------
pred_q10 = model_q10.predict(X_test)
pred_q50 = model_q50.predict(X_test)
pred_q90 = model_q90.predict(X_test)

# ----------------------------
# 4. Evaluate (optional)
# ----------------------------
print("MAE (median model):", mean_absolute_error(y_test, pred_q50))

# ----------------------------
# 5. Show prediction intervals
# ----------------------------
print("\nSample predictions with intervals:")
for i in range(5):
    print(f"""
Actual: {y_test[i]:.1f}
P10: {pred_q10[i]:.1f}
P50 (median): {pred_q50[i]:.1f}
P90: {pred_q90[i]:.1f}
""")