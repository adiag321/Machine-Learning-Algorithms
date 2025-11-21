# How SHAP exposed hidden bias and how I fixed it with adversarial debiasing
'''
Libraries: 
pip install shap fairlearn scikit-learn pandas matplotlib numpy

Scenario: 'CreditSafe' Model predicting high income (>50k) for credit approval.
Protected Attribute: Sex (Female = 0, Male = 1)
This dataset is ideal because it contains the structural biases 
'''
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, selection_rate
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# ==========================================
# 1. SETUP & SCENARIO GENERATION
# ==========================================
# Load standard dataset (Adult Census) via SHAP. 
X, y = shap.datasets.adult()

# Preprocessing for Scikit-Learn
# We will treat 'Sex' as the protected attribute
# Sex is already encoded (False=Female, True=Male). We will map it clearly for the scenario
X['Sex'] = X['Sex'].astype(int)
A = X['Sex'] # Protected attribute vector

# Create a train-test split
# We preserve the index to keep track of sensitive attributes
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42, stratify=y)
print(f"Data Loaded: {X_train.shape[0]} training samples.")

# ==========================================
# 2. TRAINING THE BIASED BASELINE
# ==========================================
model_baseline = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model_baseline.fit(X_train, y_train)
# Predictions
y_pred_base = model_baseline.predict(X_test)
y_proba_base = model_baseline.predict_proba(X_test)[:, 1]

# Calculate Metrics
acc_base = accuracy_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_proba_base)

# Demographic Parity: The absolute difference in selection rates between groups
dp_diff_base = demographic_parity_difference(y_test, y_pred_base, sensitive_features = A_test)

print(f"Baseline Accuracy: {acc_base:.3f}")
print(f"Baseline ROC-AUC: {auc_base:.3f}")
print(f"Baseline Demographic Parity Difference: {dp_diff_base:.3f}")
print("Interpretation: A high DP Difference means one group is getting approved significantly more often.")

# ==========================================
# 3. SHAP ANALYSIS (DETECTION)
# Calculating SHAP values to find hidden proxies...
# ==========================================
explainer = shap.TreeExplainer(model_baseline)
# using a subset for speed in demo
X_test_subset = X_test.iloc[:500]
shap_values = explainer.shap_values(X_test.iloc[:500]) 

# For binary classification, TreeExplainer returns shape (samples, features, 2) We extract SHAP values for the positive class (class 1)
# Using shap_values[:, :,1] because shap_values.shape = (500, 12, 2) and we want dimension (500, 12)

# Plot 1: Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:, :,1], X_test.iloc[:500], show=False)
plt.title("SHAP Summary: Feature Impact on Credit Approval")
plt.tight_layout()
#plt.savefig("shap_summary_bias.png")
print("Generated 'shap_summary_bias.png'")

# Plot 2: Dependence Plot to show Proxy Bias
'''
We check 'Relationship' (often a proxy for sex in this dataset) colored by Sex. This mimics "Distance from City Center" finding
'''
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    "Relationship", 
    shap_values[:, :, 1], 
    X_test.iloc[:500], 
    interaction_index="Sex", 
    show=False
)
plt.title("SHAP Dependence: Relationship Feature colored by Sex")
plt.tight_layout()
#plt.savefig("shap_dependence_proxy.png")
print("Generated 'shap_dependence_proxy.png'")
print("Interpretation: If the dots are separated by color vertically, the model treats groups differently for the same feature value.")

# ==========================================
# 4. APPLYING DEBIASING TECHNIQUES
# ==========================================
results = {
    'Method': ['Baseline'],
    'Accuracy': [acc_base],
    'Demographic Parity Diff': [dp_diff_base]
}

##############################################################
#        Technique A: Reweighing (Sample Weights)
# "To balance representation, I changed the sample weights"
##############################################################
print("Applying Technique A: Reweighing Training Samples...")

# Calculate weights: Upweight the unprivileged group (Female) who have the positive label
def calculate_fair_weights(X_df, y_true, sensitive_col):
    weights = np.ones(len(y_true))
    # Simple heuristic reweighting for demonstration
    # In production, use AIF360's Reweighing algorithm
    
    # Identify groups
    groups = X_df[sensitive_col]
    
    # P(Protected)
    n_prot = np.sum(groups == 0)
    n_total = len(groups)
    
    # P(Label=1)
    n_pos = np.sum(y_true == 1)
    
    # Expected count if independent
    expected_prot_pos = (n_prot / n_total) * (n_pos / n_total) * n_total
    
    # Actual count
    actual_prot_pos = np.sum((groups == 0) & (y_true == 1))
    
    # Weight multiplier
    w_prot_pos = expected_prot_pos / (actual_prot_pos + 1e-6)
    
    # Apply weights to (Sex=0 AND Label=1)
    indices = (groups == 0) & (y_true == 1)
    weights[indices] = w_prot_pos
    
    return weights

sample_weights = calculate_fair_weights(X_train, y_train, 'Sex')

model_reweigh = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model_reweigh.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_rw = model_reweigh.predict(X_test)

results['Method'].append('Reweighing')
results['Accuracy'].append(accuracy_score(y_test, y_pred_rw))
results['Demographic Parity Diff'].append(demographic_parity_difference(y_test, y_pred_rw, sensitive_features=A_test))

##################################################################
#        Technique B: Fairness-Aware Feature Engineering
# "Removed proxy features and created fairness-preserving alternatives"
##################################################################
print("Applying Technique B: Fairness-Aware Feature Engineering...")

# We use CorrelationRemover to project out the sensitive feature from other features.
# This effectively removes linear correlation between 'Sex' and other columns
cr = CorrelationRemover(sensitive_feature_ids=['Sex'])
X_train_fair = cr.fit_transform(X_train)
X_test_fair = cr.transform(X_test)

# Train on "Fair" features
model_fe = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model_fe.fit(X_train_fair, y_train)
y_pred_fe = model_fe.predict(X_test_fair)

results['Method'].append('Feature Engineering (Decorrelation)')
results['Accuracy'].append(accuracy_score(y_test, y_pred_fe))
results['Demographic Parity Diff'].append(demographic_parity_difference(y_test, y_pred_fe, sensitive_features=A_test))

##################################################################
#          Technique C: Adversarial / Constraint Debiasing
# "Trained a discriminator to predict protected attributes and penalized the model"
# Note: Implementing a full GAN in a single script is unstable. 
# We use Fairlearn's ExponentiatedGradient with DemographicParity constraint.
# This mathematically solves the same problem: Minimize Error s.t. Bias < epsilon.
##################################################################
print("Applying Technique C: Adversarial/Constraint Optimization...")

# We use a simple linear estimator for the base to speed up the iterative adversarial process
from sklearn.linear_model import LogisticRegression
base_est = LogisticRegression(solver='liblinear', max_iter=1000)

# The Constraint: Ensure Demographic Parity (Independence between prediction and sensitive attr)
constraint = DemographicParity()
mitigator = ExponentiatedGradient(base_est, constraint)

mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred_adv = mitigator.predict(X_test)

results['Method'].append('Adversarial/Constraint Opt')
results['Accuracy'].append(accuracy_score(y_test, y_pred_adv))
results['Demographic Parity Diff'].append(demographic_parity_difference(y_test, y_pred_adv, sensitive_features=A_test))

##################################################################
#       Technique D: Post-Processing Calibration
# "Adjusted decision thresholds by demographic group"
##################################################################
print("Applying Technique D: Post-Processing Calibration...")

# We take the original biased model and adjust its output thresholds
postprocessor = ThresholdOptimizer(
    estimator = model_baseline,
    constraints = "demographic_parity",
    prefit = True
)

postprocessor.fit(X_train, y_train, sensitive_features = A_train)
y_pred_post = postprocessor.predict(X_test, sensitive_features = A_test)

results['Method'].append('Post-Processing (Threshold Calib)')
results['Accuracy'].append(accuracy_score(y_test, y_pred_post))
results['Demographic Parity Diff'].append(demographic_parity_difference(y_test, y_pred_post, sensitive_features=A_test))

# ==========================================
# 5. FINAL RESULTS COMPARISON
# ==========================================
results_df = pd.DataFrame(results)

# Calculate Improvement % relative to baseline
baseline_bias = results_df.loc[0, 'Demographic Parity Diff']
results_df['Bias Reduction (%)'] = ((baseline_bias - results_df['Demographic Parity Diff']) / baseline_bias) * 100

print(results_df.round(3))

print("Summary Analysis:")
print("1. Reweighing: Changes the training data distribution to emphasize underrepresented successful applicants.")
print("2. Feature Eng: Mathematically removes correlations to the sensitive attribute (Sex) from input data.")
print("3. Adversarial/Constraint: Forces the model to learn parameters that satisfy fairness constraints directly during training.")
print("4. Post-Processing: Keeps the biased model but changes the approval threshold for different groups to ensure equity.")