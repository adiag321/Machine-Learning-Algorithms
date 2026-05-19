# Calibration Interpretation Guide

> Reference script: `01_Calibration.py`

---

## What Is This Script Doing?

Many classifiers (like Random Forest or Naive Bayes) output a "probability" score, but this score often doesn't represent the **true** probability of an event occurring. For example, a score of 0.8 might only be correct 60% of the time. **Calibration** aims to fix this so that a predicted probability of 0.8 actually means there is an 80% chance of the event occurring.

The script:
1. Trains two models (Random Forest, Naive Bayes) on synthetic data.
2. Gets **uncalibrated** probabilities from each model.
3. Applies two calibration techniques — **Platt Scaling** (sigmoid fit) and **Isotonic Regression** (non-parametric monotonic fit) — to produce **calibrated** probabilities.
4. Evaluates all three versions (uncalibrated, Platt, Isotonic) across 20 decision thresholds (0.00 to 0.95).
5. Produces three output dataframes and reliability diagrams.

### Data Split Strategy (3-Way Split)

| Set | Size | Purpose |
|-----|------|---------|
| **Train** | 60% (3,000 samples) | Train the base model |
| **Calibration** | 20% (1,000 samples) | Train the calibrators (Platt/Isotonic). Must be separate from training data to avoid overfitting. |
| **Test** | 20% (1,000 samples) | Evaluate final performance of uncalibrated and calibrated models |

### Key Metrics Used

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Brier Score** | Mean squared error between predicted probabilities and true labels. Measures calibration + discrimination jointly. | Close to 0 |
| **ECE (Expected Calibration Error)** | Weighted average of |predicted probability − actual frequency| across bins. Pure calibration metric. | Close to 0 |
| **ROC AUC** | How well the model ranks positive samples above negative ones. Pure discrimination metric. | Close to 1 |
| **Accuracy** | Fraction of correct binary predictions at a given threshold. | Close to 1 |
| **Precision** | Of all positive predictions, how many are actually positive. | Close to 1 |
| **Recall** | Of all actual positives, how many did the model catch. | Close to 1 |

---

## DataFrame 1: `final_summary_df` (Threshold Summary)

### Structure

Each row is a unique combination of **Algorithm × Calibration State × Threshold**.

| Column | Description |
|--------|-------------|
| `Algorithm` | Which model (`Random Forest` or `Naive Bayes`) |
| `Calibration_State` | Which probability version (`Uncalibrated`, `Platt Scaling`, or `Isotonic`) |
| `Threshold` | Decision cutoff (0.00, 0.05, 0.10, … 0.95). If `probability ≥ threshold`, predict positive. |
| `Brier_Score` | Calibration quality of the probabilities — **same across all thresholds** for a given Algorithm + Calibration combo |
| `ECE` | Expected Calibration Error — also **threshold-independent** |
| `ROC_AUC` | Discriminative power — also **threshold-independent** |
| `Accuracy` | Fraction of correct predictions **at this specific threshold** |
| `Precision` | Of positive predictions at this threshold, fraction that are truly positive |
| `Recall` | Of actual positives, fraction caught at this threshold |
| `Positive_Preds` | Raw count of samples predicted as positive |
| `Train_Samples / Calib_Samples / Test_Samples` | Dataset sizes (constant metadata) |

### How to Read It

**Example row** (Random Forest, Platt Scaling, Threshold=0.60):

> "Using the Random Forest model with Platt-calibrated probabilities and a decision cutoff of 0.60: the Brier score is 0.0617, accuracy is 90.2%, precision is X%, recall is Y%, and Z samples were predicted positive."

**Key insight:** `Brier_Score`, `ECE`, and `ROC_AUC` stay the same across all thresholds for a given Algorithm + Calibration combo. These measure the quality of the **raw probabilities**, not the binary predictions. Use them to compare calibration methods. The remaining columns (`Accuracy`, `Precision`, `Recall`, `Positive_Preds`) **change with threshold** — use them to find the best operating point.

### Interpretation Tips

1. **Compare calibration methods at the same threshold:** Filter to threshold=0.50 and compare Brier/ECE across Uncalibrated, Platt, Isotonic. Lower Brier/ECE = better calibrated probabilities.
2. **Find the right operating point:** Filter to one calibration state and scan across thresholds:
   - Low threshold (e.g., 0.10) → high Recall, low Precision, many Positive_Preds
   - High threshold (e.g., 0.90) → low Recall, high Precision, few Positive_Preds
3. **Key rule:** Calibration should improve Brier/ECE **without hurting** ROC_AUC. If AUC stays the same but Brier drops, the probabilities became more trustworthy without losing ranking ability.

---

## DataFrame 2: `final_predictions_df` (Row-Wise Prediction Log)

### Structure

Each row represents **one test observation at one specific threshold**. The same observation appears 20 times (once per threshold).

| Column | Description |
|--------|-------------|
| `Algorithm` | Which model |
| `Threshold` | Decision cutoff for this row |
| `y_actual` | Ground truth label (0 or 1) |
| `Prob_Uncalibrated` | Raw probability from the base model |
| `Prob_Platt` | Probability after Platt Scaling |
| `Prob_Isotonic` | Probability after Isotonic Regression |
| `Pred_Uncalibrated` | Binary prediction: 1 if `Prob_Uncalibrated ≥ Threshold`, else 0 |
| `Pred_Platt` | Binary prediction: 1 if `Prob_Platt ≥ Threshold`, else 0 |
| `Pred_Isotonic` | Binary prediction: 1 if `Prob_Isotonic ≥ Threshold`, else 0 |

### How to Read a Single Row

**Example:**

| Algorithm | Threshold | y_actual | Prob_Uncal | Prob_Platt | Prob_Iso | Pred_Uncal | Pred_Platt | Pred_Iso |
|-----------|-----------|----------|------------|------------|----------|------------|------------|----------|
| Random Forest | 0.50 | 1 | 0.92 | 0.88 | 0.85 | 1 | 1 | 1 |

> "For this sample, the true label is 1. The uncalibrated model gave 92% probability, Platt gave 88%, Isotonic gave 85%. At a 0.50 cutoff, all three predict positive — and they're all correct."

**Another example:**

| Algorithm | Threshold | y_actual | Prob_Uncal | Prob_Platt | Prob_Iso | Pred_Uncal | Pred_Platt | Pred_Iso |
|-----------|-----------|----------|------------|------------|----------|------------|------------|----------|
| Random Forest | 0.50 | 0 | 0.64 | 0.4356 | 0.2774 | 1 | 0 | 0 |

> "For this sample, the true label is 0. The uncalibrated model said 64% probability — a false positive! But Platt corrected it to 43.6% and Isotonic to 27.7%, both below the 0.50 cutoff. Calibration fixed this error."

### Interpretation Tips

1. **Spot disagreements:** Filter to a threshold and look for rows where `Pred_Uncalibrated ≠ Pred_Platt`. These are the samples where calibration actually **changed the decision** — the most interesting cases to audit.
2. **Understand probability shifts:** Compare the three `Prob_*` columns for the same sample. Calibration should pull overconfident probabilities down and push underconfident ones up.
3. **Error analysis:** Filter to `y_actual = 1` and `Pred_Platt = 0` (false negatives) at a given threshold. Are the probabilities borderline (e.g., 0.48)? That tells you those samples are ambiguous.
4. **Threshold sensitivity:** Scan the same observation across thresholds to see at what point it "flips" from positive to negative.

### Table Size

The table is massive: **2 models × 1,000 test samples × 20 thresholds = 40,000 rows**. It is designed to be queried and filtered, not read end-to-end.

---

## DataFrame 3: `combined_df` (Combined Comparison)

### Structure

Each row represents one **Algorithm × Threshold** combination. It merges aggregate accuracy from the summary table with observation-level flip analysis from the predictions table.

| Column | Description |
|--------|-------------|
| `Algorithm` | Which model |
| `Threshold` | Decision cutoff |
| `Acc_Uncal` | Accuracy of the uncalibrated model |
| `Acc_Platt` | Accuracy after Platt Scaling |
| `Acc_Iso` | Accuracy after Isotonic Regression |
| `Platt_Flips` | Total predictions that **changed** (uncalibrated → Platt) |
| `Platt_Fixed` | Of those flips, how many **corrected** a wrong prediction (was wrong → now right) |
| `Platt_Broke` | Of those flips, how many **ruined** a correct prediction (was right → now wrong) |
| `Platt_Net` | `Fixed − Broke`. **Positive = calibration helped. Negative = calibration hurt.** |
| `Iso_Flips` | Same as above but for Isotonic Regression |
| `Iso_Fixed` | Isotonic fixes |
| `Iso_Broke` | Isotonic errors introduced |
| `Iso_Net` | Isotonic net improvement |

### Concrete Examples

#### Example 1: Calibration Is a Hero (Random Forest, t=0.95)

```
Algorithm      Threshold  Acc_Uncal  Acc_Platt  Acc_Iso  Platt_Flips  Platt_Fixed  Platt_Broke  Platt_Net  Iso_Flips  Iso_Fixed  Iso_Broke  Iso_Net
Random Forest       0.95      0.578      0.792    0.816          220          217            3        214        250        244          6      238
```

**Interpretation:** At a very high threshold (0.95), the uncalibrated RF only gets 57.8% accuracy — near coin flip. RF's raw probabilities rarely reach 0.95, so it predicts almost everything as negative.

- Isotonic flipped 250 predictions. **244 were fixes**, only 6 were broken. Net = **+238**.
- Accuracy jumped from 57.8% → 81.6%.
- **Takeaway:** If your business requires a very high confidence threshold (e.g., medical diagnosis where you want ≥95% certainty), calibration is **essential** — the raw model is near-useless but calibrated probabilities rescue it.

---

#### Example 2: Calibration Is a Villain (Random Forest, t=0.40)

```
Algorithm      Threshold  Acc_Uncal  Acc_Platt  Acc_Iso  Platt_Flips  Platt_Fixed  Platt_Broke  Platt_Net  Iso_Flips  Iso_Fixed  Iso_Broke  Iso_Net
Random Forest       0.40      0.947      0.921    0.919           52           13           39        -26         58         15         43      -28
```

**Interpretation:** At RF's sweet spot (t=0.40, highest accuracy at 94.7%), calibration **hurts**.

- Platt flipped 52 predictions: only 13 were fixes, but **39 were broken**. Net = **-26**.
- Accuracy dropped from 94.7% → 92.1%.
- **Takeaway:** RF is already well-calibrated around the mid-range. Applying calibration here is like "fixing" something that isn't broken — it over-corrects.

---

#### Example 3: The Crossover Point (Random Forest, t=0.25)

```
Algorithm      Threshold  Acc_Uncal  Acc_Platt  Acc_Iso  Platt_Flips  Platt_Fixed  Platt_Broke  Platt_Net
Random Forest       0.25      0.892      0.909    0.909           75           46           29         17
```

**Interpretation:** At t=0.25, calibration is barely positive (Net = +17). This is the **breakeven zone**. Below this threshold calibration helps massively; above it, calibration hurts. This tells you RF's probability distortion lives mainly in the 0.05–0.25 range.

---

#### Example 4: Naive Bayes Is a Mess (Naive Bayes, t=0.90)

```
Algorithm    Threshold  Acc_Uncal  Acc_Platt  Acc_Iso  Platt_Flips  Platt_Fixed  Platt_Broke  Platt_Net  Iso_Flips  Iso_Fixed  Iso_Broke  Iso_Net
Naive Bayes       0.90      0.705      0.512    0.697          213           10          203       -193         54         23         31       -8
```

**Interpretation:** Platt scaling is **catastrophic** here — it flipped 213 predictions, 203 were broken, only 10 fixed (Net = -193). Accuracy crashed to 51.2% (coin flip!). But Isotonic only had Net = -8, barely hurting.

- **Takeaway:** The two calibration methods behave very differently. Platt assumes a sigmoid-shaped correction — if the model's error pattern doesn't match a sigmoid, Platt fails badly. Isotonic is non-parametric and more flexible, handling irregular distributions better.

---

### Rules of Thumb for `combined_df`

| What You See | What It Means |
|---|---|
| `Net` is large positive | ✅ Calibration is fixing many errors at this threshold — **use it** |
| `Net` is near zero | ⚖️ Calibration makes no material difference — skip it for simplicity |
| `Net` is large negative | ❌ Calibration is hurting — **do NOT use it** at this threshold |
| `Flips` is high but `Net` ≈ 0 | ⚠️ Calibration is **churning** — changing lots of predictions but not improving anything |
| `Acc_Platt > Acc_Iso` | Platt (sigmoid) fits this model's error pattern better |
| `Acc_Iso > Acc_Platt` | Isotonic (flexible) fits better — the model's errors aren't sigmoid-shaped |

---

## Big-Picture Takeaways

1. **Calibration ≠ better accuracy.** Calibration makes probabilities more **trustworthy**, not necessarily more **accurate** at the default 0.50 threshold. ECE/Brier can improve while accuracy drops.

2. **Random Forest is already well-calibrated** around mid-range thresholds, so calibration mostly hurts there. But at extreme thresholds (very low or very high), calibration is transformative.

3. **Naive Bayes is poorly calibrated** because it assumes feature independence, which pushes probabilities toward 0 and 1. Calibration (especially Isotonic) can help, but Platt Scaling can make things worse if the error shape isn't sigmoid.

4. **Always evaluate calibration at YOUR threshold** — the threshold you'll actually deploy at in production. A method that helps at t=0.10 may hurt at t=0.50.

5. **The `Net` column is the ultimate answer:** it tells you, at each threshold, the exact number of predictions that calibration improved minus the number it degraded. If Net > 0, calibration earned its keep.
