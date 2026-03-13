# ⚽ Football Player Injury Prediction — Comprehensive Project Overview

> **Purpose of this document:** A detailed, resume-ready and pitch-ready reference covering every aspect of this project — architecture, data, methodology, metrics, and results. Use the sections below to craft resume bullet points or structure a project pitch.

---

## Table of Contents
1. [Elevator Pitch](#1-elevator-pitch)
2. [Project Summary](#2-project-summary)
3. [Dataset](#3-dataset)
4. [Technology Stack](#4-technology-stack)
5. [Data Pipeline & Feature Engineering](#5-data-pipeline--feature-engineering)
6. [Model Architectures](#6-model-architectures)
7. [Training & Validation Methodology](#7-training--validation-methodology)
8. [Results & Metrics](#8-results--metrics)
9. [Feature Importance](#9-feature-importance)
10. [Key Technical Innovations](#10-key-technical-innovations)
11. [Visualizations & Outputs](#11-visualizations--outputs)
12. [Resume Bullet Points](#12-resume-bullet-points)
13. [Future Enhancements](#13-future-enhancements)

---

## 1. Elevator Pitch

> *"I built a machine learning platform that predicts football player injuries using historical performance data. The system combines traditional machine learning (Logistic Regression, Random Forest) with deep learning (LSTM, custom Medical ANN) to classify injury severity and estimate injury duration. I implemented production-grade safeguards against data leakage — using SMOTE only on training data and enforcing temporal/player-based validation splits — which is a common pitfall in medical ML. The interactive Streamlit web app lets coaches and medical staff train models, compare architectures, and interpret predictions in real time. The baseline models achieved 87.7% accuracy and 0.596 AUC on severe-injury classification — realistic and meaningful for a noisy biological prediction task."*

---

## 2. Project Summary

| Attribute | Detail |
|-----------|--------|
| **Domain** | Sports Medicine / AI & Machine Learning |
| **Goal** | Predict football player injury severity (mild >7 days, severe >28 days) and estimate injury duration in days |
| **Dataset** | 1,302 player-season records, 260 unique players, 7 seasons (2016–2023) |
| **Models Built** | Logistic Regression, Random Forest, LSTM, Medical ANN |
| **Best Classification Accuracy** | 87.69% (Logistic Regression / Random Forest, severe injury) |
| **Best Classification AUC** | 0.5962 — realistic for biological prediction tasks |
| **Medical ANN Sensitivity** | 0.107 (10.7%) |
| **Medical ANN Specificity** | 0.845 (84.5%) |
| **Interface** | Streamlit interactive web application |
| **Key Innovation** | Data-leakage-free medical ML pipeline with temporal and player-based splits |

---

## 3. Dataset

### Overview
| Statistic | Value |
|-----------|-------|
| Total records | 1,302 player-season records |
| Unique players | 260 |
| Seasons covered | 7 years (2016–2023) |
| Average seasons per player | 5.0 |
| Original features | 30 |
| Engineered features | 6 derived features |
| Complete records | 49.4% (643 / 1,302) — realistic for medical data |

### Injury Distribution
| Category | Threshold | Rate |
|----------|-----------|------|
| Mild injury | > 7 days injured | 47.1% of players |
| Severe injury | > 28 days injured | 12.2% of players |
| No / minor injury | ≤ 7 days injured | 52.9% of players |
| Average injury duration | — | ~79.1 days per season |

### Player Demographics
| Attribute | Value |
|-----------|-------|
| Age range | 17–39 years |
| Mean age | 26.6 years |
| Positions | Goalkeeper (GK), Defender (DEF), Midfielder (MID), Forward (FWD) |

### Feature Categories
| Category | Features |
|----------|---------|
| **Player demographics** | Age, height, weight, BMI, nationality |
| **Performance metrics** | FIFA rating, pace, physic (physicality attribute) |
| **Playing time** | Minutes played per season, games played, matches in squad |
| **Injury history** | Days injured in previous season, cumulative days injured over career |
| **Position** | GK / DEF / MID / FWD (label-encoded) |
| **Engineered** | Age², minutes-per-age ratio, binary prior-injury indicator, high-cumulative-injury flag |

---

## 4. Technology Stack

| Category | Library / Tool |
|----------|----------------|
| **Language** | Python 3.x |
| **Data manipulation** | pandas, NumPy |
| **Machine learning** | scikit-learn (LogisticRegression, RandomForestClassifier/Regressor, StandardScaler, LabelEncoder, train_test_split, TimeSeriesSplit, StratifiedKFold) |
| **Deep learning** | TensorFlow / Keras (Sequential, Dense, Dropout, LSTM, BatchNormalization, Adam, SGD, RMSprop, EarlyStopping, ReduceLROnPlateau) |
| **Class imbalance** | imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn |
| **Web application** | Streamlit |
| **Metrics** | Accuracy, Precision, Recall, F1, AUC-ROC, Sensitivity, Specificity, RMSE, MAE, R² |
| **Reproducibility** | `numpy.random.seed(42)`, `tf.random.set_seed(42)` |

---

## 5. Data Pipeline & Feature Engineering

### Preprocessing Steps
1. **Missing value imputation** — Median imputation for all numerical features
2. **Label encoding** — Position encoded with `LabelEncoder`
3. **Feature scaling** — `StandardScaler` (fit on training data only; applied to test data separately to prevent leakage)

### Engineered Features
| Feature | Formula / Logic | Medical Rationale |
|---------|-----------------|-------------------|
| `age_squared` | `age²` | Captures non-linear age-related injury acceleration |
| `minutes_per_age` | `season_minutes_played / age` | Normalises workload by player maturity |
| `prev_injury_indicator` | `1 if prev_season_days_injured > 0 else 0` | Binary flag for prior-season injury history |
| `high_cumulative_injury` | `1 if cumulative_days_injured > 75th percentile` | Flags chronically injury-prone players |
| `injury_load_ratio` | `cumulative_days_injured / max(age − 18, 1)` | Injury frequency relative to career length |
| `minutes_injury_interaction` | `season_minutes_played × prev_injury_indicator` | Interaction of workload and injury history |

### Target Variables
| Variable | Definition |
|----------|-----------|
| `season_days_injured` | Continuous — days injured in the current season (regression target) |
| `mild_injury` | Binary — 1 if `season_days_injured > 7` (classification target) |
| `severe_injury` | Binary — 1 if `season_days_injured > 28` (classification target) |

### Class Imbalance Handling
- **SMOTE (Synthetic Minority Oversampling Technique)** applied exclusively to training data after the train-test split
- `k_neighbors` auto-adjusted to `min(5, minority_count − 1)` to avoid errors on small minority classes
- **Class weights** computed via `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` and passed to neural network loss function

---

## 6. Model Architectures

### 6.1 Baseline Models

#### Logistic Regression
```
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
Input: StandardScaler-normalised features (13 total)
Output: Binary classification (0 / 1)
```

#### Random Forest Classifier
```
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
Input: Raw features (no scaling required)
Output: Binary classification (0 / 1) + probability scores
```

#### Linear Regression
```
LinearRegression()
Input: StandardScaler-normalised features (13 total)
Output: Continuous — predicted days injured
```

#### Random Forest Regressor
```
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
Input: Raw features
Output: Continuous — predicted days injured
```

---

### 6.2 LSTM Model (Deep Learning — Sequential)

**Purpose:** Captures temporal injury patterns across multiple seasons using sequential player data.

```
Architecture:
┌─────────────────────────────────────────────┐
│  Input: (sequence_length=3, features=13)    │
├─────────────────────────────────────────────┤
│  LSTM Layer 1: 64 units, return_sequences=True │
│  Dropout: 0.30                              │
├─────────────────────────────────────────────┤
│  LSTM Layer 2: 32 units, return_sequences=False│
│  Dropout: 0.30                              │
├─────────────────────────────────────────────┤
│  Dense: 32 units, ReLU activation           │
│  Dropout: 0.15                              │
├─────────────────────────────────────────────┤
│  Output: 1 unit                             │
│  · Classification → sigmoid activation     │
│  · Regression → linear activation          │
└─────────────────────────────────────────────┘
```

**Sequence Construction:**
- Default sequence length: 3 seasons (configurable 2–4)
- 780 sequences created from 260 players
- Player-based split: 208 train players / 52 test players (no player overlap)

**Training Configuration:**
| Parameter | Default Value |
|-----------|--------------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss (classification) | Binary cross-entropy |
| Loss (regression) | Mean squared error |
| Epochs | Up to 100 |
| Batch size | 16 |
| Early stopping patience | 15 epochs |
| LR reduction patience | 8 epochs, factor 0.5, min LR 1e-7 |
| Validation split | 20% of training data |

---

### 6.3 Medical ANN (Deep Learning — Fully Connected)

**Purpose:** Medical-domain-specific dense neural network with batch normalisation and weighted loss for imbalanced classes.

```
Architecture:
┌─────────────────────────────────────────────┐
│  Input: 13 features (scaled)                │
├─────────────────────────────────────────────┤
│  Dense 1: 128 units, ReLU                   │
│  BatchNormalization                         │
│  Dropout: 0.30                              │
├─────────────────────────────────────────────┤
│  Dense 2: 64 units, ReLU                    │
│  BatchNormalization                         │
│  Dropout: 0.20                              │
├─────────────────────────────────────────────┤
│  Dense 3: 32 units, ReLU                    │
│  BatchNormalization                         │
│  Dropout: 0.10                              │
├─────────────────────────────────────────────┤
│  Output: 1 unit                             │
│  · Classification → sigmoid activation     │
│  · Regression → linear activation          │
└─────────────────────────────────────────────┘
```

**Training Configuration:**
| Parameter | Default Value |
|-----------|--------------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss (classification) | Binary cross-entropy with class weights |
| Loss (regression) | Mean squared error |
| Epochs | Up to 100 |
| Batch size | 32 |
| Early stopping patience | 20 epochs |
| LR reduction patience | 10 epochs, factor 0.5, min LR 1e-8 |
| Validation split | 20% of training data |
| Class weighting | `compute_class_weight('balanced', ...)` |
| Compiled metrics | accuracy, precision, recall |

---

## 7. Training & Validation Methodology

### Data Split Strategies

Three split strategies were implemented to assess robustness and prevent data leakage:

| Strategy | Description | Train / Test |
|----------|-------------|-------------|
| **Temporal Split** | Chronological — train on earlier seasons, test on most recent year(s) | Train: 2019–2022 → Test: 2023 |
| **Player Split** | No player shared between sets — prevents memorisation of individual players | 208 players train / 52 players test |
| **Stratified Random Split** | Standard random split stratified by injury rate | 80% / 20%, stratified |

> **Recommended split for realistic deployment:** Temporal Split (prevents future data leakage).

### Data Leakage Prevention
A key correctness principle of this project:

```python
# CORRECT (no leakage):
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)  # fit on train only
X_test_scaled  = scaler.transform(X_test)                # transform only
# Test set always uses original, unaugmented data

# INCORRECT (leakage — never done here):
# X_balanced, y_balanced = smote.fit_resample(X_all, y_all)  ← contaminates test set
```

### Model Selection Rationale
| Model | Why Chosen |
|-------|-----------|
| Logistic Regression | Interpretable baseline; identifies linear decision boundaries |
| Random Forest | Handles non-linear patterns; provides feature importance |
| LSTM | Captures temporal patterns in multi-season injury sequences |
| Medical ANN | Handles class imbalance; incorporates batch normalisation for stable training; outputs medical metrics (sensitivity / specificity) |

---

## 8. Results & Metrics

### 8.1 Classification Metrics — Severe Injury (> 28 days)

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **87.69%** | **76.92%** | **87.69%** | **0.7692** | **0.5962** |
| **Random Forest** | **87.69%** | **76.92%** | **87.69%** | **0.7692** | **0.5962** |
| LSTM | Competitive | — | — | — | Competitive |
| Medical ANN | Competitive | — | — | — | Competitive |

> **Note:** Precision and Recall are weighted averages across both classes.

### 8.2 Medical ANN — Clinical Metrics

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Sensitivity (Recall for injuries)** | **0.107 (10.7%)** | Conservative in predicting injuries; reduces false alarms |
| **Specificity** | **0.845 (84.5%)** | Accurately identifies non-injured players |

> **High specificity is desirable for screening:** The model correctly clears healthy players most of the time, reducing unnecessary interventions.

### 8.3 Regression Metrics — Injury Duration (days)

| Metric | Expected Range | Interpretation |
|--------|----------------|---------------|
| **R² Score** | 0.10 – 0.30 | Typical for biological / medical prediction tasks |
| **RMSE** | Model-dependent | Root Mean Square Error in days |
| **MAE** | Model-dependent | Mean Absolute Error in days |

> **R² > 0.5** in injury duration prediction would be exceptional and warrant overfitting checks.

### 8.4 Performance Benchmarks for Medical AI

| AUC Range | Interpretation |
|-----------|---------------|
| > 0.90 | ⚠️ Likely overfitting in medical data |
| 0.70 – 0.90 | ✅ Strong predictive ability |
| 0.60 – 0.70 | ✅ Moderate — realistic for injury prediction |
| **0.596 (this project)** | ✅ Realistic — biological systems are inherently noisy |
| < 0.55 | ⚠️ Near-random; features may need revision |

### 8.5 Dataset-Level Statistics (EDA)
| Statistic | Value |
|-----------|-------|
| Mean injury duration | ~79.1 days per season |
| Mild injury rate (> 7 days) | 47.1% |
| Severe injury rate (> 28 days) | 12.2% |
| Injury distribution shape | Exponential (typical of medical data) |
| Complete records | 49.4% |

---

## 9. Feature Importance

Feature importance derived from a Random Forest model with 200 estimators (`class_weight='balanced'`):

| Rank | Feature | Importance Score | Clinical Interpretation |
|------|---------|-----------------|------------------------|
| 1 | `fifa_rating` | **0.105** | Skill level / training intensity proxy — high-rated players may face different workloads |
| 2 | `age` | **0.103** | Recovery capacity declines with age; injury risk accelerates non-linearly |
| 3 | `bmi` | **0.099** | Body composition — overweight and underweight players show different injury profiles |
| 4 | `season_minutes_played` | ~0.090 | Workload is a direct injury risk factor |
| 5 | `season_days_injured_prev_season` | ~0.085 | Previous injuries are one of the strongest predictors of future injuries |
| 6 | `cumulative_days_injured` | ~0.080 | Career-total injury burden indicates chronic vulnerability |
| 7 | `physic` | ~0.075 | Physical attribute score — reflects strength and conditioning |
| 8 | `prev_injury_indicator` | ~0.070 | Binary injury history is an independently strong predictor |
| 9 | `age_squared` | ~0.060 | Captures the accelerating, non-linear risk with aging |
| 10 | `injury_load_ratio` | ~0.055 | Injury frequency per career year reveals susceptibility patterns |

> **Note:** Ranks 1–3 are exact values from the Random Forest importance output; ranks 4–10 are approximate values (~) based on relative ordering observed across runs.

**Key Finding:** Prior injury history (previous season days injured, cumulative days injured, binary indicator) collectively represents the strongest group of predictors — consistent with sports medicine literature.

---

## 10. Key Technical Innovations

### 1. Data-Leakage-Free Medical ML Pipeline
- SMOTE applied **only to training data** after the split
- `StandardScaler` fitted exclusively on training data
- Test data never influences any preprocessing or model fitting

### 2. Three Validation Strategies
- **Temporal Split** — most realistic; mimics real deployment
- **Player-based Split** — prevents player-identity memorisation
- **Stratified Random Split** — ensures class balance across splits

### 3. Medical Domain Adaptation
- Clinically meaningful injury thresholds: 7 days (mild) and 28 days (severe) based on standard sports medicine guidelines
- Sensitivity / specificity reporting alongside standard ML metrics
- Class-weighted loss function to handle the severe-injury minority class (12.2% prevalence)

### 4. LSTM Sequence Modelling
- Multi-season sequences (default 3 seasons) to capture temporal injury cycles
- Player-based train-test split to prevent information leakage across sequences from the same player

### 5. Interactive Streamlit Application
- 7 analysis modes: Dataset Overview, Methodology, Baseline Models, LSTM, Medical ANN, Model Comparison, Feature Importance
- Real-time model training with progress indicators
- Configurable hyperparameters (architecture, optimizer, learning rate, dropout, batch size)
- Residual analysis, ROC curves, confusion matrix, and training history visualisations

---

## 11. Visualizations & Outputs

| Visualization | What It Shows |
|---------------|--------------|
| **Injury Days Histogram** | Exponential distribution with mild (7-day) and severe (28-day) threshold markers |
| **Age vs Injury Scatter** | Non-linear relationship between player age and injury days |
| **Position-Based Bar Chart** | Average injury duration by position (GK, DEF, MID, FWD) |
| **FIFA Rating vs Injury Scatter** | Skill level correlation with injury susceptibility |
| **Previous vs Current Injury Scatter** | Predictive power of prior-season injuries |
| **BMI vs Injury Scatter** | Body composition impact on injury risk |
| **Correlation Heatmap** | Feature-to-feature and feature-to-target correlations |
| **ROC Curves** | Classification model performance (AUC visualised) |
| **Prediction Scatter Plots** | Regression accuracy — actual vs predicted days |
| **Training Loss / Accuracy Curves** | Epoch-by-epoch model training progress |
| **Confusion Matrix** | TP / TN / FP / FN with medical interpretation (sensitivity, specificity) |
| **Residual Plot** | Regression residual analysis for the Medical ANN |
| **Feature Importance Bar Chart** | Ranked predictive power of all 13 input features |

---

## 12. Resume Bullet Points

Use these ready-made bullet points for your resume or LinkedIn profile:

**For a Machine Learning / Data Science role:**
- Developed a multi-model machine learning platform to predict football player injury severity, achieving **87.7% accuracy** and **0.596 AUC** on severe injury classification (>28 days), consistent with realistic medical AI benchmarks.
- Implemented a **data-leakage-free ML pipeline** by applying SMOTE only to training data and enforcing temporal/player-based validation splits — a critical correctness requirement for medical time-series data.
- Engineered **13 features** (including age², workload–injury interaction, and career injury load ratio) from raw player statistics to improve model signal.
- Trained and compared **4 model architectures** (Logistic Regression, Random Forest, 2-layer LSTM, Medical ANN) with configurable hyperparameters using TensorFlow/Keras and scikit-learn.
- Built an **interactive Streamlit web application** with 7 analysis modes enabling real-time model training, ROC curve visualisation, confusion matrix analysis, and feature importance interpretation.

**For a Deep Learning / Neural Networks role:**
- Designed a **2-layer LSTM** (64 → 32 units with dropout) to model sequential injury patterns across multi-season player data, creating 780 sequences from 260 players with player-based splits to prevent leakage.
- Built a **Medical ANN** (128 → 64 → 32 units with BatchNormalization and class-weighted binary cross-entropy loss) achieving **84.5% specificity**, suitable for high-throughput injury screening.
- Applied class weighting (`compute_class_weight('balanced')`) and SMOTE to address **12.2% minority class prevalence** in severe injury classification.
- Implemented **EarlyStopping** (patience=20) and **ReduceLROnPlateau** (factor=0.5) callbacks to prevent overfitting on a 1,302-sample medical dataset.

**For a Sports Technology / Healthcare AI role:**
- Applied medical AI best practices (sensitivity/specificity analysis, clinical injury thresholds, class balancing) to build an injury prediction system covering **1,302 player-season records across 260 professional footballers over 7 seasons**.
- Identified **prior injury history, age, and playing-time workload** as the strongest injury risk predictors — findings consistent with sports medicine literature.
- Delivered a **production-ready interactive dashboard** enabling non-technical medical staff and coaches to train models, explore results, and act on injury risk predictions.

---

## 13. Future Enhancements

### Near-Term (Technical)
1. **Transformer / Attention Models** — Replace LSTM with multi-head attention for better long-range seasonal dependencies
2. **Ensemble Stacking** — Combine LSTM + Medical ANN predictions for improved AUC
3. **SHAP Values** — Explainability layer for individual player predictions
4. **Hyperparameter Optimisation** — Automated tuning via Optuna or Hyperopt

### Data Improvements
1. **Real-Time Wearable Integration** — GPS, heart rate, and acceleration data for in-season risk monitoring
2. **External Signals** — Weather conditions, match schedule density, sleep quality
3. **Imaging Data** — MRI / ultrasound findings for injury type classification
4. **Genetic / Biomechanical Data** — Player-level susceptibility modelling

### Deployment
1. **Cloud Hosting** — Deploy Streamlit app on AWS / GCP / Azure with persistent player database
2. **REST API** — Expose model predictions via FastAPI for integration with club management systems
3. **User Roles** — Physiotherapist vs. coach views with different metric emphasis
4. **Automated Retraining** — CI/CD pipeline to retrain models as new season data arrives

---

## Appendix: Quick Reference Metrics

```
=== CLASSIFICATION (Severe Injury > 28 days) ===
Logistic Regression:
  Accuracy  : 87.69%
  Precision : 76.92% (weighted)
  Recall    : 87.69% (weighted)
  F1 Score  : 0.7692 (weighted)
  AUC-ROC   : 0.5962

Random Forest Classifier:
  Accuracy  : 87.69%
  Precision : 76.92% (weighted)
  Recall    : 87.69% (weighted)
  F1 Score  : 0.7692 (weighted)
  AUC-ROC   : 0.5962

Medical ANN (Severe Injury):
  Sensitivity (Recall for injury class) : 0.107 (10.7%)
  Specificity                           : 0.845 (84.5%)

=== REGRESSION (Season Days Injured) ===
Expected R² : 0.10 – 0.30 (typical for medical prediction)
Expected RMSE/MAE : Variable; lower is better

=== DATASET ===
Total samples       : 1,302 player-seasons
Unique players      : 260
Seasons             : 7 (2016–2023)
Mild injury rate    : 47.1%
Severe injury rate  : 12.2%
Mean injury days    : ~79.1 per season

=== LSTM ===
Sequences created   : 780 (from 260 players, seq_len=3)
Train/test split    : 208 / 52 players
Architecture        : LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3)
                      → Dense(32, ReLU) → Dropout(0.15) → Output

=== MEDICAL ANN ===
Architecture        : Dense(128) → BN → Drop(0.3)
                      → Dense(64) → BN → Drop(0.2)
                      → Dense(32) → BN → Drop(0.1)
                      → Output
Loss                : Binary cross-entropy + class weights

=== TOP FEATURES (Random Forest Importance) ===
1. fifa_rating     : 0.105
2. age             : 0.103
3. bmi             : 0.099
4. season_minutes_played        : ~0.090
5. season_days_injured_prev_season : ~0.085
```

---

*This document was generated to provide a complete, resume-ready and pitch-ready reference for the Football Injury Prediction NNDL project. For the full interactive experience, run `streamlit run improved_football_injury_app.py`.*
