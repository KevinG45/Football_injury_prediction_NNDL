# Football Injury Prediction NNDL - Streamlit App Screenshots

This folder contains comprehensive screenshots of the Football Injury Prediction NNDL Streamlit application, showcasing all features and capabilities.

## Application Overview

The app demonstrates a complete machine learning pipeline for predicting football player injuries using multiple models and techniques. It features:

- **Different Target Values**: Mild injury (>7 days), Severe injury (>28 days), Regression (days injured)
- **Different Split Methods**: Temporal split, Player split, Stratified random split
- **Multiple Optimizers**: Adam, SGD, RMSprop
- **Various Models**: Random Forest, Logistic Regression, LSTM, Medical ANN

## Screenshots Captured

### 1. Dataset Overview (`01_dataset_overview.png`)
- Initial landing page showing dataset statistics
- Data distribution visualization (injury days histogram)
- Sample data table
- System features overview

### 2. Methodology Pages (`02_methodology*.png`)
- **`02_methodology.png`**: Data leakage prevention explanation, injury classification thresholds, temporal validation with Temporal Split demonstration
- **`02_methodology_player_split.png`**: Same page showing Player Split method results for comparison

### 3. Baseline Models (`03_baseline_models*.png`)
- **`03_baseline_models_config.png`**: Configuration interface showing target type, split method, SMOTE option, and test size slider
- **`03_baseline_models_results.png`**: Results for Severe Injury Classification showing model performance metrics, confusion matrix, and ROC curves
- **`03_baseline_models_mild_injury.png`**: Results for Mild Injury Classification target, demonstrating different performance metrics (AUC: 0.641 vs 0.598)

### 4. LSTM Model (`04_lstm*.png`)
- **`04_lstm_config.png`**: Basic LSTM configuration showing sequence length and target selection
- **`04_lstm_config_expanded.png`**: Expanded configuration showing:
  - Architecture parameters (LSTM units, dropout rate)
  - Optimizer selection (Adam, RMSprop, SGD)
  - Learning rate options
- **`04_lstm_results.png`**: LSTM training results with:
  - Classification performance metrics
  - ROC curve visualization
  - Training progress charts (loss and accuracy)
  - Model architecture summary

### 5. Medical ANN (`05_medical_ann*.png`)
- **`05_medical_ann_config.png`**: Comprehensive configuration showing:
  - Target and split method selection
  - Model Architecture: Hidden layer sizes, dropout rates, batch normalization
  - Training Configuration: Optimizer (RMSprop shown), learning rate, epochs, batch size
- **`05_medical_ann_results.png`**: Medical ANN results featuring:
  - Performance metrics table
  - Medical-specific metrics (Sensitivity: 0.143, Specificity: 0.793)
  - Training analysis with loss/accuracy curves
  - Confusion matrix visualization

### 6. Model Comparison (`06_model_comparison.png`)
- Comparative analysis of all trained models
- Performance analysis with expected ranges for medical data
- Methodology validation showing split method and SMOTE application
- Best practices interpretation

### 7. Feature Importance (`07_feature_importance*.png`)
- **`07_feature_importance_initial.png`**: Initial interface for feature importance analysis with target selection
- **`07_feature_importance_results.png`**: Complete feature importance analysis showing:
  - Top features ranking table
  - Feature importance visualization plot
  - Clinical interpretation of key features (fifa_rating, age, bmi)

## Key Features Demonstrated

### Multiple Target Values
- **Severe Injury Classification (>28 days)**: AUC ~0.598
- **Mild Injury Classification (>7 days)**: AUC ~0.641 (better performance)
- **Regression (Days Injured)**: Available for continuous prediction

### Split Methods Comparison
- **Temporal Split**: Uses 2019-2022 for training, 2023 for testing (prevents data leakage)
- **Player Split**: 208 train players, 52 test players (tests generalization to new players)
- **Random Split**: Traditional stratified split (may allow temporal leakage)

### Optimizer Options
All models support three optimizers:
- **Adam**: Adaptive moment estimation (default choice)
- **RMSprop**: Root mean square propagation
- **SGD**: Stochastic gradient descent with momentum

### Data Leakage Prevention
- SMOTE applied only to training data
- Feature scaling fitted only on training data
- Temporal/player-based splits prevent information leakage

### Medical Domain Adaptation
- Clinically relevant injury thresholds
- Medical performance interpretation
- Sensitivity/specificity focus for screening
- Class balancing for rare severe injuries

## Technical Stack Showcased
- **Frontend**: Streamlit web application with interactive UI
- **Backend**: Python ML/DL pipeline
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Deep Learning**: TensorFlow/Keras (LSTM, Medical ANN)
- **Visualization**: Matplotlib, Seaborn
- **Class Balancing**: Imbalanced-learn (SMOTE)

## Total Screenshots: 14
The complete collection provides a comprehensive view of the application's capabilities, different configurations, and results across all model types and target variables.