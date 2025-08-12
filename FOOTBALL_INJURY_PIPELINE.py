import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Input
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

st.set_page_config(page_title="Football Injury Prediction", layout="wide")
st.title("Football Player Injury Prediction - Neural Networks")

# Sidebar
st.sidebar.title("Neural Network Models")
page = st.sidebar.radio(
    "Select Model",
    ["Dataset Overview", "ANN Model", "LSTM Model", "Autoencoder", "1D CNN Model", "RBM Model", "Model Comparison"]
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    # Create binary classification target
    df['high_injury_risk'] = (df['season_days_injured'] > df['season_days_injured'].median()).astype(int)
    return df

df = load_data()

# Core features from EDA analysis
CORE_FEATURES = ['age', 'bmi', 'fifa_rating', 'season_minutes_played', 'pace', 'physic']
TARGET_REGRESSION = 'season_days_injured'
TARGET_CLASSIFICATION = 'high_injury_risk'

def preprocess_data(df, features, target, task_type='regression'):
    # Remove rows with missing values in selected features and target
    clean_df = df.dropna(subset=features + [target])
    
    X = clean_df[features].values
    y = clean_df[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, clean_df

###########################################
# ANN MODEL
###########################################

def build_ann_model(input_dim, task_type='regression'):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid' if task_type == 'classification' else 'linear')
    ])
    
    if task_type == 'classification':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def train_ann(X_train, y_train, task_type='regression'):
    model = build_ann_model(X_train.shape[1], task_type)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    return model, history

###########################################
# LSTM MODEL
###########################################

def create_sequences_simple(df, features, target, sequence_length=3):
    # Create sequences based on player age progression (simulated time series)
    df_sorted = df.sort_values('age').reset_index(drop=True)
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(df_sorted) - sequence_length + 1):
        # Create sequence of features
        seq = df_sorted[features].iloc[i:i+sequence_length].values
        # Target is the injury days of the last player in sequence
        target_val = df_sorted[target].iloc[i+sequence_length-1]
        
        X_sequences.append(seq)
        y_sequences.append(target_val)
    
    return np.array(X_sequences), np.array(y_sequences)

def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(X_train, y_train):
    model = build_lstm_model(X_train.shape[1], X_train.shape[2])
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    return model, history

###########################################
# AUTOENCODER
###########################################

def build_autoencoder(input_dim, encoding_dim=8):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu', name='encoded')(encoded)
    
    # Decoder
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def train_autoencoder(X_train):
    autoencoder, encoder = build_autoencoder(X_train.shape[1])
    
    history = autoencoder.fit(
        X_train, X_train,  # Reconstruct input
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )
    
    return autoencoder, encoder, history

def detect_anomalies(autoencoder, X, threshold_percentile=95):
    reconstructed = autoencoder.predict(X, verbose=0)
    mse_loss = np.mean(np.power(X - reconstructed, 2), axis=1)
    threshold = np.percentile(mse_loss, threshold_percentile)
    anomalies = mse_loss > threshold
    
    return mse_loss, anomalies, threshold

###########################################
# 1D CNN MODEL
###########################################

def build_1d_cnn_model(input_shape):
    model = Sequential([
        # Reshape for CNN
        tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        
        # 1D Convolution layers
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_1d_cnn(X_train, y_train):
    model = build_1d_cnn_model(X_train.shape[1:])
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    return model, history

###########################################
# RBM MODEL
###########################################

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.b_visible = np.zeros(n_visible)
        self.b_hidden = np.zeros(n_hidden)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sample_hidden(self, visible):
        hidden_prob = self.sigmoid(np.dot(visible, self.W) + self.b_hidden)
        hidden_sample = (hidden_prob > np.random.random(hidden_prob.shape)).astype(np.float32)
        return hidden_prob, hidden_sample
    
    def sample_visible(self, hidden):
        visible_prob = self.sigmoid(np.dot(hidden, self.W.T) + self.b_visible)
        visible_sample = (visible_prob > np.random.random(visible_prob.shape)).astype(np.float32)
        return visible_prob, visible_sample
    
    def contrastive_divergence(self, visible_data, k=1):
        # Positive phase
        pos_hidden_prob, pos_hidden_sample = self.sample_hidden(visible_data)
        
        # Negative phase
        neg_visible_prob = visible_data
        for _ in range(k):
            neg_hidden_prob, neg_hidden_sample = self.sample_hidden(neg_visible_prob)
            neg_visible_prob, _ = self.sample_visible(neg_hidden_sample)
        
        neg_hidden_prob, _ = self.sample_hidden(neg_visible_prob)
        
        # Update weights and biases
        self.W += self.learning_rate * (np.dot(visible_data.T, pos_hidden_prob) - 
                                       np.dot(neg_visible_prob.T, neg_hidden_prob))
        self.b_visible += self.learning_rate * np.mean(visible_data - neg_visible_prob, axis=0)
        self.b_hidden += self.learning_rate * np.mean(pos_hidden_prob - neg_hidden_prob, axis=0)
        
        # Reconstruction error
        error = np.mean((visible_data - neg_visible_prob) ** 2)
        return error
    
    def fit(self, X, epochs=50):
        # Binarize input data for RBM
        X_binary = (X > np.mean(X, axis=0)).astype(np.float32)
        
        errors = []
        batch_size = 32
        
        for epoch in range(epochs):
            epoch_error = 0
            n_batches = 0
            
            for i in range(0, len(X_binary), batch_size):
                batch = X_binary[i:i+batch_size]
                error = self.contrastive_divergence(batch)
                epoch_error += error
                n_batches += 1
            
            avg_error = epoch_error / n_batches
            errors.append(avg_error)
        
        return errors
    
    def transform(self, X):
        X_binary = (X > np.mean(X, axis=0)).astype(np.float32)
        hidden_prob, _ = self.sample_hidden(X_binary)
        return hidden_prob

def train_rbm_with_ann(X_train, y_train, n_hidden=16):
    # Train RBM
    rbm = RBM(X_train.shape[1], n_hidden)
    rbm_errors = rbm.fit(X_train, epochs=30)
    
    # Extract features
    X_train_rbm = rbm.transform(X_train)
    
    # Train ANN on RBM features
    ann_model = Sequential([
        Dense(32, activation='relu', input_shape=(n_hidden,)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    ann_history = ann_model.fit(
        X_train_rbm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    return rbm, ann_model, rbm_errors, ann_history

###########################################
# VISUALIZATION FUNCTIONS
###########################################

def plot_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    if 'mae' in history.history:
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title(f'{title} - MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
    elif 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title(f'{title} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
    
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    return fig

def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.grid(True)
    return fig

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

###########################################
# STREAMLIT PAGES
###########################################

if page == "Dataset Overview":
    st.header("Football Injury Dataset Overview")
    
    st.subheader("Dataset Information")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Features: {len(df.columns)}")
    st.write(f"Average Injury Days: {df['season_days_injured'].mean():.1f}")
    st.write(f"High Risk Players: {df['high_injury_risk'].sum()} ({df['high_injury_risk'].mean()*100:.1f}%)")
    
    # Display key statistics
    st.subheader("Key Features Statistics")
    st.dataframe(df[CORE_FEATURES + [TARGET_REGRESSION]].describe())
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    corr_data = df[CORE_FEATURES + [TARGET_REGRESSION]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, ax=ax)
    st.pyplot(fig)
    
    # Target distribution
    st.subheader("Target Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(df['season_days_injured'], bins=30, alpha=0.7)
    ax1.set_title('Days Injured Distribution')
    ax1.set_xlabel('Days Injured')
    
    ax2.bar(['Low Risk', 'High Risk'], df['high_injury_risk'].value_counts())
    ax2.set_title('Injury Risk Distribution')
    
    st.pyplot(fig)

elif page == "ANN Model":
    st.header("Artificial Neural Network (ANN)")
    st.write("Feedforward neural network for injury prediction")
    
    task = st.radio("Task Type", ["Regression", "Classification"])
    
    if st.button("Train ANN Model"):
        target = TARGET_REGRESSION if task == "Regression" else TARGET_CLASSIFICATION
        task_type = task.lower()
        
        with st.spinner("Training ANN..."):
            X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(df, CORE_FEATURES, target, task_type)
            model, history = train_ann(X_train, y_train, task_type)
            
            y_pred = model.predict(X_test, verbose=0)
            
            if task_type == 'classification':
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                accuracy = np.mean(y_pred_binary == y_test)
                st.success(f"Training completed! Accuracy: {accuracy:.4f}")
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred_binary)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            else:
                y_pred = y_pred.flatten()
                metrics = evaluate_regression(y_test, y_pred)
                st.success(f"Training completed! RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}")
                
                # Prediction plot
                fig = plot_predictions(y_test, y_pred, "ANN Predictions")
                st.pyplot(fig)
        
        # Training history
        st.subheader("Training History")
        fig = plot_history(history, "ANN")
        st.pyplot(fig)

elif page == "LSTM Model":
    st.header("Long Short-Term Memory (LSTM)")
    st.write("Sequential neural network for temporal pattern analysis")
    
    if st.button("Train LSTM Model"):
        with st.spinner("Creating sequences and training LSTM..."):
            # Create sequences
            X_seq, y_seq = create_sequences_simple(df, CORE_FEATURES, TARGET_REGRESSION, 3)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
            
            # Train model
            model, history = train_lstm(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test, verbose=0).flatten()
            metrics = evaluate_regression(y_test, y_pred)
            
            st.success(f"Training completed! RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}")
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training History")
            fig = plot_history(history, "LSTM")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Predictions")
            fig = plot_predictions(y_test, y_pred, "LSTM Predictions")
            st.pyplot(fig)

elif page == "Autoencoder":
    st.header("Autoencoder")
    st.write("Unsupervised feature learning and anomaly detection")
    
    if st.button("Train Autoencoder"):
        with st.spinner("Training autoencoder..."):
            X_train, X_test, _, _, scaler, clean_df = preprocess_data(df, CORE_FEATURES, TARGET_REGRESSION)
            
            # Train autoencoder
            autoencoder, encoder, history = train_autoencoder(X_train)
            
            # Detect anomalies
            X_all = np.vstack([X_train, X_test])
            mse_loss, anomalies, threshold = detect_anomalies(autoencoder, X_all)
            
            st.success(f"Training completed! Detected {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.1f}%)")
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training History")
            fig = plot_history(history, "Autoencoder")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Anomaly Detection")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(mse_loss, bins=50, alpha=0.7)
            ax.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({np.percentile(mse_loss, 95):.4f})')
            ax.set_xlabel('Reconstruction Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Reconstruction Error Distribution')
            ax.legend()
            st.pyplot(fig)
        
        # Encoded features visualization
        st.subheader("Learned Feature Representation")
        encoded_features = encoder.predict(X_all, verbose=0)
        
        # Use first 2 dimensions for visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(encoded_features[:, 0], encoded_features[:, 1], 
                           c=clean_df[TARGET_REGRESSION].values, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Encoded Feature 1')
        ax.set_ylabel('Encoded Feature 2')
        ax.set_title('Encoded Feature Space')
        plt.colorbar(scatter, label='Days Injured')
        st.pyplot(fig)

elif page == "1D CNN Model":
    st.header("1D Convolutional Neural Network")
    st.write("Convolutional layers for pattern detection in player attributes")
    
    if st.button("Train 1D CNN Model"):
        with st.spinner("Training 1D CNN..."):
            X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(df, CORE_FEATURES, TARGET_REGRESSION)
            
            # Train model
            model, history = train_1d_cnn(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test, verbose=0).flatten()
            metrics = evaluate_regression(y_test, y_pred)
            
            st.success(f"Training completed! RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}")
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training History")
            fig = plot_history(history, "1D CNN")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Predictions")
            fig = plot_predictions(y_test, y_pred, "1D CNN Predictions")
            st.pyplot(fig)

elif page == "RBM Model":
    st.header("Restricted Boltzmann Machine + ANN")
    st.write("Unsupervised feature learning with RBM followed by supervised ANN")
    
    if st.button("Train RBM + ANN Model"):
        with st.spinner("Training RBM and ANN..."):
            X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(df, CORE_FEATURES, TARGET_REGRESSION)
            
            # Train RBM + ANN
            rbm, ann_model, rbm_errors, ann_history = train_rbm_with_ann(X_train, y_train)
            
            # Transform test data and predict
            X_test_rbm = rbm.transform(X_test)
            y_pred = ann_model.predict(X_test_rbm, verbose=0).flatten()
            metrics = evaluate_regression(y_test, y_pred)
            
            st.success(f"Training completed! RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R²']:.4f}")
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RBM Training Progress")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(rbm_errors)
            ax.set_title('RBM Reconstruction Error')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Error')
            ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            st.subheader("ANN Training History")
            fig = plot_history(ann_history, "RBM + ANN")
            st.pyplot(fig)
        
        st.subheader("Predictions")
        fig = plot_predictions(y_test, y_pred, "RBM + ANN Predictions")
        st.pyplot(fig)

elif page == "Model Comparison":
    st.header("Model Comparison")
    st.write("Compare all neural network models on injury prediction task")
    
    if st.button("Run Full Comparison"):
        results = {}
        
        with st.spinner("Training all models..."):
            X_train, X_test, y_train, y_test, scaler, _ = preprocess_data(df, CORE_FEATURES, TARGET_REGRESSION)
            
            # ANN
            st.write("Training ANN...")
            ann_model, _ = train_ann(X_train, y_train, 'regression')
            ann_pred = ann_model.predict(X_test, verbose=0).flatten()
            results['ANN'] = evaluate_regression(y_test, ann_pred)
            
            # LSTM
            st.write("Training LSTM...")
            X_seq, y_seq = create_sequences_simple(df, CORE_FEATURES, TARGET_REGRESSION, 3)
            X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
            lstm_model, _ = train_lstm(X_train_seq, y_train_seq)
            lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
            results['LSTM'] = evaluate_regression(y_test_seq, lstm_pred)
            
            # Autoencoder + Regression
            st.write("Training Autoencoder...")
            autoencoder, encoder, _ = train_autoencoder(X_train)
            X_train_encoded = encoder.predict(X_train, verbose=0)
            X_test_encoded = encoder.predict(X_test, verbose=0)
            
            ae_model = Sequential([
                Dense(16, activation='relu', input_shape=(X_train_encoded.shape[1],)),
                Dense(1, activation='linear')
            ])
            ae_model.compile(optimizer='adam', loss='mse')
            ae_model.fit(X_train_encoded, y_train, epochs=50, verbose=0)
            ae_pred = ae_model.predict(X_test_encoded, verbose=0).flatten()
            results['Autoencoder'] = evaluate_regression(y_test, ae_pred)
            
            # 1D CNN
            st.write("Training 1D CNN...")
            cnn_model, _ = train_1d_cnn(X_train, y_train)
            cnn_pred = cnn_model.predict(X_test, verbose=0).flatten()
            results['1D CNN'] = evaluate_regression(y_test, cnn_pred)
            
            # RBM + ANN
            st.write("Training RBM + ANN...")
            rbm, rbm_ann_model, _, _ = train_rbm_with_ann(X_train, y_train)
            X_test_rbm = rbm.transform(X_test)
            rbm_pred = rbm_ann_model.predict(X_test_rbm, verbose=0).flatten()
            results['RBM + ANN'] = evaluate_regression(y_test, rbm_pred)
        
        # Display comparison
        st.subheader("Model Performance Comparison")
        
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        comparison_df['Rank'] = comparison_df['R²'].rank(ascending=False)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        st.dataframe(comparison_df)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = comparison_df.index
        rmse_values = comparison_df['RMSE']
        r2_values = comparison_df['R²']
        
        ax1.bar(models, rmse_values)
        ax1.set_title('RMSE Comparison (Lower is Better)')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(models, r2_values)
        ax2.set_title('R² Score Comparison (Higher is Better)')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model summary
        best_model = comparison_df.index[0]
        best_r2 = comparison_df.loc[best_model, 'R²']
        
        st.success(f"🏆 Best Model: **{best_model}** with R² Score: **{best_r2:.4f}**")
        
        st.subheader("Key Insights")
        st.write(f"""
        **Model Performance Analysis:**
        
        1. **Best Performer**: {best_model} achieved the highest R² score of {best_r2:.4f}
        2. **RMSE Range**: {comparison_df['RMSE'].min():.2f} - {comparison_df['RMSE'].max():.2f} days
        3. **R² Range**: {comparison_df['R²'].min():.4f} - {comparison_df['R²'].max():.4f}
        
        **Neural Network Insights:**
        - **ANN**: Baseline feedforward network for injury prediction
        - **LSTM**: Captures temporal patterns in player progression  
        - **Autoencoder**: Learns compressed feature representations
        - **1D CNN**: Detects local patterns in player attributes
        - **RBM**: Unsupervised feature learning with probabilistic modeling
        
        **Dataset Characteristics:**
        - Average injury duration: {df['season_days_injured'].mean():.1f} days
        - Players analyzed: {len(df)} records
        - Core features: {len(CORE_FEATURES)} attributes
        """)

# Run the app with: streamlit run football_injury_pipeline.py