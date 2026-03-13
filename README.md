# ⚽ Football Injury Prediction with Neural Networks & Deep Learning

Welcome to the Football Injury Prediction NNDL Project! 🧠💥

## What is this?
This project is a data-driven adventure into the world of football (soccer) injuries. Using real player data and advanced machine learning, we predict when a player might get injured, helping teams stay ahead of the game!

## Why Predict Injuries?
Injuries can change the course of a season. By predicting them, clubs can:
- Protect their star players
- Plan better strategies
- Save money on medical bills
- Win more matches!

## How Does It Work?
We use two powerful AI models:
- **LSTM (Long Short-Term Memory):** Like a football coach with a great memory, LSTM looks at a player’s history and spots patterns over time.
- **Medical ANN (Artificial Neural Network):** Inspired by medical experts, this model understands complex health data and predicts risks.

## What’s in the Data?
Our dataset includes:
- Player stats (minutes played, games, age, position)
- Injury records (days injured, previous injuries)
- Physical attributes (height, weight, BMI)
- FIFA ratings and more!

## The Magic Recipe
1. **Data Cleaning:** We tidy up the stats, fill in missing info, and get everything match-ready.
2. **Feature Engineering:** We create new features (like injury trends) to help our models learn better.
3. **Model Training:** LSTM and ANN learn from the data, just like players learn from practice.
4. **Prediction:** The models predict injury risk for each player and season.

## Why LSTM & Medical ANN?
- **LSTM** is perfect for time-series data (like player history).
- **Medical ANN** is great for health-related predictions.
- Other models (like CNNs) are better for images, but we need brains, not eyes!

## Results
- Accurate injury predictions (87.7% accuracy, 0.596 AUC on severe injury classification)
- Insights for coaches and medical staff
- A fun way to combine football and AI!

## Detailed Project Overview
For a complete technical breakdown — including all model architectures, metrics, feature importance rankings, resume-ready bullet points, and pitch content — see **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**.

## How to Use
1. Clone the repo
2. Install dependencies: `pip install streamlit pandas numpy scikit-learn matplotlib seaborn imbalanced-learn tensorflow`
3. Run the app: `streamlit run improved_football_injury_app.py`
4. Explore the predictions and visualizations

## Meet the Team
- Data Scientists ⚗️
- Football Fans ⚽
- AI Enthusiasts 🤖

## Want to Play?
Fork the repo, try new models, or use your own data. Let’s make football safer and smarter—together!

---

Made with ❤️ for football and science.
