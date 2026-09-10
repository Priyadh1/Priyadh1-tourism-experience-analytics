# Priyadh1-tourism-experience-analytics
# 🌍 Tourism Experience Analytics

### Classification, Prediction, and Recommendation System

A machine learning platform that analyzes tourist behavior to predict trip ratings, classify visit modes, and recommend attractions — built on 52,930 real tourism transactions across 9 relational tables.

## 🔗 Live App
**[Launch the App →](https://priyadh1-tourism-experience-analytics.streamlit.app)**

## 📌 Project Overview
This project helps a tourism platform understand and personalize the traveler experience through three integrated ML components:

- **⭐ Regression** — Predicts the rating a user is likely to give an attraction
- **🧳 Classification** — Predicts a user's visit mode (Business, Couples, Family, Friends, Solo)
- **🧭 Recommendation System** — Suggests attractions using item-based collaborative filtering

## 🗂️ Dataset
9 relational tables merged into a single dataset of 52,930 transactions:
`Transaction`, `User`, `City`, `Country`, `Region`, `Continent`, `Item`, `Type`, `Mode`

## 🛠️ Tech Stack
- **Data Processing:** Python, Pandas, NumPy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git, GitHub

## 📊 Model Performance

| Task | Best Model | Test Metric |
|---|---|---|
| Classification (Visit Mode) | XGBoost (Tuned) | 42.6% Accuracy |
| Regression (Rating) | Random Forest (Tuned) | R² = 12.8%, RMSE = 0.91 |
| Recommendation | Cosine Similarity (Item-based CF) | — |

Both final models were selected after comparing 3 algorithms per task and tuning via `RandomizedSearchCV` with 3-fold cross-validation, improving classification accuracy from a 28.1% baseline and regression R² from 4.0%.

## 🚀 App Features
- **✈️ Plan My Trip** — Cascading location picker with real-time rating and visit-mode predictions
- **📊 Explore Insights** — Interactive, filterable data visualizations
- **🧭 Discover Attractions** — Personalized attraction recommendations
- **🤖 Model Performance** — Transparent model comparison and evaluation

## 👤 Author
**Priyadharshini Murugan**
