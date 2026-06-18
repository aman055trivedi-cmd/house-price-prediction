# 🏠 House Price Prediction — Regression Analysis

An end-to-end machine learning regression project predicting house sale prices using the famous **Ames Housing Dataset** from Kaggle.

> Achieved **Top 20% on the Kaggle Leaderboard** using XGBoost with cross-validation tuning.

---

## 📌 About This Project

This was my most challenging project so far — the Ames Housing dataset has 79 features and a ton of missing values. Feature engineering took the most time but made the biggest difference. Log-transforming the target variable (SalePrice) was a key step that improved RMSE significantly.

---

## 🛠️ Tech Stack

- Python 3.x
- XGBoost
- scikit-learn (Ridge, Lasso, RandomForest, GradientBoosting)
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 📂 Project Structure

```
house-price-prediction/
│
├── house_price_prediction.py   # Main project code
├── eda_plots.png               # EDA visualizations
├── feature_importance.png      # XGBoost feature importances
├── actual_vs_predicted.png     # Actual vs predicted scatter plot
└── README.md
```

---

## ⚙️ How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Download dataset from Kaggle first:
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# Run the project
python house_price_prediction.py
```

---

## 📊 Models Compared

| Model               | RMSE     | R²     |
|---------------------|----------|--------|
| Ridge Regression    | ~$28,000 | ~0.83  |
| Lasso Regression    | ~$27,500 | ~0.84  |
| Random Forest       | ~$22,000 | ~0.89  |
| Gradient Boosting   | ~$20,000 | ~0.91  |
| **XGBoost**         | **~$18,500** | **~0.93** |

---

## 🔬 Key Steps

1. **EDA** — Distribution plots, scatter plots, correlation heatmap
2. **Feature Engineering** — Created `HouseAge`, `RemodAge`, `TotalSF`, `TotalBathrooms`
3. **Missing Value Imputation** — Median imputation for numerical columns
4. **Label Encoding** — For categorical features
5. **Log Transform** — Applied `log1p` to SalePrice to reduce skewness
6. **Model Training** — Compared 5 models
7. **Cross-Validation** — 5-fold CV on XGBoost
8. **Feature Importance** — Top 12 most important features visualized

---

## 💡 Key Learnings

- Log-transforming the target variable was the single biggest improvement
- Feature engineering (creating TotalSF) mattered more than model tuning
- XGBoost with `subsample` and `colsample_bytree` helped prevent overfitting

---

## 👤 Author

**Aman Trivedi**  
B.Tech ECE — PSIT Kanpur  
[LinkedIn](https://linkedin.com/in/aman-trivedi-764899343) | [Kaggle](https://kaggle.com/amantrivedi09) | [GitHub](https://github.com/aman055trivedi-cmd)
