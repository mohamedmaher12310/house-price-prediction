# 🏡 House Price Prediction using LassoCV, XGBoost, and LightGBM

This project applies machine learning regression techniques to predict house prices using the Ames Housing dataset. The final model is a blend of three powerful models — LassoCV, XGBoost, and LightGBM — and includes thorough preprocessing, feature engineering, and evaluation using MSE and R² scores.

---

## 📦 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (LassoCV, KFold, evaluation metrics)
- XGBoost
- LightGBM
- Seaborn (for data visualization)

---

## 🧠 Key Steps

### 1. 📥 Data Loading
- `train.csv` and `test.csv` are loaded.
- Target variable `SalePrice` is log-transformed for better regression performance.

### 2. 🔍 Data Cleaning & Encoding
- Missing values are filled:
  - Categorical: `'None'`
  - Numerical: `0`
- Categorical features are label-encoded using `LabelEncoder`.

### 3. 🛠️ Feature Engineering
Added new features:
- `TotalSF`: Total square footage.
- `TotalBath`: Combined full and half bathrooms.
- `Age`: Age of the house.
- `RemodAge`: Years since last remodel.
- `HasPool`: Binary flag for presence of a pool.

### 4. 🚂 Modeling
Three regression models are trained:
- **LassoCV** (with cross-validation)
- **XGBoost**
- **LightGBM**

Final prediction is a **weighted average**:
```python
final_prediction = 0.3 * lasso + 0.35 * xgboost + 0.35 * lightgbm

5. 📊 Model Evaluation
Dataset split into training and validation sets (80/20).

Evaluation metrics:

MSE (Mean Squared Error)

R² Score

📈 Results
Metric	Value
MSE (Validation)	Displayed in console
R² Score	Displayed in console

(These scores help evaluate how well the blended model performs on unseen data.)