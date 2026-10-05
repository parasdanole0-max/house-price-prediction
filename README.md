# 🏠 House Price Prediction (Machine Learning End-to-End)

An end-to-end Machine Learning project that predicts residential house prices using Linear Regression and an interactive Streamlit UI.

## 📊 Features
- `Area_SqFt`: Total carpet area in square feet
- `Bedrooms`: Number of bedrooms (1 to 5)
- `Bathrooms`: Number of bathrooms (1 to 5)
- `Floors`: Total floors in property (1 to 3)
- `Parking`: Available parking spaces (0 to 3)
- `Location_Rating`: Location quality score (1 to 5)
- `Property_Age`: Age of construction in years

## ⚙️ Model Evaluation Results
- **MAE (Mean Absolute Error):** 3.84 Lakhs
- **MSE (Mean Squared Error):** 22.93
- **RMSE (Root Mean Squared Error):** 4.79 Lakhs
- **R² Score:** 0.9750 (97.5% of variance explained)

## 🚀 How to Run
1. Activate virtual environment:
   `.\venv\Scripts\Activate.ps1`
2. Install packages:
   `pip install -r requirements.txt`
3. Train model:
   `python train.py`
4. Launch Streamlit UI:
   `python -m streamlit run app.py`

## 🖼️ Application Screenshots
- `screenshots/app_ui_overview.png`
- `screenshots/prediction_output.png`
