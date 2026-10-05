import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def generate_or_load_dataset(filepath="data/house_data.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        df = pd.read_csv(filepath)
    else:
        np.random.seed(42)
        n = 1200
        sqft = np.random.normal(loc=1800, scale=600, size=n).clip(500, 5000).astype(int)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.35, 0.35, 0.15, 0.05], size=n)
        bathrooms = np.clip(np.round(bedrooms * 0.7 + np.random.choice([0, 1], p=[0.4, 0.6], size=n)), 1, 5).astype(int)
        floors = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1], size=n)
        parking = np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.25, 0.05], size=n)
        location_rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1], size=n)
        property_age = np.random.randint(0, 40, size=n)

        price = (
            12.0
            + 0.042 * sqft
            + 4.8 * bedrooms
            + 3.2 * bathrooms
            + 2.8 * floors
            + 3.5 * parking
            + 8.5 * location_rating
            - 0.45 * property_age
            + np.random.normal(0, 5.0, size=n)
        )

        df = pd.DataFrame({
            "Area_SqFt": sqft,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "Parking": parking,
            "Location_Rating": location_rating,
            "Property_Age": property_age,
            "Price_Lakhs": np.round(price, 2)
        })
        df.to_csv(filepath, index=False)
    return df

def preprocess_and_train():
    df = generate_or_load_dataset()
    df = df.drop_duplicates()
    df = df.dropna()

    X = df[["Area_SqFt", "Bedrooms", "Bathrooms", "Floors", "Parking", "Location_Rating", "Property_Age"]]
    y = df["Price_Lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=" * 45)
    print("       MODEL EVALUATION METRICS")
    print("=" * 45)
    print(f"Mean Absolute Error (MAE) : {mae:.3f} Lakhs")
    print(f"Mean Squared Error (MSE)  : {mse:.3f}")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.3f} Lakhs")
    print(f"R2 Score                  : {r2:.4f}")
    print("=" * 45)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/house_price_model.pkl")
    print("Model serialized and saved to 'model/house_price_model.pkl'.")

if __name__ == "__main__":
    preprocess_and_train()
