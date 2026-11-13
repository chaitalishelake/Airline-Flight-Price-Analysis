
# AIRLINE TICKET PRICE PREDICTION
# Author: Chaitali Shelake

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("AIRLINE TICKET PRICE PREDICTION")
print("Developed by Chaitali Shelake")

# 1. Load dataset
df = pd.read_csv("airlines_flights_data.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Remove unnecessary columns
df = df.drop(columns=["index", "flight"], errors="ignore")

# 3. Handle missing values
df = df.dropna(subset=["price"])

# 4. Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# 5. Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 6. Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 7. Create Machine Learning model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 8. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining the Machine Learning model...")

# 9. Train model
model.fit(X_train, y_train)

print("Model training completed!")

# 10. Make predictions
predictions = model.predict(X_test)

# 11. Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("--------------------------")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 12. Save trained model
joblib.dump(model, "airline_price_model.pkl")

print("\nTrained model saved successfully!")
print("File: airline_price_model.pkl")

# 13. Sample prediction
sample = X_test.iloc[[0]]
predicted_price = model.predict(sample)[0]

print("\nSAMPLE PREDICTION")
print("Predicted Ticket Price: ₹", round(predicted_price, 2))
print("Actual Ticket Price: ₹", round(y_test.iloc[0], 2))

print("\nPROJECT COMPLETED SUCCESSFULLY!")
