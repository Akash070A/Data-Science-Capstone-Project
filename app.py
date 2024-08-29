import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:/Users/Akshu/Desktop/ML/car_details.csv'
car_data = pd.read_csv(file_path)

# First few rows
print(car_data.head())

# Summary statistics
print(car_data.describe())

# Data types and non-null counts
print(car_data.info())

missing_values = car_data.isnull().sum()

car_data_cleaned = car_data.dropna()
# Handling missing values
numerical_cols = ['year', 'selling_price', 'km_driven']
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

# Imputer for numerical columns
numerical_imputer = SimpleImputer(strategy='mean')
car_data[numerical_cols] = numerical_imputer.fit_transform(car_data[numerical_cols])

# Imputer for categorical columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
car_data[categorical_cols] = categorical_imputer.fit_transform(car_data[categorical_cols])

# One-Hot Encoding for categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical_data = encoder.fit_transform(car_data[categorical_cols])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_cols))

# Scaling numerical features
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(car_data[numerical_cols])
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_cols)

# Combine processed data
processed_data = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

# Features and target variable
X = processed_data.drop(columns=['selling_price'])  # All features except target
y = scaled_numerical_df['selling_price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Bagging': BaggingRegressor(random_state=42)
}

# Dictionary to store model performance
performance = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    performance[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': r2}

    print(f"{name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print("\n")

# Selecting the best model based on R-squared
best_model_name = max(performance, key=lambda k: performance[k]['R-squared'])
best_model = models[best_model_name]

print(f"The best model is {best_model_name} with R-squared: {performance[best_model_name]['R-squared']:.4f}")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"Best model {best_model_name} saved as 'best_model.pkl'")
# Load the saved model
loaded_model = joblib.load('best_model.pkl')

# Making predictions on new data (using X_test as an example)
new_predictions = loaded_model.predict(X_test)

# Display the first few predictions
print("Predictions on the test data:\n", new_predictions[:10])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict using the model
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.4f}")

# Display predictions
print("Predictions:\n", predictions)

sample_data = car_data.sample(n=20, random_state=42)
# Impute missing values
sample_data[numerical_cols] = numerical_imputer.transform(sample_data[numerical_cols])
sample_data[categorical_cols] = categorical_imputer.transform(sample_data[categorical_cols])

# Encode categorical variables
encoded_sample_data = encoder.transform(sample_data[categorical_cols])
encoded_sample_df = pd.DataFrame(encoded_sample_data, columns=encoder.get_feature_names_out(categorical_cols))

# Ensure the encoded data columns match the training data
encoded_sample_df = encoded_sample_df.reindex(columns=encoder.get_feature_names_out(categorical_cols), fill_value=0)

# Scale numerical variables
scaled_sample_data = scaler.transform(sample_data[numerical_cols])
scaled_sample_df = pd.DataFrame(scaled_sample_data, columns=numerical_cols)

# Combine the processed numerical and encoded categorical data
processed_sample_data = pd.concat([scaled_sample_df, encoded_sample_df], axis=1)

# Ensure the final data frame has the same columns as the training data
processed_sample_data = processed_sample_data.reindex(columns=X_train.columns, fill_value=0)

# Load the saved model
loaded_model = joblib.load('best_model.pkl')

# Predict using the loaded model
sample_predictions = loaded_model.predict(processed_sample_data)

# Display the predictions
print("Predictions on the sample data:\n", sample_predictions)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Compare predictions with actual values (assuming target column is still present in sample_data)
actual_values = sample_data['selling_price'].values

# Calculate evaluation metrics
mae = mean_absolute_error(actual_values, sample_predictions)
mse = mean_squared_error(actual_values, sample_predictions)
rmse = mse ** 0.5
r2 = r2_score(actual_values, sample_predictions)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
