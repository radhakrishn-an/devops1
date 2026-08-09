from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load Boston Housing dataset from OpenML
boston = fetch_openml(
    name="boston",
    version=1,
    as_frame=True
)

# Input features
X = boston.data

# Target house prices
y = boston.target.astype(float)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.joblib")

print("Model trained successfully!")
print("Model saved as model.joblib")
print("Number of features:", X.shape[1])
