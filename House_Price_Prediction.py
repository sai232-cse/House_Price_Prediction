import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Load your dataset
df = pd.read_csv('/content/archive.zip')

# 2. Select features (X) and target variable (y)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']]
y = df['Price']

# 3. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Evaluate the model performance
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: ${mae:.2f}")

# Example: Predict price for a new house
new_house = [[70000, 6, 7, 30000]] # Sample feature values
predicted_price = model.predict(new_house)
print(f"Predicted Price: ${predicted_price[0]:.2f}")
