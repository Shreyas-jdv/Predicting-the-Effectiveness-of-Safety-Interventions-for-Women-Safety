import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")


# Simulate dataset
np.random.seed(42)  # For reproducibility

# Number of samples
n = 100

# Features
intervention_types = ['Awareness Campaign', 'CCTV Installation', 'Street Lighting', 'Police Patrols']
data = {
    'Intervention_Type': np.random.choice(intervention_types, n),
    'Budget_in_USD': np.random.randint(5000, 50000, n),
    'Duration_in_Months': np.random.randint(6, 36, n),
    'People_Reached': np.random.randint(100, 10000, n),
    'Effectiveness_Score': np.random.randint(50, 100, n)  # Target variable
}

df = pd.DataFrame(data)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Encode categorical variable using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Intervention_Type'], drop_first=True)

print("\nData after encoding:")
print(df_encoded.head())


# Pairplot to see relationships
sns.pairplot(df, hue='Intervention_Type')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot for Intervention Types vs Effectiveness Score
plt.figure(figsize=(8,6))
sns.boxplot(x='Intervention_Type', y='Effectiveness_Score', data=df)
plt.title('Effectiveness Score by Intervention Type')
plt.show()


# Define feature matrix X and target vector y
X = df_encoded.drop('Effectiveness_Score', axis=1)
y = df_encoded['Effectiveness_Score']

print("\nFeatures used for modeling:")
print(X.columns)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nModel Coefficients:")
print(coefficients)


# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 ): {r2:.2f}")


# Scatter plot of actual vs predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Effectiveness Score')
plt.ylabel('Predicted Effectiveness Score')
plt.title('Actual vs Predicted Effectiveness Score')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Effectiveness Score')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()


