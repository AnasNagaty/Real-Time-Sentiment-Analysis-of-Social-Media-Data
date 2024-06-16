import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml

# Load the Ames Housing dataset
housing = fetch_openml(name="house_prices", as_frame=True)
X = housing.data
y = housing.target

# Simulate additional features
np.random.seed(0)
X['ECONOMIC_INDEX'] = np.random.normal(loc=0, scale=1, size=len(X))
X['CRIME_RATE'] = np.random.normal(loc=0, scale=1, size=len(X))

# Create a smaller subset
subset_size = 50  # Adjust the size as needed
X_subset = X.sample(n=subset_size, random_state=42)
y_subset = y[X_subset.index]

# Drop 'PoolQC' feature if it's not needed or has too many missing values
X_subset.drop(columns=['PoolQC'], inplace=True, errors='ignore')

# Separate numerical and categorical features after dropping 'PoolQC'
numerical_features = X_subset.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_subset.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a preprocessing and modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', RandomForestRegressor())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# Define the grid of hyperparameters
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [5, 10]
}

# Model training and hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
