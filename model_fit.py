import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the dataset
file_path = 'data/merged.csv'
data = pd.read_csv(file_path)

# Calculate the correlation matrix and extract the 'energy' correlations
#correlation_matrix = data.corr()
#energy_correlations = correlation_matrix['energy'].sort_values(ascending=False)

# Preparing the data for modeling
X = data.drop(['energy', 'time'], axis=1)  # Features
y = data['energy']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'LinearRegression': make_pipeline(StandardScaler(), LinearRegression()),
    'RandomForestRegressor': make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42)),
    'GradientBoostingRegressor': make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
}

# Dictionary to hold the results
model_results = {}

# Train and evaluate each model
for name, model in models.items():
    # Training the model
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Storing the results
    model_results[name] = {'MSE': mse, 'R2': r2}

print(model_results)

#model = models['LinearRegression']
#model = models['RandomForestRegressor']
model = models['GradientBoostingRegressor']
data['predicted_energy'] = model.predict(X)

# Calculating the difference between actual and predicted values
data['difference'] = data['energy'] - data['predicted_energy']

data['relative_error'] = data['difference'] / data['energy']

# Selecting the required columns to display
results_columns = ['predicted_energy', 'energy', 'difference', 'relative_error']
results = data[results_columns]

print(results)
