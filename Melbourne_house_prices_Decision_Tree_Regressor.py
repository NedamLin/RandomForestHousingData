import pandas as pd
melbourne_file_path = r"C:\Users\johnd\Documents\PythonProjects\melb_data.csv"  # Change to your path that has the data
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
print(melbourne_data.describe())
print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price

# selecting the features/predictors we want for out model
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

# setting them as our predictors i,e x variable
X = melbourne_data[melbourne_features]

# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())

print("The predictions are")
print(melbourne_model.predict(X.head()))

#################################################################################################################

#Model Validation Mean Absoloute Error
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)

print('Absoloute Mean Error')
print(mae)