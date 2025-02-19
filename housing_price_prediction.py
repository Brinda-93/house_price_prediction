# Housing price prediction
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Loading the dataset,i.e., california housing dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()

# Taking the dependent and the independent variables separately
df_x = pd.DataFrame(california.data, columns = california.feature_names)
df_y = pd.DataFrame(california.target)

# Creating an instance of the model
reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.15, random_state = 42)

# Training the model
reg.fit(x_train, y_train)

# This shows how much the dependent variables change if the independent variable change by 1
print("Coefficients:", reg.coef_)

# predicting the price as array
a = reg.predict(x_test)

# Finding the Mean Square Error
mse = mean_squared_error(y_test, a)
print(f"Mean Squared Error: {mse}")
