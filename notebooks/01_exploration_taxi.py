import matplotlib.pyplot as plt
import pandas as pd

import common

CSV_PATH = f"{common.CONFIG['paths']['raw_data']}/{common.CONFIG['dataset']['csv_file']}"

data = pd.read_csv(CSV_PATH)

# display table dimensions
print(data.shape)

# display 10 random rows
print(data.sample(10))

# display descriptive statistics
print(data.describe())
print(data.describe(include="object"))

# check if there are any columns containing unique values for each row. If so, drop them.
data = data.drop(columns=["id"])

# dropoff_datetime variable is added only to train data and thus cannot be used by the predictive model.
data = data.drop(columns=["dropoff_datetime"])

# pickup_datetime contains date and time when the meter was engaged.
print(data["pickup_datetime"].dtype)
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
print(data["pickup_datetime"].head())

# Check the distribution of the target variable values.
data.trip_duration.hist()
plt.show()
