import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

import common
from model.features import add_time_features, build_abnormal_dates
from model.preprocessing import transform_target

CSV_PATH = f"{common.CONFIG['paths']['raw_data']}/{common.CONFIG['dataset']['csv_file']}"
RANDOM_STATE = int(common.CONFIG["ml"]["random_state"])

data = pd.read_csv(CSV_PATH)
data = data.drop(columns=["id", "dropoff_datetime"])
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

# distribution of target
data.trip_duration[data.trip_duration < data.trip_duration.quantile(0.99)].hist(bins=100)
plt.show()

X = data.drop(columns=["trip_duration"])
y = data["trip_duration"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
)

y = transform_target(y)
y_train = transform_target(y_train)
y_test = transform_target(y_test)

y.hist(bins=100)
plt.show()

# baseline
y_baseline = y_train.mean()
print(f"Baseline prediction: {y_baseline:.2f} (transformed)")
print(f"Baseline prediction: {np.expm1(y_baseline):.0f} (seconds)")
print(f"RMSLE on train data: {root_mean_squared_error([y_baseline] * len(y_train), y_train):.3f}")
print(f"RMSLE on test data: {root_mean_squared_error([y_baseline] * len(y_test), y_test):.3f}")

# trips by date
X["pickup_date"] = X["pickup_datetime"].dt.date
plt.plot(X["pickup_date"].groupby(X["pickup_date"]).count(), "o-")
plt.show()

plt.plot(X_train["pickup_datetime"].groupby(X_train["pickup_datetime"].dt.date).count(), "o-", label="train")
plt.plot(X_test["pickup_datetime"].groupby(X_test["pickup_datetime"].dt.date).count(), "o-", label="test")
plt.title("Number of trips by date")
plt.legend(loc=0)
plt.ylabel("Number of trips")
plt.show()

abnormal_dates = build_abnormal_dates(X)
print(abnormal_dates)

dict_weekday = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

weekday_num = X["pickup_datetime"].dt.weekday
weekday = X["pickup_datetime"].dt.weekday.map(dict_weekday).rename("weekday")
hourofday = X["pickup_datetime"].dt.hour.rename("hour")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(x=weekday_num, ax=ax[0])
ax[0].set(xlabel="weekday_num")
ax[0].tick_params("x", labelrotation=45)

sns.countplot(x=weekday, ax=ax[1])
ax[1].set(xlabel="weekday")
ax[1].tick_params("x", labelrotation=45)

sns.countplot(x=hourofday, ax=ax[2])
ax[2].set(xlabel="hour")
ax[2].tick_params("x", labelrotation=90)

plt.show()

X = add_time_features(X, abnormal_dates)
X_train = add_time_features(X_train, abnormal_dates)
X_test = add_time_features(X_test, abnormal_dates)

print(X[["pickup_datetime", "weekday", "month", "hour", "abnormal_period"]].head())
print(X["weekday"].value_counts().head())
print(X["month"].value_counts().head())
print(X["hour"].value_counts().head())
print(X["abnormal_period"].value_counts())
