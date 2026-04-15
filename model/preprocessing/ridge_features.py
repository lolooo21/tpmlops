# Shared feature lists keep training and serving aligned on the exact same columns.
NUM_FEATURES = [
    "log_distance_haversine",
    "hour",
    "hour_sin",
    "hour_cos",
    "abnormal_period",
    "is_weekend",
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_high_traffic_trip",
    "is_high_speed_trip",
    "is_rare_pickup_point",
    "is_rare_dropoff_point",
]
CAT_FEATURES = ["weekday", "month", "time_bucket"]
TRAIN_FEATURES = NUM_FEATURES + CAT_FEATURES
