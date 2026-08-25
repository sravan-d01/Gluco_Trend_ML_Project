import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_minmax_scaling(df):

    print("\n" + "=" * 60)
    print("MIN-MAX SCALING")
    print("=" * 60)

    data = df.copy()

    print("Dataset shape:", data.shape)
    print("\nColumns:")
    print(data.columns.tolist())

    # Convert timestamp
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    # Sort chronologically per user (time-series safe version of a
    # random train/test split)
    data = data.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    # Extract time-based features
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    nums_cols = [
        "glucose_lag_1",
        "glucose_lag_3",
        "glucose_lag_6",
        "heart_rate",
        "carbs",
        "insulin_bolus",
        "insulin_basal",
        "exercise_steps",
        "stress_level",
        "hour",
        "minute",
        "day_of_week"
    ]

    nums_cols = [col for col in nums_cols if col in data.columns]

    # Drop rows missing lag values (no history yet, can't be used)
    lag_cols = ["glucose_lag_1", "glucose_lag_3", "glucose_lag_6"]
    existing_lags = [c for c in lag_cols if c in data.columns]
    data = data.dropna(subset=existing_lags)

    # Chronological 80/20 split, per user
    train_parts, test_parts = [], []

    for user_id, user_data in data.groupby("user_id"):
        split_index = int(len(user_data) * 0.8)
        train_parts.append(user_data.iloc[:split_index])
        test_parts.append(user_data.iloc[split_index:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    print("\nTraining data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)

    print("\nTraining Data before scaling:")
    print(train_df[nums_cols].head())

    print("\nTesting Data before scaling:")
    print(test_df[nums_cols].head())

    scaler = MinMaxScaler()

    train_df[nums_cols] = scaler.fit_transform(train_df[nums_cols])
    test_df[nums_cols] = scaler.transform(test_df[nums_cols])

    print("\nScaled Training Data:")
    print(train_df[nums_cols].head())

    print("\nScaled Testing Data:")
    print(test_df[nums_cols].head())

    return train_df, test_df, scaler