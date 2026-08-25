import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_standard_scaling(train_df, test_df):

    print("\n" + "=" * 60)
    print("STANDARD SCALING")
    print("=" * 60)

    train_data = train_df.copy()
    test_data = test_df.copy()

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

    nums_cols = [col for col in nums_cols if col in train_data.columns]

    print("\nTraining Data before scaling:")
    print(train_data[nums_cols].head())

    print("\nTesting Data before scaling:")
    print(test_data[nums_cols].head())

    scaler = StandardScaler()

    # Fit only on training data
    train_data[nums_cols] = scaler.fit_transform(train_data[nums_cols])

    # Transform test data using the same fitted scaler
    test_data[nums_cols] = scaler.transform(test_data[nums_cols])

    print("\nScaled training shape:", train_data.shape)
    print("Scaled testing shape:", test_data.shape)

    print("\nScaled Training Data:")
    print(train_data[nums_cols].head())

    print("\nScaled Testing Data:")
    print(test_data[nums_cols].head())

    return train_data, test_data, scaler