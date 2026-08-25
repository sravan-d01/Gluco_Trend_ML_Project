import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def run_ordinal_encoding(train_df, test_df):

    print("\n" + "=" * 60)
    print("ORDINAL ENCODING")
    print("=" * 60)

    train_data = train_df.copy()
    test_data = test_df.copy()

    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    # Define the order for the ordinal category
    exercise_intensity_order = ["none", "low", "medium", "high"]

    ord_enc = OrdinalEncoder(
        categories=[exercise_intensity_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    ordinal_cols = ["exercise_intensity"]

    train_data["exercise_intensity_enc"] = ord_enc.fit_transform(
        train_data[ordinal_cols]
    )

    test_data["exercise_intensity_enc"] = ord_enc.transform(
        test_data[ordinal_cols]
    )

    # Drop original column
    train_data = train_data.drop(columns=ordinal_cols)
    test_data = test_data.drop(columns=ordinal_cols)

    print("\nTraining data:")
    print(train_data[["exercise_intensity_enc"]].head(10))

    print("\nTesting data:")
    print(test_data[["exercise_intensity_enc"]].head(10))

    return train_data, test_data, ord_enc