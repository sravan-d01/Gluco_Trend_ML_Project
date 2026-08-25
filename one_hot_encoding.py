import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def run_onehot_encoding(train_df, test_df):

    print("\n" + "=" * 60)
    print("ONE-HOT ENCODING")
    print("=" * 60)

    train_data = train_df.copy()
    test_data = test_df.copy()

    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    # Categorical / nominal columns
    nominal_cols = [
        "meal_type",
        "sleep_stage",
        "medication_other",
        "device_id",
        "sex",
        "timezone",
        "region"
    ]

    nominal_cols = [col for col in nominal_cols if col in train_data.columns]

    print("\nCategorical Columns:")
    print(nominal_cols)

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_ohe = ohe.fit_transform(train_data[nominal_cols])
    test_ohe = ohe.transform(test_data[nominal_cols])

    ohe_cols = ohe.get_feature_names_out(nominal_cols)

    train_ohe_df = pd.DataFrame(
        train_ohe, columns=ohe_cols, index=train_data.index
    )

    test_ohe_df = pd.DataFrame(
        test_ohe, columns=ohe_cols, index=test_data.index
    )

    print("\nNumber of original categorical columns:", len(nominal_cols))
    print("Number of generated encoded columns:", len(ohe_cols))

    print("\nGenerated encoded columns:")
    for col in ohe_cols:
        print(col)

    # Drop originals, attach encoded columns
    train_data = train_data.drop(columns=nominal_cols)
    test_data = test_data.drop(columns=nominal_cols)

    train_data = pd.concat([train_data, train_ohe_df], axis=1)
    test_data = pd.concat([test_data, test_ohe_df], axis=1)

    print("\nFirst 5 rows of encoded training data:")
    print(train_ohe_df.head())

    print("\nFirst 5 rows of encoded testing data:")
    print(test_ohe_df.head())

    return train_data, test_data, ohe