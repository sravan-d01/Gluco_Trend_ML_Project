from flask import Flask, render_template
import traceback

from load_data import load_data, get_data_summary, DATA_PATH
from Mini_Max_Scaling import run_minmax_scaling
from one_hot_encoding import run_onehot_encoding
from Ordinal_Encoding import run_ordinal_encoding
from Standard_Scaler import run_standard_scaling

import gluco_eda


app = Flask(__name__)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ==========================================================
# DATA LOADING
# ==========================================================

@app.route("/data-loading")
def data_loading():

    try:

        print("\n" + "=" * 70)
        print("                 LOADING GLUCO TREND DATASET")
        print("=" * 70)

        print("\nDataset Path:")
        print(DATA_PATH)

        summary = get_data_summary()

        print("\nDataset loaded successfully!")
        print("Dataset Shape:", (summary["n_rows"], summary["n_cols"]))

        return render_template(
            "data_loading.html",
            active="data-loading",
            summary=summary
        )

    except Exception as e:

        traceback.print_exc()

        return render_template(
            "data_loading.html",
            active="data-loading",
            error=str(e)
        )


# ==========================================================
# EDA
# ==========================================================

@app.route("/eda")
def eda_page():

    try:

        results = gluco_eda.run_eda()

        return render_template(
            "eda.html",
            active="eda",
            results=results
        )

    except Exception as e:

        traceback.print_exc()

        return render_template(
            "eda.html",
            active="eda",
            error=str(e)
        )


# ==========================================================
# PREPROCESSING
# ==========================================================

@app.route("/preprocessing")
def preprocessing():

    try:

        print("\n")
        print("=" * 70)
        print("          GLUCO TREND PREPROCESSING PIPELINE")
        print("=" * 70)

        # STEP 1 - LOAD DATA
        print("\nSTEP 1 : Loading dataset...")

        df = load_data()

        print("\nOriginal Dataset Shape:", df.shape)

        # STEP 2 - MIN-MAX SCALING
        print("\nSTEP 2 : Min-Max Scaling...")

        train_df, test_df, minmax_scaler = run_minmax_scaling(df)

        # STEP 3 - ONE-HOT ENCODING
        print("\nSTEP 3 : One-Hot Encoding...")

        train_df, test_df, ohe = run_onehot_encoding(
            train_df,
            test_df
        )

        # STEP 4 - ORDINAL ENCODING
        print("\nSTEP 4 : Ordinal Encoding...")

        train_df, test_df, ordinal_encoder = run_ordinal_encoding(
            train_df,
            test_df
        )

        # STEP 5 - STANDARD SCALING
        print("\nSTEP 5 : Standard Scaling...")

        train_df, test_df, standard_scaler = run_standard_scaling(
            train_df,
            test_df
        )

        # TARGET COLUMN
        target_column = "glucose"

        if target_column not in train_df.columns:
            raise ValueError(
                "Target column 'glucose' was not found in the dataset."
            )

        y_train = train_df[target_column].copy()
        y_test = test_df[target_column].copy()

        # REMOVE NON-FEATURE COLUMNS
        columns_to_remove = [
            "glucose",
            "timestamp",
            "user_id",
            "glucose_roll_mean_1h"
        ]

        X_train = train_df.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

        X_test = test_df.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

        print("\n")
        print("=" * 70)
        print("       PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nX_train Shape:", X_train.shape)
        print("X_test Shape:", X_test.shape)
        print("y_train Shape:", y_train.shape)
        print("y_test Shape:", y_test.shape)

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            success=True,
            original_shape=df.shape,
            train_shape=X_train.shape,
            test_shape=X_test.shape,
            target_train_shape=y_train.shape,
            target_test_shape=y_test.shape,
            feature_count=X_train.shape[1],
            train_preview=X_train.head(10).to_html(
                classes="data-table",
                index=False
            ),
            test_preview=X_test.head(10).to_html(
                classes="data-table",
                index=False
            )
        )

    except Exception as e:

        traceback.print_exc()

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            success=False,
            error=str(e)
        )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )