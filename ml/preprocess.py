import pandas as pd


FEATURES = [
    "quantity",
    "quantity_lag_1",
    "quantity_lag_7",
    "quantity_lag_14",
    "quantity_lag_28",
    "quantity_mean_7d",
    "quantity_mean_14d",
    "quantity_mean_28d",
    "quantity_std_28d",
    "quantity_sum_7d",
    "quantity_sum_14d",
    "quantity_sum_28d",
    "quantity_sum_56d",
    "quantity_sum_previous_7d",
    "quantity_sum_previous_14d",
    "quantity_sum_previous_28d",
    "ratio_previous_7d_28d",
    "ratio_previous_7d_recent_7d",
    "trend_7d_vs_28d",
    "trend_14d_vs_28d",
    "day_of_week",
    "month",
    "day_of_year",
    "sin_day_of_year",
    "cos_day_of_year",
    "is_weekend",
    "is_holiday",
]


TARGETS = [
    "target_1d",
    "target_2d",
    "target_3d",
    "target_4d",
    "target_5d",
    "target_6d",
    "target_7d",
]


def prepare_data(df: pd.DataFrame):

    df = df.copy()

    # Conversion de la date
    df["transactionDate"] = pd.to_datetime(
        df["transactionDate"]
    )

    # =========================================================
    # TARGET DIRECT 7 JOURS
    # =========================================================

    df["target_quantity_7d"] = df[TARGETS].sum(axis=1)

    # =========================================================
    # FEATURES
    # =========================================================

    X = df[FEATURES].copy()

    # =========================================================
    # TARGETS
    # =========================================================

    Y = df[
        TARGETS + ["target_quantity_7d"]
    ].copy().astype(float)

    print("\nDataset préparé")
    print("----------------")

    print("X shape :", X.shape)
    print("Y shape :", Y.shape)

    print("\nFeatures :")
    print(X.columns.tolist())

    print("\nTargets :")
    print(Y.columns.tolist())

    print("\nStatistiques des targets :")
    print(Y.describe())

    print("\nValeurs manquantes dans X :")
    print(X.isnull().sum())

    print("\nValeurs manquantes dans Y :")
    print(Y.isnull().sum())

    return X, Y