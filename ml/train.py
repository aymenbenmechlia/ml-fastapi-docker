import os

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.extract import get_sales_data
from ml.preprocess import prepare_data


# =========================================================
# CONFIGURATION
# =========================================================

GAP_DAYS = 7

TARGETS = [
    "target_1d",
    "target_2d",
    "target_3d",
    "target_4d",
    "target_5d",
    "target_6d",
    "target_7d",
]

DIRECT_TARGET = "target_quantity_7d"

RANDOM_STATE = 42


# =========================================================
# METRICS
# =========================================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# =========================================================
# CONFIGURATIONS LIGHTGBM
# =========================================================

CONFIGURATIONS = [

    {
        "name": "A_small",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },

    {
        "name": "B_standard",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },

    {
        "name": "C_regularized",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "D_complex",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "E_deep",
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },

    {
        "name": "F_high_regularization",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    },
]


# =========================================================
# CREATION MODELE
# =========================================================

def create_model(config):

    model = lgb.LGBMRegressor(
        objective="regression",

        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],
        min_child_samples=config["min_child_samples"],

        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],

        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],

        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    return model


# =========================================================
# OPTIMISATION D'UN HORIZON
# =========================================================

def optimize_horizon(
    X_train,
    y_train,
    target_name
):

    print("\n----------------------------------------")
    print(f"OPTIMISATION {target_name}")
    print("----------------------------------------")

    n_train = len(X_train)

    # -----------------------------------------------------
    # Validation temporelle
    # -----------------------------------------------------

    validation_size = max(
        60,
        int(n_train * 0.20)
    )

    validation_start = (
        n_train
        - validation_size
    )

    validation_train_end = (
        validation_start
        - GAP_DAYS
    )

    X_fit = X_train.iloc[
        :validation_train_end
    ]

    y_fit = y_train.iloc[
        :validation_train_end
    ]

    X_validation = X_train.iloc[
        validation_start:
    ]

    y_validation = y_train.iloc[
        validation_start:
    ]

    print(
        f"Train optimisation : "
        f"{len(X_fit)} lignes"
    )

    print(
        f"Validation          : "
        f"{len(X_validation)} lignes"
    )

    # -----------------------------------------------------
    # Test configurations
    # -----------------------------------------------------

    best_config = None
    best_mae = float("inf")

    tuning_results = []

    for config in CONFIGURATIONS:

        print(
            f"\n→ {target_name} | "
            f"{config['name']}"
        )

        model = create_model(config)

        model.fit(
            X_fit,
            y_fit
        )

        prediction = model.predict(
            X_validation
        )

        mae, rmse, r2 = evaluate(
            y_validation,
            prediction
        )

        print(
            f"MAE  : {mae:.4f} | "
            f"RMSE : {rmse:.4f} | "
            f"R²   : {r2:.4f}"
        )

        tuning_results.append({
            "target": target_name,
            "configuration": config["name"],
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        if mae < best_mae:

            best_mae = mae
            best_config = config

    # -----------------------------------------------------
    # Résultat optimisation
    # -----------------------------------------------------

    print("\nMEILLEURE CONFIGURATION")

    print(
        f"{target_name} → "
        f"{best_config['name']}"
    )

    print(
        f"Validation MAE : "
        f"{best_mae:.4f}"
    )

    return best_config, tuning_results


# =========================================================
# ENTRAINEMENT FINAL
# =========================================================

def train_final_model(
    X_train,
    y_train,
    config
):

    model = create_model(config)

    model.fit(
        X_train,
        y_train
    )

    return model


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # 1. EXTRACTION
    # =====================================================

    print("\n========================")
    print("EXTRACTION BIGQUERY")
    print("========================")

    df = get_sales_data()

    print(
        f"\n{len(df)} lignes récupérées"
    )

    # =====================================================
    # 2. PREPROCESSING
    # =====================================================

    print("\n========================")
    print("PREPROCESSING")
    print("========================")

    X, Y = prepare_data(df)

    dates = pd.to_datetime(
        df["transactionDate"]
    )

    # =====================================================
    # 3. TRI CHRONOLOGIQUE
    # =====================================================

    order = np.argsort(
        dates.values
    )

    X = (
        X.iloc[order]
        .reset_index(drop=True)
    )

    Y = (
        Y.iloc[order]
        .reset_index(drop=True)
    )

    dates = (
        dates.iloc[order]
        .reset_index(drop=True)
    )

    # =====================================================
    # 3 BIS. FEATURE BASELINE 28 JOURS
    # =====================================================

    # Baseline actuelle :
    # moyenne des 28 derniers jours × 7 jours
    #
    # On la donne explicitement à LightGBM
    # pour que le modèle puisse apprendre
    # une correction autour de cette baseline.

    X["baseline_28d_7days"] = (
        X["quantity_mean_28d"] * 7
    )

    print(
        "\nFeature ajoutée : "
        "baseline_28d_7days"
    )

    print(
        f"X shape après ajout : {X.shape}"
    )

    # =====================================================
    # 4. SPLIT TRAIN / TEST
    # =====================================================

    n = len(X)

    train_end = int(
        n * 0.80
    )

    test_start = (
        train_end
        + GAP_DAYS
    )

    X_train = X.iloc[
        :train_end
    ]

    X_test = X.iloc[
        test_start:
    ]

    Y_train = Y.iloc[
        :train_end
    ]

    Y_test = Y.iloc[
        test_start:
    ]

    dates_train = dates.iloc[
        :train_end
    ]

    dates_test = dates.iloc[
        test_start:
    ]

    print("\n========================")
    print("SPLIT TEMPOREL")
    print("========================")

    print(
        f"Train : {len(X_train)} lignes "
        f"({dates_train.min().date()} → "
        f"{dates_train.max().date()})"
    )

    print(
        f"Gap   : {GAP_DAYS} jours"
    )

    print(
        f"Test  : {len(X_test)} lignes "
        f"({dates_test.min().date()} → "
        f"{dates_test.max().date()})"
    )

    # =====================================================
    # 5. OPTIMISATION DES 7 MODELES
    # =====================================================

    print("\n========================")
    print("OPTIMISATION 7 MODELES")
    print("========================")

    best_configs = {}
    all_tuning_results = []

    for target in TARGETS:

        best_config, tuning_results = optimize_horizon(
            X_train,
            Y_train[target],
            target
        )

        best_configs[target] = best_config

        all_tuning_results.extend(
            tuning_results
        )

    # =====================================================
    # 6. OPTIMISATION MODELE DIRECT 7 JOURS
    # =====================================================

    print("\n========================")
    print("OPTIMISATION DIRECT 7 JOURS")
    print("========================")

    direct_best_config, direct_tuning_results = optimize_horizon(
        X_train,
        Y_train[DIRECT_TARGET],
        DIRECT_TARGET
    )

    all_tuning_results.extend(
        direct_tuning_results
    )

    # =====================================================
    # 7. ENTRAINEMENT FINAL DES 7 MODELES
    # =====================================================

    print("\n========================")
    print("ENTRAINEMENT FINAL")
    print("========================")

    models = {}
    predictions = {}

    for target in TARGETS:

        print(
            f"\n→ Entraînement final "
            f"{target}"
        )

        config = best_configs[target]

        model = train_final_model(
            X_train,
            Y_train[target],
            config
        )

        prediction = model.predict(
            X_test
        )

        models[target] = model
        predictions[target] = prediction

    # =====================================================
    # 8. ENTRAINEMENT FINAL DIRECT 7 JOURS
    # =====================================================

    print(
        "\n→ Entraînement final "
        "DIRECT 7 JOURS"
    )

    direct_model = train_final_model(
        X_train,
        Y_train[DIRECT_TARGET],
        direct_best_config
    )

    direct_prediction = direct_model.predict(
        X_test
    )

    # =====================================================
    # 9. PERFORMANCE PAR HORIZON
    # =====================================================

    print("\n========================")
    print("PERFORMANCE PAR HORIZON")
    print("========================")

    results = []

    for target in TARGETS:

        y_true = (
            Y_test[target]
            .values
        )

        y_pred = predictions[target]

        # Baseline 28 jours
        baseline = (
            X_test[
                "quantity_mean_28d"
            ]
            .values
        )

        (
            mae_model,
            rmse_model,
            r2_model
        ) = evaluate(
            y_true,
            y_pred
        )

        (
            mae_baseline,
            rmse_baseline,
            r2_baseline
        ) = evaluate(
            y_true,
            baseline
        )

        improvement = (
            1
            - mae_model / mae_baseline
        ) * 100

        results.append({

            "horizon": target,

            "baseline_mae":
                mae_baseline,

            "model_mae":
                mae_model,

            "improvement_%":
                improvement,

            "baseline_rmse":
                rmse_baseline,

            "model_rmse":
                rmse_model,

            "model_r2":
                r2_model,
        })

        print(f"\n{target}")

        print(
            f"Configuration → "
            f"{best_configs[target]['name']}"
        )

        print(
            f"Baseline 28j → "
            f"MAE: {mae_baseline:.4f} | "
            f"RMSE: {rmse_baseline:.4f} | "
            f"R²: {r2_baseline:.4f}"
        )

        print(
            f"LightGBM     → "
            f"MAE: {mae_model:.4f} | "
            f"RMSE: {rmse_model:.4f} | "
            f"R²: {r2_model:.4f}"
        )

        print(
            f"Amélioration MAE : "
            f"{improvement:.2f}%"
        )

    # =====================================================
    # 10. TABLEAU 7 HORIZONS
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n========================")
    print("TABLEAU 7 HORIZONS")
    print("========================")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # =====================================================
    # 11. TARGET DIRECT 7 JOURS
    # =====================================================

    y_test_7d = (
        Y_test[
            DIRECT_TARGET
        ]
        .values
    )

    # =====================================================
    # 12. PERFORMANCE MODELE DIRECT
    # =====================================================

    (
        mae_direct,
        rmse_direct,
        r2_direct
    ) = evaluate(
        y_test_7d,
        direct_prediction
    )

    print("\n========================")
    print("MODELE DIRECT 7 JOURS")
    print("========================")

    print(
        f"Configuration → "
        f"{direct_best_config['name']}"
    )

    print(
        f"LightGBM Direct → "
        f"MAE: {mae_direct:.4f} | "
        f"RMSE: {rmse_direct:.4f} | "
        f"R²: {r2_direct:.4f}"
    )

    # =====================================================
    # 13. SOMME DES 7 MODELES
    # =====================================================

    prediction_7d_from_7_models = np.sum(
        np.column_stack([
            predictions[target]
            for target in TARGETS
        ]),
        axis=1
    )

    # =====================================================
    # 14. BASELINES
    # =====================================================

    baseline_7d = (
        X_test[
            "quantity_mean_7d"
        ].values
        * 7
    )

    baseline_28d = (
        X_test[
            "quantity_mean_28d"
        ].values
        * 7
    )

    # =====================================================
    # 15. PERFORMANCE BASELINE 7 JOURS
    # =====================================================

    (
        mae_7d,
        rmse_7d,
        r2_7d
    ) = evaluate(
        y_test_7d,
        baseline_7d
    )

    # =====================================================
    # 16. PERFORMANCE BASELINE 28 JOURS
    # =====================================================

    (
        mae_28d,
        rmse_28d,
        r2_28d
    ) = evaluate(
        y_test_7d,
        baseline_28d
    )

    # =====================================================
    # 17. PERFORMANCE 7 MODELES
    # =====================================================

    (
        mae_multi,
        rmse_multi,
        r2_multi
    ) = evaluate(
        y_test_7d,
        prediction_7d_from_7_models
    )

    # =====================================================
    # 18. AMELIORATION DIRECT VS BASELINE 28J
    # =====================================================

    improvement_direct = (
        1
        - mae_direct / mae_28d
    ) * 100

    # =====================================================
    # 19. COMPARAISON FINALE
    # =====================================================

    print("\n========================")
    print("COMPARAISON FINALE 7 JOURS")
    print("========================")

    print(
        "\nBaseline moyenne 7 jours"
    )

    print(
        f"MAE  : {mae_7d:.4f}"
    )

    print(
        f"RMSE : {rmse_7d:.4f}"
    )

    print(
        f"R²   : {r2_7d:.4f}"
    )

    print(
        "\nBaseline moyenne 28 jours"
    )

    print(
        f"MAE  : {mae_28d:.4f}"
    )

    print(
        f"RMSE : {rmse_28d:.4f}"
    )

    print(
        f"R²   : {r2_28d:.4f}"
    )

    print(
        "\n7 modèles séparés → somme"
    )

    print(
        f"MAE  : {mae_multi:.4f}"
    )

    print(
        f"RMSE : {rmse_multi:.4f}"
    )

    print(
        f"R²   : {r2_multi:.4f}"
    )

    print(
        "\nLightGBM direct 7 jours"
    )

    print(
        f"MAE  : {mae_direct:.4f}"
    )

    print(
        f"RMSE : {rmse_direct:.4f}"
    )

    print(
        f"R²   : {r2_direct:.4f}"
    )

    print(
        f"Amélioration vs baseline 28j : "
        f"{improvement_direct:.2f}%"
    )

    # =====================================================
    # 20. TABLEAU FINAL
    # =====================================================

    final_results = pd.DataFrame([

        {
            "model":
                "Baseline 7 jours",

            "MAE":
                mae_7d,

            "RMSE":
                rmse_7d,

            "R2":
                r2_7d,
        },

        {
            "model":
                "Baseline 28 jours",

            "MAE":
                mae_28d,

            "RMSE":
                rmse_28d,

            "R2":
                r2_28d,
        },

        {
            "model":
                "7 modèles optimisés",

            "MAE":
                mae_multi,

            "RMSE":
                rmse_multi,

            "R2":
                r2_multi,
        },

        {
            "model":
                "LightGBM direct 7 jours",

            "MAE":
                mae_direct,

            "RMSE":
                rmse_direct,

            "R2":
                r2_direct,
        },
    ])

    print("\n========================")
    print("TABLEAU FINAL")
    print("========================")

    print(
        final_results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # =====================================================
    # 21. EXEMPLES DE PREDICTIONS
    # =====================================================

    print("\n========================")
    print("EXEMPLES DE PREDICTIONS")
    print("========================")

    for i in range(
        min(10, len(X_test))
    ):

        print(
            f"{dates_test.iloc[i].date()} | "
            f"Réel : {y_test_7d[i]:.1f} | "
            f"Direct : "
            f"{direct_prediction[i]:.1f} | "
            f"7 modèles : "
            f"{prediction_7d_from_7_models[i]:.1f} | "
            f"Baseline 28j : "
            f"{baseline_28d[i]:.1f}"
        )

    # =====================================================
    # 22. IMPORTANCE FEATURES MODELE DIRECT
    # =====================================================

    print("\n========================")
    print("IMPORTANCE DES FEATURES")
    print("========================")

    importance_df = pd.DataFrame({

        "feature":
            X.columns,

        "importance":
            direct_model.feature_importances_

    }).sort_values(
        "importance",
        ascending=False
    )

    for _, row in importance_df.iterrows():

        print(
            f"{row['feature']:<30} : "
            f"{int(row['importance'])}"
        )

    # =====================================================
    # 23. CONFIGURATIONS RETENUES
    # =====================================================

    print("\n========================")
    print("CONFIGURATIONS RETENUES")
    print("========================")

    for target in TARGETS:

        print(
            f"{target:<15} → "
            f"{best_configs[target]['name']}"
        )

    print(
        f"{DIRECT_TARGET:<15} → "
        f"{direct_best_config['name']}"
    )

    # =====================================================
    # 24. SAUVEGARDE
    # =====================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/models_7d_optimized.joblib"
    )

    artifact = {

        # 7 modèles séparés
        "models":
            models,

        # Modèle direct
        "direct_model":
            direct_model,

        "targets":
            TARGETS,

        "direct_target":
            DIRECT_TARGET,

        "features":
            X.columns.tolist(),

        "best_configs":
            best_configs,

        "direct_best_config":
            direct_best_config,

        "metrics": {

            "baseline_7d": {
                "MAE": mae_7d,
                "RMSE": rmse_7d,
                "R2": r2_7d,
            },

            "baseline_28d": {
                "MAE": mae_28d,
                "RMSE": rmse_28d,
                "R2": r2_28d,
            },

            "multi_model_7d": {
                "MAE": mae_multi,
                "RMSE": rmse_multi,
                "R2": r2_multi,
            },

            "direct_7d": {
                "MAE": mae_direct,
                "RMSE": rmse_direct,
                "R2": r2_direct,
            },
        },

        "gap_days":
            GAP_DAYS,

        # Information sur la nouvelle feature
        "direct_baseline_feature":
            "baseline_28d_7days",
    }

    joblib.dump(
        artifact,
        model_path
    )

    print(
        f"\nModèles sauvegardés : "
        f"{model_path}"
    )


# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":
    main()
