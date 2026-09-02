from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "ddp-dtm-perf-prd-frlm"

TABLE = (
    "ddp-dtm-perf-prd-frlm."
    "base_finance_performance_bu001."
    "int_salesMarginWAC"
)


def get_bigquery_client():
    return bigquery.Client(project=PROJECT_ID)


def get_sales_data(
    operational_unit_id: int,
    item_id: int,
):
    client = get_bigquery_client()

    query = f"""
    WITH daily_sales AS (
        SELECT
            transactionDate,
            operationalUnitIdentifier,
            itemIdentifier,
            SUM(itemQuantity) AS quantity
        FROM `{TABLE}`
        WHERE transactionDate >= '2024-01-01'
          AND operationalUnitIdentifier = @operational_unit_id
          AND itemIdentifier = @item_id
        GROUP BY
            transactionDate,
            operationalUnitIdentifier,
            itemIdentifier
    ),

    holidays AS (
        -- 2024
        SELECT DATE '2024-01-01' AS holiday_date UNION ALL
        SELECT DATE '2024-04-01' UNION ALL
        SELECT DATE '2024-05-01' UNION ALL
        SELECT DATE '2024-05-08' UNION ALL
        SELECT DATE '2024-05-09' UNION ALL
        SELECT DATE '2024-05-20' UNION ALL
        SELECT DATE '2024-07-14' UNION ALL
        SELECT DATE '2024-08-15' UNION ALL
        SELECT DATE '2024-11-01' UNION ALL
        SELECT DATE '2024-11-11' UNION ALL
        SELECT DATE '2024-12-25' UNION ALL

        -- 2025
        SELECT DATE '2025-01-01' UNION ALL
        SELECT DATE '2025-04-21' UNION ALL
        SELECT DATE '2025-05-01' UNION ALL
        SELECT DATE '2025-05-08' UNION ALL
        SELECT DATE '2025-05-29' UNION ALL
        SELECT DATE '2025-06-09' UNION ALL
        SELECT DATE '2025-07-14' UNION ALL
        SELECT DATE '2025-08-15' UNION ALL
        SELECT DATE '2025-11-01' UNION ALL
        SELECT DATE '2025-11-11' UNION ALL
        SELECT DATE '2025-12-25' UNION ALL

        -- 2026
        SELECT DATE '2026-01-01' UNION ALL
        SELECT DATE '2026-04-06' UNION ALL
        SELECT DATE '2026-05-01' UNION ALL
        SELECT DATE '2026-05-08' UNION ALL
        SELECT DATE '2026-05-14' UNION ALL
        SELECT DATE '2026-05-25' UNION ALL
        SELECT DATE '2026-07-14' UNION ALL
        SELECT DATE '2026-08-15' UNION ALL
        SELECT DATE '2026-11-01' UNION ALL
        SELECT DATE '2026-11-11' UNION ALL
        SELECT DATE '2026-12-25'
    ),

    date_range AS (
        SELECT
            MIN(transactionDate) AS min_date,
            MAX(transactionDate) AS max_date
        FROM daily_sales
    ),

    calendar AS (
        SELECT
            date AS transactionDate
        FROM date_range,
        UNNEST(
            GENERATE_DATE_ARRAY(min_date, max_date)
        ) AS date
    ),

    complete_sales AS (
        SELECT
            c.transactionDate,
            @operational_unit_id AS operationalUnitIdentifier,
            @item_id AS itemIdentifier,
            COALESCE(d.quantity, 0) AS quantity
        FROM calendar c

        LEFT JOIN daily_sales d
            ON c.transactionDate = d.transactionDate
    ),

    features AS (
        SELECT
            s.transactionDate,
            s.operationalUnitIdentifier,
            s.itemIdentifier,
            s.quantity,

            -- ========================
            -- LAGS
            -- ========================

            LAG(s.quantity, 1) OVER (
                ORDER BY s.transactionDate
            ) AS quantity_lag_1,

            LAG(s.quantity, 7) OVER (
                ORDER BY s.transactionDate
            ) AS quantity_lag_7,

            LAG(s.quantity, 14) OVER (
                ORDER BY s.transactionDate
            ) AS quantity_lag_14,

            LAG(s.quantity, 28) OVER (
                ORDER BY s.transactionDate
            ) AS quantity_lag_28,


            -- ========================
            -- MOYENNES HISTORIQUES
            -- ========================

            AVG(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS quantity_mean_7d,

            AVG(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
            ) AS quantity_mean_14d,

            AVG(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
            ) AS quantity_mean_28d,

            STDDEV(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
            ) AS quantity_std_28d,


            -- ========================
            -- SOMMES HISTORIQUES
            -- ========================

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS quantity_sum_7d,

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
            ) AS quantity_sum_14d,

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
            ) AS quantity_sum_28d,

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 56 PRECEDING AND 1 PRECEDING
            ) AS quantity_sum_56d,


            -- ========================
            -- SEMAINE PRECEDENTE
            -- ========================

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 14 PRECEDING AND 8 PRECEDING
            ) AS quantity_sum_previous_7d,


            -- ========================
            -- 2 SEMAINES PRECEDENTES
            -- ========================

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 21 PRECEDING AND 8 PRECEDING
            ) AS quantity_sum_previous_14d,


            -- ========================
            -- 4 SEMAINES PRECEDENTES
            -- ========================

            SUM(s.quantity) OVER (
                ORDER BY s.transactionDate
                ROWS BETWEEN 35 PRECEDING AND 8 PRECEDING
            ) AS quantity_sum_previous_28d,


            -- ========================
            -- SAISONNALITE
            -- ========================

            MOD(
                EXTRACT(DAYOFWEEK FROM s.transactionDate) + 5,
                7
            ) AS day_of_week,

            EXTRACT(MONTH FROM s.transactionDate) AS month,

            EXTRACT(DAYOFYEAR FROM s.transactionDate) AS day_of_year,

            SIN(
                2 * 3.14159265359 *
                EXTRACT(DAYOFYEAR FROM s.transactionDate) / 365.25
            ) AS sin_day_of_year,

            COS(
                2 * 3.14159265359 *
                EXTRACT(DAYOFYEAR FROM s.transactionDate) / 365.25
            ) AS cos_day_of_year,


            -- ========================
            -- WEEK-END
            -- ========================

            CASE
                WHEN EXTRACT(DAYOFWEEK FROM s.transactionDate) IN (1, 7)
                THEN 1
                ELSE 0
            END AS is_weekend,


            -- ========================
            -- JOUR FERIE
            -- ========================

            CASE
                WHEN h.holiday_date IS NOT NULL
                THEN 1
                ELSE 0
            END AS is_holiday

        FROM complete_sales s

        LEFT JOIN holidays h
            ON s.transactionDate = h.holiday_date
    ),


    -- ========================
    -- TARGETS
    -- ========================

    final AS (
        SELECT
            *,

            LEAD(quantity, 1) OVER (
                ORDER BY transactionDate
            ) AS target_1d,

            LEAD(quantity, 2) OVER (
                ORDER BY transactionDate
            ) AS target_2d,

            LEAD(quantity, 3) OVER (
                ORDER BY transactionDate
            ) AS target_3d,

            LEAD(quantity, 4) OVER (
                ORDER BY transactionDate
            ) AS target_4d,

            LEAD(quantity, 5) OVER (
                ORDER BY transactionDate
            ) AS target_5d,

            LEAD(quantity, 6) OVER (
                ORDER BY transactionDate
            ) AS target_6d,

            LEAD(quantity, 7) OVER (
                ORDER BY transactionDate
            ) AS target_7d,

            (
                LEAD(quantity, 1) OVER (ORDER BY transactionDate) +
                LEAD(quantity, 2) OVER (ORDER BY transactionDate) +
                LEAD(quantity, 3) OVER (ORDER BY transactionDate) +
                LEAD(quantity, 4) OVER (ORDER BY transactionDate) +
                LEAD(quantity, 5) OVER (ORDER BY transactionDate) +
                LEAD(quantity, 6) OVER (ORDER BY transactionDate) +
                LEAD(quantity, 7) OVER (ORDER BY transactionDate)
            ) AS target_quantity_7d

        FROM features
    )


    -- ========================
    -- FINAL DATASET
    -- ========================

    SELECT
        *,

        -- Tendances
        quantity_mean_7d /
            NULLIF(quantity_mean_28d, 0)
            AS trend_7d_vs_28d,

        quantity_mean_14d /
            NULLIF(quantity_mean_28d, 0)
            AS trend_14d_vs_28d,

        -- Ratio semaine précédente / moyenne 28 jours
        quantity_sum_previous_7d /
            NULLIF(quantity_sum_28d, 0)
            AS ratio_previous_7d_28d,

        -- Ratio semaine précédente / semaine actuelle historique
        quantity_sum_previous_7d /
            NULLIF(quantity_sum_7d, 0)
            AS ratio_previous_7d_recent_7d

    FROM final

    WHERE quantity_lag_28 IS NOT NULL
      AND target_7d IS NOT NULL

    ORDER BY transactionDate
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "operational_unit_id",
                "INT64",
                operational_unit_id,
            ),
            bigquery.ScalarQueryParameter(
                "item_id",
                "INT64",
                item_id,
            ),
        ]
    )

    return client.query(
        query,
        job_config=job_config,
    ).to_dataframe()


if __name__ == "__main__":
    df = get_sales_data(
        operational_unit_id=146,
        item_id=49016767,
    )

    print("Shape :", df.shape)

    print("\nColonnes :")
    print(df.columns.tolist())

    print("\nPremières lignes :")
    print(df.head())

    print("\nDernières lignes :")
    print(df.tail())