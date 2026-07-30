from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} tablosunda eksik sütunlar: {sorted(missing)}"
        )


def _nice_multiplier(value: float) -> int:
    if not np.isfinite(value) or value <= 0:
        return 1

    exponent = int(np.floor(np.log10(value)))
    fraction = value / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return max(int(nice_fraction * (10 ** exponent)), 1)


def choose_demo_unit_multiplier(
    normalized_sales: pd.Series,
    *,
    target_median_daily_units: float = 20.0,
) -> int:
    """
    Bilinmeyen gerçek normalizasyon katsayısını geri kazanmaz.
    Pozitif normalize satış medyanını seçilen demo medyan adede taşıyan
    açıklanabilir bir global çarpan üretir.
    """
    numeric = pd.to_numeric(normalized_sales, errors="coerce")
    positive = numeric[np.isfinite(numeric) & numeric.gt(0)]

    if positive.empty:
        raise ValueError(
            "Demo birim çarpanı için pozitif normalize satış bulunamadı."
        )

    if target_median_daily_units <= 0:
        raise ValueError("target_median_daily_units pozitif olmalıdır.")

    normalized_median = float(positive.median())
    raw_multiplier = target_median_daily_units / normalized_median
    return _nice_multiplier(raw_multiplier)


def _stable_uniform(
    key: object,
    *,
    seed: int,
    low: float,
    high: float,
) -> float:
    payload = f"{seed}|{key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    unit = integer / float(2**64 - 1)
    return low + (high - low) * unit


def _retail_price(value: float, *, minimum: float, maximum: float) -> float:
    clipped = float(np.clip(value, minimum, maximum))
    whole = max(int(np.floor(clipped)), 1)
    return float(np.clip(whole + 0.90, minimum, maximum))


def build_demo_product_prices(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    min_price_try: float = 19.90,
    max_price_try: float = 399.90,
) -> pd.DataFrame:
    """
    Yalnızca FreshRetailNet demosu için ürün bazlı sentetik liste fiyatı üretir.
    Üretim sisteminde gerçek price sütunu kullanılmalıdır.
    """
    _require_columns(df, {"product_id"}, "df")

    columns = ["product_id"]
    for column in ("category_1", "category_2", "category_3"):
        if column in df.columns:
            columns.append(column)

    catalog = (
        df[columns]
        .drop_duplicates("product_id")
        .reset_index(drop=True)
        .copy()
    )

    category_column = (
        "category_1" if "category_1" in catalog.columns else None
    )

    category_base: dict[str, float] = {}

    if category_column is not None:
        for category in catalog[category_column].astype(str).unique():
            unit = _stable_uniform(
                f"category:{category}",
                seed=seed,
                low=0.0,
                high=1.0,
            )
            category_base[category] = float(
                np.exp(
                    np.log(29.90)
                    + unit * (np.log(199.90) - np.log(29.90))
                )
            )

    prices: list[float] = []

    for row in catalog.itertuples(index=False):
        product_id = str(row.product_id)

        if category_column is not None:
            base = category_base[str(getattr(row, category_column))]
        else:
            base = 79.90

        product_factor = _stable_uniform(
            f"product:{product_id}",
            seed=seed,
            low=0.60,
            high=1.80,
        )

        prices.append(
            _retail_price(
                base * product_factor,
                minimum=min_price_try,
                maximum=max_price_try,
            )
        )

    catalog["price"] = prices
    catalog["price_source"] = "synthetic_demo"
    catalog["currency"] = "TRY"
    catalog["demo_price_seed"] = int(seed)

    return catalog


def _build_discount_lookup(
    raw_df: pd.DataFrame,
    *,
    date_column: str,
    discount_column: str,
    series_id_column: str,
) -> pd.DataFrame:
    _require_columns(
        raw_df,
        {date_column, discount_column, series_id_column},
        "raw_df",
    )

    lookup = raw_df[
        [series_id_column, date_column, discount_column]
    ].copy()

    lookup["series_id"] = lookup[series_id_column].astype(str)
    lookup["date"] = pd.to_datetime(
        lookup[date_column],
        errors="raise",
    ).dt.normalize()

    lookup["discount_rate"] = pd.to_numeric(
        lookup[discount_column],
        errors="coerce",
    ).fillna(1.0)

    lookup["discount_rate"] = lookup["discount_rate"].clip(
        lower=0.30,
        upper=1.00,
    )

    return (
        lookup[["series_id", "date", "discount_rate"]]
        .groupby(["series_id", "date"], as_index=False)
        .agg(discount_rate=("discount_rate", "mean"))
    )


def prepare_freshretail_demo_commercial_data(
    adjusted_df: pd.DataFrame,
    *,
    raw_df: Optional[pd.DataFrame] = None,
    unit_multiplier: Optional[int] = None,
    target_median_daily_units: float = 20.0,
    seed: int = 42,
    min_price_try: float = 19.90,
    max_price_try: float = 399.90,
    raw_date_column: str = "dt",
    raw_discount_column: str = "discount",
    raw_series_id_column: str = "temporary_series_id",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    FreshRetailNet için tek seferlik demo ön işleme katmanı.
    Üretim pipeline'ının parçası değildir.
    """
    _require_columns(
        adjusted_df,
        {
            "series_id",
            "date",
            "store_id",
            "product_id",
            "sales",
            "demand_adjusted",
            "is_stockout",
        },
        "adjusted_df",
    )

    df = adjusted_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    df["sales_normalized"] = pd.to_numeric(
        df["sales"],
        errors="coerce",
    )
    df["demand_adjusted_normalized"] = pd.to_numeric(
        df["demand_adjusted"],
        errors="coerce",
    )

    if df[
        ["sales_normalized", "demand_adjusted_normalized"]
    ].isna().any().any():
        raise ValueError("Normalize satış/talep sütununda geçersiz değer var.")

    if unit_multiplier is None:
        unit_multiplier = choose_demo_unit_multiplier(
            df["sales_normalized"],
            target_median_daily_units=target_median_daily_units,
        )

    if unit_multiplier <= 0:
        raise ValueError("unit_multiplier pozitif olmalıdır.")

    df["demo_unit_multiplier"] = int(unit_multiplier)

    df["sales"] = np.rint(
        df["sales_normalized"] * unit_multiplier
    ).clip(lower=0).astype(int)

    df["demand_adjusted"] = np.rint(
        df["demand_adjusted_normalized"] * unit_multiplier
    ).clip(lower=0).astype(int)

    df["demand_adjusted"] = np.maximum(
        df["demand_adjusted"],
        df["sales"],
    ).astype(int)

    df["demand_adjustment"] = (
        df["demand_adjusted"] - df["sales"]
    ).astype(int)
    df["demand_was_imputed"] = df["demand_adjustment"].gt(0)

    catalog = build_demo_product_prices(
        df,
        seed=seed,
        min_price_try=min_price_try,
        max_price_try=max_price_try,
    )

    df = df.merge(
        catalog[
            [
                "product_id",
                "price",
                "price_source",
                "currency",
                "demo_price_seed",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    df = df.rename(columns={"price": "list_price"})

    if (
        raw_df is not None
        and raw_discount_column in raw_df.columns
        and raw_series_id_column in raw_df.columns
    ):
        discount_lookup = _build_discount_lookup(
            raw_df,
            date_column=raw_date_column,
            discount_column=raw_discount_column,
            series_id_column=raw_series_id_column,
        )

        df = df.merge(
            discount_lookup,
            on=["series_id", "date"],
            how="left",
            validate="one_to_one",
        )
        df["discount_rate"] = df["discount_rate"].fillna(1.0)
    else:
        df["discount_rate"] = 1.0

    df["price"] = (
        df["list_price"] * df["discount_rate"]
    ).round(2)

    df["observed_revenue"] = (
        df["sales"] * df["price"]
    ).round(2)

    df["adjusted_revenue"] = (
        df["demand_adjusted"] * df["price"]
    ).round(2)

    df["estimated_lost_revenue"] = (
        df["adjusted_revenue"] - df["observed_revenue"]
    ).clip(lower=0).round(2)

    df["sales_unit_source"] = "scenario_scaled_demo"
    df["is_demo_commercial_data"] = True

    return (
        df.sort_values(["series_id", "date"]).reset_index(drop=True),
        catalog.sort_values("product_id").reset_index(drop=True),
        int(unit_multiplier),
    )


def scale_forecast_for_freshretail_demo(
    future_forecast_df: pd.DataFrame,
    *,
    unit_multiplier: int,
) -> pd.DataFrame:
    _require_columns(
        future_forecast_df,
        {"series_id", "date", "predictions"},
        "future_forecast_df",
    )

    if unit_multiplier <= 0:
        raise ValueError("unit_multiplier pozitif olmalıdır.")

    result = future_forecast_df.copy()
    result["predictions_normalized"] = pd.to_numeric(
        result["predictions"],
        errors="coerce",
    )

    if result["predictions_normalized"].isna().any():
        raise ValueError("Tahmin sütununda geçersiz değer var.")

    result["predictions"] = np.rint(
        result["predictions_normalized"] * unit_multiplier
    ).clip(lower=0).astype(int)

    result["prediction_unit_source"] = "scenario_scaled_demo"
    result["demo_unit_multiplier"] = int(unit_multiplier)

    return result.sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True)

def make_demo_prepared_df(
    prepared_df: pd.DataFrame,
    demo_adjusted_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    prepared_df içindeki normalize satışları FreshRetailNet demo satış adediyle
    değiştirir ve sentetik demo fiyatını ekler.

    Üretim ortamında bu fonksiyon kullanılmaz; gerçek sales ve price sütunları
    doğrudan kullanıcı verisinden gelir.
    """
    _require_columns(
        prepared_df,
        {"series_id", "date", "sales"},
        "prepared_df",
    )
    _require_columns(
        demo_adjusted_df,
        {
            "series_id",
            "date",
            "sales",
            "price",
            "list_price",
            "discount_rate",
        },
        "demo_adjusted_df",
    )

    base = prepared_df.copy()
    base["date"] = pd.to_datetime(base["date"], errors="raise")

    columns_to_drop = [
        column
        for column in (
            "sales",
            "price",
            "list_price",
            "discount_rate",
            "sales_normalized",
            "demo_unit_multiplier",
            "price_source",
            "currency",
            "is_demo_commercial_data",
        )
        if column in base.columns
    ]
    base = base.drop(columns=columns_to_drop)

    replacement = demo_adjusted_df[
        [
            "series_id",
            "date",
            "sales",
            "price",
            "list_price",
            "discount_rate",
            "sales_normalized",
            "demo_unit_multiplier",
            "price_source",
            "currency",
            "is_demo_commercial_data",
        ]
    ].copy()

    replacement["date"] = pd.to_datetime(
        replacement["date"],
        errors="raise",
    )

    result = base.merge(
        replacement,
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    if result["sales"].isna().any():
        raise ValueError(
            "Bazı prepared_df satırları demo ticari veriyle eşleşmedi."
        )

    return result.sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True)


def generate_demo_inventory(
    demo_adjusted_df: pd.DataFrame,
    *,
    demand_column: str = "demand_adjusted",
    rolling_window: int = 28,
    min_history: int = 7,
    cover_periods: float = 3.0,
    safety_periods: float = 1.0,
) -> pd.DataFrame:
    """
    FreshRetailNet demosu için satış, düzeltilmiş talep ve stokout geçmişinden
    sentetik dönem başı/dönem sonu stok hareketi üretir.

    Bu stok gerçek envanter değildir ve yalnızca demo/senaryo amaçlıdır.
    """
    _require_columns(
        demo_adjusted_df,
        {
            "series_id",
            "date",
            "store_id",
            "product_id",
            "sales",
            "is_stockout",
            demand_column,
        },
        "demo_adjusted_df",
    )

    if rolling_window <= 0 or min_history <= 0:
        raise ValueError(
            "rolling_window ve min_history pozitif olmalıdır."
        )
    if cover_periods < 0 or safety_periods < 0:
        raise ValueError(
            "cover_periods ve safety_periods negatif olamaz."
        )

    df = demo_adjusted_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df[demand_column] = pd.to_numeric(
        df[demand_column],
        errors="coerce",
    )

    if df[["sales", demand_column]].isna().any().any():
        raise ValueError("Satış veya talep sütununda geçersiz değer var.")

    df = df.sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True)

    historical_expected = (
        df.groupby("series_id", sort=False)[demand_column]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window=rolling_window,
                    min_periods=min_history,
                )
                .median()
            )
        )
    )

    df["expected_period_demand"] = (
        historical_expected
        .fillna(df[demand_column])
        .clip(lower=0)
    )

    df["synthetic_buffer_stock"] = (
        df["expected_period_demand"]
        * (cover_periods + safety_periods)
    )

    # Stokout varsa dönem sonunda stokun sıfıra düştüğü varsayılır.
    # Stokout yoksa dönem sonunda talep kapsama stoğu bırakılır.
    df["synthetic_opening_stock"] = np.where(
        df["is_stockout"].astype(bool),
        df["sales"],
        df["sales"] + df["synthetic_buffer_stock"],
    )

    df["synthetic_ending_stock"] = (
        df["synthetic_opening_stock"] - df["sales"]
    ).clip(lower=0)

    df["previous_synthetic_stock"] = (
        df.groupby("series_id", sort=False)[
            "synthetic_ending_stock"
        ]
        .shift(1)
        .fillna(0)
    )

    df["synthetic_replenishment"] = (
        df["synthetic_opening_stock"]
        - df["previous_synthetic_stock"]
    ).clip(lower=0)

    df["synthetic_inventory_adjustment"] = (
        df["previous_synthetic_stock"]
        - df["synthetic_opening_stock"]
    ).clip(lower=0)

    integer_columns = [
        "expected_period_demand",
        "synthetic_buffer_stock",
        "synthetic_opening_stock",
        "synthetic_ending_stock",
        "synthetic_replenishment",
        "synthetic_inventory_adjustment",
    ]
    df[integer_columns] = np.rint(
        df[integer_columns]
    ).astype(int)

    df["stock_source"] = "synthetic_demo"
    return df


def make_demo_current_stock(
    demo_inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Her mağaza-ürün serisinin son sentetik dönem sonu stokunu güncel stok
    olarak döndürür.
    """
    _require_columns(
        demo_inventory_df,
        {
            "series_id",
            "date",
            "store_id",
            "product_id",
            "synthetic_ending_stock",
        },
        "demo_inventory_df",
    )

    return (
        demo_inventory_df
        .sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[
            [
                "series_id",
                "store_id",
                "product_id",
                "synthetic_ending_stock",
            ]
        ]
        .rename(
            columns={
                "synthetic_ending_stock": "current_stock",
            }
        )
        .reset_index(drop=True)
    )
