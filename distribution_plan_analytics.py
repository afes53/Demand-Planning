from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionPlanMapping:
    date: str
    store: str
    product: str
    starting_stock: str
    planned_shipment: str
    price: Optional[str] = None


def _require_columns(
    df: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} tablosunda eksik sütunlar: {sorted(missing)}"
        )


def _clean_id(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"'{name}' boş değer içeriyor.")

    result = series.astype(str).str.strip()
    if result.eq("").any():
        raise ValueError(f"'{name}' boş metin içeriyor.")
    return result


def prepare_distribution_plan(
    raw_plan_df: pd.DataFrame,
    *,
    mapping: DistributionPlanMapping,
    prepared_history_df: pd.DataFrame,
    pandas_frequency: str,
) -> pd.DataFrame:
    """
    Kullanıcının geleceğe ait stok dağıtım planını standartlaştırır.

    Plan kuralları:
    - Her mağaza-ürün için aynı gelecek tarihleri bulunmalıdır.
    - İlk plan tarihi, ilgili serinin son geçmiş tarihinden sonraki ilk dönemdir.
    - İlk plan tarihinde başlangıç stoğu zorunludur.
    - Planlanan sevkiyat negatif olamaz.
    """
    source_columns = [
        mapping.date,
        mapping.store,
        mapping.product,
        mapping.starting_stock,
        mapping.planned_shipment,
    ]
    if mapping.price is not None:
        source_columns.append(mapping.price)

    missing_source_columns = [
        column
        for column in source_columns
        if column not in raw_plan_df.columns
    ]
    if missing_source_columns:
        raise ValueError(
            "Dağıtım planında bulunmayan seçilmiş sütunlar: "
            f"{missing_source_columns}"
        )

    plan = pd.DataFrame(
        {
            "date": pd.to_datetime(
                raw_plan_df[mapping.date],
                errors="coerce",
            ),
            "store_id": _clean_id(
                raw_plan_df[mapping.store],
                mapping.store,
            ),
            "product_id": _clean_id(
                raw_plan_df[mapping.product],
                mapping.product,
            ),
            "starting_stock": pd.to_numeric(
                raw_plan_df[mapping.starting_stock],
                errors="coerce",
            ),
            "planned_shipment": pd.to_numeric(
                raw_plan_df[mapping.planned_shipment],
                errors="coerce",
            ),
        }
    )

    if plan["date"].isna().any():
        raise ValueError(
            "Dağıtım planının tarih sütununda boş veya geçersiz değer var."
        )

    plan["planned_shipment"] = plan["planned_shipment"].fillna(0.0)
    if plan["planned_shipment"].lt(0).any():
        raise ValueError("Planlanan sevkiyat negatif olamaz.")
    if plan["starting_stock"].dropna().lt(0).any():
        raise ValueError("Başlangıç stoğu negatif olamaz.")

    if mapping.price is not None:
        plan["price"] = pd.to_numeric(
            raw_plan_df[mapping.price],
            errors="coerce",
        )
        if plan["price"].dropna().lt(0).any():
            raise ValueError("Gelecek planındaki fiyat negatif olamaz.")

    plan["series_id"] = (
        plan["store_id"]
        + "||"
        + plan["product_id"]
    )

    duplicate_mask = plan.duplicated(
        ["series_id", "date"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            plan.loc[
                duplicate_mask,
                ["store_id", "product_id", "date"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Aynı mağaza-ürün-tarih için birden fazla plan satırı var. "
            f"Örnekler: {examples}"
        )

    _require_columns(
        prepared_history_df,
        {
            "series_id",
            "date",
            "store_id",
            "product_id",
        },
        "prepared_history_df",
    )

    history = prepared_history_df.copy()
    history["series_id"] = history["series_id"].astype(str)
    history["date"] = pd.to_datetime(
        history["date"],
        errors="raise",
    )

    available_series = set(history["series_id"].unique())
    plan_series = set(plan["series_id"].unique())
    unknown_series = sorted(plan_series - available_series)

    if unknown_series:
        raise ValueError(
            f"Dağıtım planındaki {len(unknown_series)} mağaza-ürün serisi "
            "geçmiş satış verisinde bulunmuyor. "
            f"Örnekler: {unknown_series[:5]}"
        )

    plan_dates = pd.DatetimeIndex(
        sorted(plan["date"].drop_duplicates())
    )
    expected_grid = pd.date_range(
        start=plan_dates.min(),
        periods=len(plan_dates),
        freq=pandas_frequency,
    )

    if not plan_dates.equals(expected_grid):
        raise ValueError(
            "Dağıtım planı tarihleri seçilen veri frekansına göre kesintisiz "
            "olmalıdır."
        )

    expected_period_count = len(plan_dates)
    count_check = (
        plan.groupby("series_id")["date"]
        .nunique()
    )
    invalid_counts = count_check.loc[
        count_check.ne(expected_period_count)
    ]
    if not invalid_counts.empty:
        raise ValueError(
            "Her mağaza-ürün için aynı plan tarihleri bulunmalıdır. "
            f"Eksik/fazla dönemli seri sayısı: {len(invalid_counts)}"
        )

    first_date_check = (
        plan.groupby("series_id")["date"]
        .min()
    )
    last_history_date = (
        history.loc[
            history["series_id"].isin(plan_series)
        ]
        .groupby("series_id")["date"]
        .max()
    )

    offset = pd.tseries.frequencies.to_offset(
        pandas_frequency
    )
    expected_first_date = last_history_date + offset

    comparison = pd.DataFrame(
        {
            "plan_first_date": first_date_check,
            "expected_first_date": expected_first_date,
        }
    )
    invalid_first_date = comparison.loc[
        comparison["plan_first_date"]
        .ne(comparison["expected_first_date"])
    ]

    if not invalid_first_date.empty:
        example = (
            invalid_first_date.head(5)
            .reset_index()
            .to_dict("records")
        )
        raise ValueError(
            "Dağıtım planı, her seri için geçmiş verinin hemen sonraki "
            "döneminde başlamalıdır. "
            f"Örnek uyuşmazlıklar: {example}"
        )

    first_rows = (
        plan.sort_values("date")
        .groupby("series_id", as_index=False)
        .head(1)
    )
    if first_rows["starting_stock"].isna().any():
        examples = (
            first_rows.loc[
                first_rows["starting_stock"].isna(),
                ["store_id", "product_id", "date"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Her mağaza-ürünün ilk plan tarihinde başlangıç stoğu "
            f"bulunmalıdır. Örnekler: {examples}"
        )

    latest_metadata_columns = [
        "series_id",
        "category_1",
        "category_2",
        "category_3",
        "price",
    ]
    available_metadata = [
        column
        for column in latest_metadata_columns
        if column in history.columns
    ]

    metadata = (
        history.sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[available_metadata]
        .drop_duplicates("series_id")
    )

    plan = plan.merge(
        metadata,
        on="series_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_history"),
    )

    if "price_history" in plan.columns:
        if "price" in plan.columns:
            plan["price"] = plan["price"].fillna(
                plan["price_history"]
            )
        else:
            plan["price"] = plan["price_history"]
        plan = plan.drop(columns=["price_history"])

    plan["starting_stock"] = (
        plan.sort_values(["series_id", "date"])
        .groupby("series_id")["starting_stock"]
        .transform(lambda values: values.ffill())
    )

    return (
        plan.sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )


def align_forecast_with_distribution_plan(
    future_forecast_df: pd.DataFrame,
    distribution_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Model tahminlerini kullanıcının dağıtım planındaki mağaza-ürün-tarihlerle
    bire bir eşleştirir.
    """
    _require_columns(
        future_forecast_df,
        {"series_id", "date", "predictions"},
        "future_forecast_df",
    )
    _require_columns(
        distribution_plan_df,
        {
            "series_id",
            "date",
            "starting_stock",
            "planned_shipment",
        },
        "distribution_plan_df",
    )

    forecast = future_forecast_df.copy()
    plan = distribution_plan_df.copy()

    forecast["series_id"] = forecast["series_id"].astype(str)
    plan["series_id"] = plan["series_id"].astype(str)
    forecast["date"] = pd.to_datetime(
        forecast["date"],
        errors="raise",
    )
    plan["date"] = pd.to_datetime(
        plan["date"],
        errors="raise",
    )

    forecast_columns = [
        "series_id",
        "date",
        "predictions",
    ]
    if "model" in forecast.columns:
        forecast_columns.append("model")

    aligned = plan.merge(
        forecast[forecast_columns],
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    if aligned["predictions"].isna().any():
        missing = (
            aligned.loc[
                aligned["predictions"].isna(),
                ["store_id", "product_id", "date"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Bazı dağıtım planı tarihleri model tahminiyle eşleşmedi. "
            "Plan tarihleri geçmiş verinin hemen sonraki dönemlerinden "
            f"oluşmalıdır. Örnekler: {missing}"
        )

    aligned["predictions"] = pd.to_numeric(
        aligned["predictions"],
        errors="coerce",
    ).clip(lower=0)

    return (
        aligned.sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )


def simulate_distribution_plan(
    aligned_plan_df: pd.DataFrame,
    *,
    historical_loss_summary_df: Optional[pd.DataFrame] = None,
    safety_periods: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kullanıcının planlanan sevkiyatlarını tahmini taleple karşılaştırır.

    Mevcut plan senaryosu:
        dönem sonu stok =
            dönem başı stok + planlanan sevkiyat - tahmini talep

    Önerilen senaryo:
        Her dönemde güvenlik stoğunu korumak için gereken minimum ek sevkiyat
        hesaplanır.
    """
    _require_columns(
        aligned_plan_df,
        {
            "series_id",
            "date",
            "store_id",
            "product_id",
            "starting_stock",
            "planned_shipment",
            "predictions",
        },
        "aligned_plan_df",
    )

    if safety_periods < 0:
        raise ValueError("safety_periods negatif olamaz.")

    plan = aligned_plan_df.copy()
    plan["date"] = pd.to_datetime(
        plan["date"],
        errors="raise",
    )
    plan["predictions"] = pd.to_numeric(
        plan["predictions"],
        errors="coerce",
    ).clip(lower=0)
    plan["planned_shipment"] = pd.to_numeric(
        plan["planned_shipment"],
        errors="coerce",
    ).fillna(0).clip(lower=0)
    plan["starting_stock"] = pd.to_numeric(
        plan["starting_stock"],
        errors="coerce",
    ).clip(lower=0)

    safety_by_series = (
        plan.groupby("series_id")["predictions"]
        .mean()
        .mul(float(safety_periods))
        .rename("safety_stock")
    )
    plan = plan.merge(
        safety_by_series,
        on="series_id",
        how="left",
        validate="many_to_one",
    )

    simulated_parts: list[pd.DataFrame] = []

    for _, group in plan.groupby("series_id", sort=False):
        group = group.sort_values("date").copy()
        initial_stock = float(group["starting_stock"].iloc[0])
        safety_stock = float(group["safety_stock"].iloc[0])

        current_plan_stock = initial_stock
        recommended_stock = initial_stock
        rows: list[dict[str, object]] = []

        for row in group.itertuples(index=False):
            demand = float(row.predictions)
            planned_shipment = float(row.planned_shipment)

            opening_stock = current_plan_stock
            available_stock = opening_stock + planned_shipment
            fulfilled_demand = min(demand, available_stock)
            shortage = max(demand - available_stock, 0.0)
            ending_stock = max(available_stock - demand, 0.0)

            recommended_opening_stock = recommended_stock
            recommended_available_before_extra = (
                recommended_opening_stock + planned_shipment
            )
            recommended_extra_shipment = max(
                demand
                + safety_stock
                - recommended_available_before_extra,
                0.0,
            )
            recommended_ending_stock = (
                recommended_available_before_extra
                + recommended_extra_shipment
                - demand
            )

            result_row = row._asdict()
            result_row.update(
                {
                    "opening_stock": opening_stock,
                    "available_stock": available_stock,
                    "fulfilled_demand": fulfilled_demand,
                    "period_shortage": shortage,
                    "projected_ending_stock": ending_stock,
                    "below_safety_stock": ending_stock < safety_stock,
                    "stockout_risk": shortage > 0,
                    "recommended_opening_stock": recommended_opening_stock,
                    "recommended_extra_shipment": (
                        recommended_extra_shipment
                    ),
                    "recommended_ending_stock": (
                        recommended_ending_stock
                    ),
                }
            )

            if "price" in group.columns and pd.notna(
                getattr(row, "price", np.nan)
            ):
                price = float(getattr(row, "price"))
                result_row["forecast_revenue"] = demand * price
                result_row["revenue_at_risk"] = shortage * price
                result_row["planned_shipment_value"] = (
                    planned_shipment * price
                )
                result_row["recommended_extra_shipment_value"] = (
                    recommended_extra_shipment * price
                )

            rows.append(result_row)
            current_plan_stock = ending_stock
            recommended_stock = recommended_ending_stock

        simulated_parts.append(pd.DataFrame(rows))

    detail = pd.concat(
        simulated_parts,
        ignore_index=True,
    ).sort_values(["series_id", "date"])

    group_columns = ["series_id", "store_id", "product_id"]
    for category in (
        "category_1",
        "category_2",
        "category_3",
    ):
        if category in detail.columns:
            group_columns.append(category)

    summary_aggregations: dict[str, tuple[str, object]] = {
        "forecast_start": ("date", "min"),
        "forecast_end": ("date", "max"),
        "horizon_periods": ("date", "size"),
        "planned_demand": ("predictions", "sum"),
        "average_period_demand": ("predictions", "mean"),
        "peak_period_demand": ("predictions", "max"),
        "current_stock": ("starting_stock", "first"),
        "planned_shipment_total": ("planned_shipment", "sum"),
        "expected_ending_stock": (
            "projected_ending_stock",
            "last",
        ),
        "expected_shortage_no_action": (
            "period_shortage",
            "sum",
        ),
        "fulfilled_demand": ("fulfilled_demand", "sum"),
        "stockout_risk": ("stockout_risk", "max"),
        "below_safety_stock": (
            "below_safety_stock",
            "max",
        ),
        "safety_stock": ("safety_stock", "first"),
        "recommended_replenishment": (
            "recommended_extra_shipment",
            "sum",
        ),
        "recommended_ending_stock": (
            "recommended_ending_stock",
            "last",
        ),
    }

    if "forecast_revenue" in detail.columns:
        summary_aggregations.update(
            {
                "planned_revenue": (
                    "forecast_revenue",
                    "sum",
                ),
                "expected_lost_revenue_no_action": (
                    "revenue_at_risk",
                    "sum",
                ),
                "planned_shipment_value": (
                    "planned_shipment_value",
                    "sum",
                ),
                "recommended_replenishment_value": (
                    "recommended_extra_shipment_value",
                    "sum",
                ),
                "latest_price": ("price", "last"),
            }
        )

    summary = (
        detail.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(**summary_aggregations)
    )

    summary["target_stock_for_horizon"] = (
        summary["planned_demand"]
        + summary["safety_stock"]
    )
    summary["plan_coverage_pct"] = np.where(
        summary["planned_demand"] > 0,
        summary["fulfilled_demand"]
        / summary["planned_demand"]
        * 100,
        100.0,
    )

    first_stockout = (
        detail.loc[detail["stockout_risk"]]
        .groupby("series_id")["date"]
        .min()
        .rename("expected_stockout_date")
    )
    summary = summary.merge(
        first_stockout,
        on="series_id",
        how="left",
    )

    summary["plan_status"] = np.select(
        [
            summary["stockout_risk"].astype(bool),
            summary["below_safety_stock"].astype(bool),
        ],
        [
            "Yetersiz",
            "Güvenlik stoğu altında",
        ],
        default="Yeterli",
    )

    if historical_loss_summary_df is not None:
        history_columns = [
            "series_id",
            "stockout_rate_pct",
            "estimated_lost_demand",
            "lost_demand_share_pct",
        ]
        if (
            "estimated_lost_revenue"
            in historical_loss_summary_df.columns
        ):
            history_columns.append(
                "estimated_lost_revenue"
            )

        available_history_columns = [
            column
            for column in history_columns
            if column in historical_loss_summary_df.columns
        ]

        summary = summary.merge(
            historical_loss_summary_df[
                available_history_columns
            ],
            on="series_id",
            how="left",
            validate="one_to_one",
        )

    demand_denominator = summary[
        "planned_demand"
    ].replace(0, np.nan)

    shortage_ratio = (
        summary["expected_shortage_no_action"]
        / demand_denominator
    ).fillna(0)
    extra_ratio = (
        summary["recommended_replenishment"]
        / demand_denominator
    ).fillna(0)
    history_rate = (
        summary.get(
            "stockout_rate_pct",
            pd.Series(0, index=summary.index),
        )
        .fillna(0)
        .div(100)
    )

    summary["priority_score"] = (
        shortage_ratio * 60
        + extra_ratio * 30
        + history_rate * 10
    )

    score = summary["priority_score"]
    summary["priority"] = np.select(
        [
            summary["plan_status"].eq("Yetersiz"),
            score >= score.quantile(0.70),
            score >= score.quantile(0.40),
        ],
        [
            "Kritik",
            "Yüksek",
            "Orta",
        ],
        default="Düşük",
    )

    def _action(row: pd.Series) -> str:
        extra = float(row["recommended_replenishment"])
        if row["plan_status"] == "Yetersiz":
            return (
                f"Dağıtım planı talebi karşılamıyor. "
                f"Toplam {extra:,.0f} birim ek sevkiyat planla."
            )
        if row["plan_status"] == "Güvenlik stoğu altında":
            return (
                f"Plan talebi karşılıyor ancak güvenlik stoğu düşük. "
                f"{extra:,.0f} birim ek sevkiyat değerlendir."
            )
        return "Mevcut dağıtım planı tahmini talebi ve güvenlik stoğunu karşılıyor."

    summary["recommended_action"] = summary.apply(
        _action,
        axis=1,
    )

    return (
        detail.reset_index(drop=True),
        summary.sort_values(
            [
                "priority_score",
                "expected_shortage_no_action",
            ],
            ascending=False,
        ).reset_index(drop=True),
    )
