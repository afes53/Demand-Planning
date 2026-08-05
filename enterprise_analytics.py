from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional
import json
import zipfile

import numpy as np
import pandas as pd

from zero_shot_demand_mvp_core_generic_v2 import (
    calculate_forecast_metrics,
)


# ---------------------------------------------------------------------
# ORTAK YARDIMCILAR
# ---------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} tablosunda eksik sütunlar: {sorted(missing)}"
        )


def safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
    default: float = 0.0,
) -> pd.Series | float:
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)

    if denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def normalise_to_unit_interval(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if np.isclose(maximum, minimum):
        return pd.Series(0.0, index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def percentile_rank(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .rank(pct=True, method="average")
    )


# ---------------------------------------------------------------------
# VERİ EŞLEMELERİ
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryExtraMapping:
    promotion: Optional[str] = None
    incoming_stock: Optional[str] = None
    order_quantity: Optional[str] = None
    lead_time: Optional[str] = None
    region: Optional[str] = None
    brand: Optional[str] = None
    unit_cost: Optional[str] = None
    profit: Optional[str] = None
    returns: Optional[str] = None
    cancellations: Optional[str] = None
    new_product: Optional[str] = None
    strategic_product: Optional[str] = None


@dataclass(frozen=True)
class DistributionPlanMapping:
    date: str
    store: str
    product: str
    starting_stock: str
    planned_shipment: str
    expected_arrival_date: Optional[str] = None
    warehouse_stock: Optional[str] = None
    store_capacity: Optional[str] = None
    price: Optional[str] = None
    region: Optional[str] = None


EXTRA_STANDARD_NAMES: dict[str, str] = {
    "promotion": "promotion",
    "incoming_stock": "incoming_stock",
    "order_quantity": "order_quantity",
    "lead_time": "lead_time",
    "region": "region",
    "brand": "brand",
    "unit_cost": "unit_cost",
    "profit": "profit",
    "returns": "returns",
    "cancellations": "cancellations",
    "new_product": "new_product",
    "strategic_product": "strategic_product",
}


def enrich_prepared_history(
    prepared_df: pd.DataFrame,
    raw_history_df: pd.DataFrame,
    *,
    raw_date_column: str,
    raw_store_column: str,
    raw_product_column: str,
    extra_mapping: HistoryExtraMapping,
) -> pd.DataFrame:
    """
    Pipeline'ın standart şemasında bulunmayan promosyon, bölge, marka,
    maliyet, tedarik süresi ve benzeri iş alanlarını tarih-mağaza-ürün
    anahtarıyla prepared_df üzerine ekler.
    """
    result = prepared_df.copy()
    source = raw_history_df.copy()

    source["date"] = pd.to_datetime(
        source[raw_date_column],
        errors="coerce",
    )
    source["store_id"] = (
        source[raw_store_column].astype(str).str.strip()
    )
    source["product_id"] = (
        source[raw_product_column].astype(str).str.strip()
    )

    selected_columns = ["date", "store_id", "product_id"]
    rename_map: dict[str, str] = {}

    for field_name, standard_name in EXTRA_STANDARD_NAMES.items():
        source_column = getattr(extra_mapping, field_name)
        if source_column is None:
            continue
        if source_column not in source.columns:
            raise ValueError(
                f"Geçmiş veri ek alanı bulunamadı: {source_column}"
            )
        selected_columns.append(source_column)
        rename_map[source_column] = standard_name

    if len(selected_columns) == 3:
        return result

    extras = (
        source[selected_columns]
        .rename(columns=rename_map)
        .drop_duplicates(
            ["date", "store_id", "product_id"],
            keep="last",
        )
    )

    boolean_columns = {
        "promotion",
        "new_product",
        "strategic_product",
    }
    numeric_columns = {
        "incoming_stock",
        "order_quantity",
        "lead_time",
        "unit_cost",
        "profit",
        "returns",
        "cancellations",
    }

    for column in boolean_columns.intersection(extras.columns):
        if extras[column].dtype != bool:
            normalised = (
                extras[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            extras[column] = normalised.isin(
                {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "evet",
                    "e",
                }
            )

    for column in numeric_columns.intersection(extras.columns):
        extras[column] = pd.to_numeric(
            extras[column],
            errors="coerce",
        )

    result = result.merge(
        extras,
        on=["date", "store_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    return result


# ---------------------------------------------------------------------
# VERİ KALİTESİ
# ---------------------------------------------------------------------

def build_data_quality_report(
    raw_history_df: pd.DataFrame,
    prepared_history_df: pd.DataFrame,
    *,
    date_column: str,
    sales_column: str,
    stock_column: Optional[str],
    stockout_column: Optional[str],
    new_product_column: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Yöneticiye gösterilecek özet kalite tablosu ve satır bazlı sorun tablosu.
    """
    raw = raw_history_df.copy()
    prepared = prepared_history_df.copy()

    parsed_dates = pd.to_datetime(
        raw[date_column],
        errors="coerce",
    )
    sales_numeric = pd.to_numeric(
        raw[sales_column],
        errors="coerce",
    )

    issue_frames: list[pd.DataFrame] = []

    invalid_date_mask = parsed_dates.isna()
    if invalid_date_mask.any():
        issue_frames.append(
            pd.DataFrame(
                {
                    "issue_type": "Geçersiz tarih",
                    "row_index": raw.index[invalid_date_mask],
                    "severity": "Yüksek",
                }
            )
        )

    missing_sales_mask = sales_numeric.isna()
    if missing_sales_mask.any():
        issue_frames.append(
            pd.DataFrame(
                {
                    "issue_type": "Eksik/geçersiz satış",
                    "row_index": raw.index[missing_sales_mask],
                    "severity": "Yüksek",
                }
            )
        )

    negative_sales_mask = sales_numeric.lt(0).fillna(False)
    if negative_sales_mask.any():
        issue_frames.append(
            pd.DataFrame(
                {
                    "issue_type": "Negatif satış",
                    "row_index": raw.index[negative_sales_mask],
                    "severity": "Yüksek",
                }
            )
        )

    if stock_column is not None and stock_column in raw.columns:
        stock_numeric = pd.to_numeric(
            raw[stock_column],
            errors="coerce",
        )
        missing_stock_count = int(stock_numeric.isna().sum())
        negative_stock_count = int(
            stock_numeric.lt(0).fillna(False).sum()
        )
    else:
        stock_numeric = pd.Series(np.nan, index=raw.index)
        missing_stock_count = len(raw)
        negative_stock_count = 0

    duplicate_count = int(
        prepared.duplicated(
            ["series_id", "date"],
            keep=False,
        ).sum()
    )

    series_count = int(prepared["series_id"].nunique())
    store_count = int(prepared["store_id"].nunique())
    product_count = int(prepared["product_id"].nunique())

    stockout_count = int(
        prepared["is_stockout"]
        .astype("boolean")
        .fillna(False)
        .sum()
    )

    if (
        new_product_column is not None
        and new_product_column in raw.columns
    ):
        new_mask = (
            raw[new_product_column]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "yes", "evet"])
        )
        product_source = (
            raw["product_id"]
            if "product_id" in raw.columns
            else raw.iloc[:, 0]
        )
        new_product_count = int(
            product_source.loc[new_mask]
            .astype(str)
            .nunique()
        )
    else:
        first_dates = (
            prepared.groupby("product_id")["date"]
            .min()
        )
        cutoff = prepared["date"].max() - pd.Timedelta(days=35)
        new_product_count = int(first_dates.ge(cutoff).sum())

    date_min = prepared["date"].min()
    date_max = prepared["date"].max()

    summary_rows = [
        {
            "Kontrol": "Tarih aralığı",
            "Sonuç": (
                f"{date_min:%Y-%m-%d} – {date_max:%Y-%m-%d}"
            ),
            "Durum": "Bilgi",
        },
        {
            "Kontrol": "Mağaza sayısı",
            "Sonuç": store_count,
            "Durum": "Bilgi",
        },
        {
            "Kontrol": "Ürün sayısı",
            "Sonuç": product_count,
            "Durum": "Bilgi",
        },
        {
            "Kontrol": "Mağaza–ürün kombinasyonu",
            "Sonuç": series_count,
            "Durum": "Bilgi",
        },
        {
            "Kontrol": "Eksik satış değeri",
            "Sonuç": f"%{missing_sales_mask.mean() * 100:.2f}",
            "Durum": (
                "İyi"
                if missing_sales_mask.mean() < 0.01
                else "Kontrol"
            ),
        },
        {
            "Kontrol": "Negatif satış",
            "Sonuç": int(negative_sales_mask.sum()),
            "Durum": (
                "İyi"
                if not negative_sales_mask.any()
                else "Kritik"
            ),
        },
        {
            "Kontrol": "Stok bilgisi eksik",
            "Sonuç": f"%{missing_stock_count / max(len(raw), 1) * 100:.2f}",
            "Durum": (
                "İyi"
                if missing_stock_count == 0
                else "Kontrol"
            ),
        },
        {
            "Kontrol": "Negatif stok",
            "Sonuç": negative_stock_count,
            "Durum": (
                "İyi"
                if negative_stock_count == 0
                else "Kritik"
            ),
        },
        {
            "Kontrol": "Stokta yok dönemleri",
            "Sonuç": stockout_count,
            "Durum": (
                "Kontrol"
                if stockout_count > 0
                else "Bilgi"
            ),
        },
        {
            "Kontrol": "Yeni ürün sayısı",
            "Sonuç": new_product_count,
            "Durum": "Bilgi",
        },
        {
            "Kontrol": "Tekrarlı seri–tarih",
            "Sonuç": duplicate_count,
            "Durum": (
                "İyi"
                if duplicate_count == 0
                else "Kritik"
            ),
        },
    ]

    issues = (
        pd.concat(issue_frames, ignore_index=True)
        if issue_frames
        else pd.DataFrame(
            columns=["issue_type", "row_index", "severity"]
        )
    )

    return pd.DataFrame(summary_rows), issues


def build_error_reason_summary(
    prepared_history_df: pd.DataFrame,
    quality_issues_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tahmin hatasının muhtemel veri/iş nedenlerini sayısallaştırır.
    """
    df = prepared_history_df.copy()
    rows: list[dict[str, Any]] = []

    stockout_series = int(
        df.loc[
            df["is_stockout"].astype("boolean").fillna(False),
            "series_id",
        ].nunique()
    )
    rows.append(
        {
            "Neden": "Stokta yok dönemleri",
            "Etkilenen kombinasyon": stockout_series,
            "Tahmini etki": (
                "Yüksek" if stockout_series > 0 else "Düşük"
            ),
        }
    )

    if "new_product" in df.columns:
        new_product_series = int(
            df.loc[
                df["new_product"]
                .astype("boolean")
                .fillna(False),
                "series_id",
            ].nunique()
        )
    else:
        first_dates = df.groupby("series_id")["date"].min()
        cutoff = df["date"].max() - pd.Timedelta(days=35)
        new_product_series = int(first_dates.ge(cutoff).sum())

    rows.append(
        {
            "Neden": "Yeni ürün",
            "Etkilenen kombinasyon": new_product_series,
            "Tahmini etki": (
                "Yüksek"
                if new_product_series > 0
                else "Düşük"
            ),
        }
    )

    if "promotion" in df.columns:
        promotion_series = int(
            df.loc[
                df["promotion"]
                .astype("boolean")
                .fillna(False),
                "series_id",
            ].nunique()
        )
    else:
        promotion_series = 0

    rows.append(
        {
            "Neden": "Promosyon",
            "Etkilenen kombinasyon": promotion_series,
            "Tahmini etki": (
                "Orta"
                if promotion_series > 0
                else "Düşük"
            ),
        }
    )

    zero_ratio_by_series = (
        df.assign(zero_sales=df["sales"].le(0))
        .groupby("series_id")["zero_sales"]
        .mean()
    )
    intermittent_count = int(
        zero_ratio_by_series.ge(0.30).sum()
    )
    rows.append(
        {
            "Neden": "Düzensiz / kesikli talep",
            "Etkilenen kombinasyon": intermittent_count,
            "Tahmini etki": (
                "Orta"
                if intermittent_count > 0
                else "Düşük"
            ),
        }
    )

    missing_issue_count = int(
        quality_issues_df[
            "issue_type"
        ].isin(
            [
                "Eksik/geçersiz satış",
                "Geçersiz tarih",
            ]
        ).sum()
        if not quality_issues_df.empty
        else 0
    )
    rows.append(
        {
            "Neden": "Eksik / geçersiz veri",
            "Etkilenen kombinasyon": missing_issue_count,
            "Tahmini etki": (
                "Yüksek"
                if missing_issue_count > 0
                else "Düşük"
            ),
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# DAĞITIM PLANI
# ---------------------------------------------------------------------

def _clean_id(
    series: pd.Series,
    name: str,
) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"'{name}' boş değer içeriyor.")
    cleaned = series.astype(str).str.strip()
    if cleaned.eq("").any():
        raise ValueError(f"'{name}' boş metin içeriyor.")
    return cleaned


def prepare_distribution_plan(
    raw_plan_df: pd.DataFrame,
    *,
    mapping: DistributionPlanMapping,
    prepared_history_df: pd.DataFrame,
    pandas_frequency: str,
) -> pd.DataFrame:
    """
    Gelecek stok/dağıtım planını standartlaştırır.

    Plan tarihleri modelin gerçek gelecek tahmin tarihleri olarak kullanılır.
    Beklenen giriş tarihi verilmişse sevkiyat o tarihte stoğa eklenir.
    """
    required_source = [
        mapping.date,
        mapping.store,
        mapping.product,
        mapping.starting_stock,
        mapping.planned_shipment,
    ]
    for column in required_source:
        if column not in raw_plan_df.columns:
            raise ValueError(
                f"Gelecek planında zorunlu sütun bulunamadı: {column}"
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
            ).fillna(0),
        }
    )

    if plan["date"].isna().any():
        raise ValueError(
            "Gelecek planı tarih sütununda boş veya geçersiz değer var."
        )
    if plan["planned_shipment"].lt(0).any():
        raise ValueError("Planlanan gönderim negatif olamaz.")
    if plan["starting_stock"].dropna().lt(0).any():
        raise ValueError("Mevcut/başlangıç stoğu negatif olamaz.")

    optional_mappings = {
        mapping.expected_arrival_date: "expected_arrival_date",
        mapping.warehouse_stock: "warehouse_stock",
        mapping.store_capacity: "store_capacity",
        mapping.price: "price",
        mapping.region: "region_plan",
    }
    for source_column, target_column in optional_mappings.items():
        if source_column is None:
            continue
        if source_column not in raw_plan_df.columns:
            raise ValueError(
                f"Gelecek planı opsiyonel sütunu bulunamadı: {source_column}"
            )
        if target_column == "expected_arrival_date":
            plan[target_column] = pd.to_datetime(
                raw_plan_df[source_column],
                errors="coerce",
            )
            plan[target_column] = plan[target_column].fillna(
                plan["date"]
            )
        elif target_column == "region_plan":
            plan[target_column] = (
                raw_plan_df[source_column]
                .astype(str)
                .str.strip()
            )
        else:
            plan[target_column] = pd.to_numeric(
                raw_plan_df[source_column],
                errors="coerce",
            )

    if "expected_arrival_date" not in plan.columns:
        plan["expected_arrival_date"] = plan["date"]

    numeric_optional = [
        column
        for column in (
            "warehouse_stock",
            "store_capacity",
            "price",
        )
        if column in plan.columns
    ]
    for column in numeric_optional:
        if plan[column].dropna().lt(0).any():
            raise ValueError(
                f"'{column}' negatif değer içeremez."
            )

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
        raise ValueError(
            "Gelecek planında aynı mağaza–ürün–tarih için "
            "birden fazla satır var."
        )

    history = prepared_history_df.copy()
    history["series_id"] = (
        history["series_id"].astype(str)
    )
    history["date"] = pd.to_datetime(
        history["date"],
        errors="raise",
    )

    plan_series = set(plan["series_id"].unique())
    history_series = set(history["series_id"].unique())
    unknown_series = sorted(plan_series - history_series)
    if unknown_series:
        raise ValueError(
            f"Gelecek plandaki {len(unknown_series)} seri geçmiş veride yok. "
            f"Örnek: {unknown_series[:5]}"
        )

    plan_dates = pd.DatetimeIndex(
        sorted(plan["date"].unique())
    )
    expected_dates = pd.date_range(
        start=plan_dates.min(),
        periods=len(plan_dates),
        freq=pandas_frequency,
    )
    if not plan_dates.equals(expected_dates):
        raise ValueError(
            "Gelecek planı tarihleri seçilen frekansta kesintisiz olmalıdır."
        )

    horizon = len(plan_dates)
    period_counts = (
        plan.groupby("series_id")["date"]
        .nunique()
    )
    if period_counts.ne(horizon).any():
        raise ValueError(
            "Her mağaza–ürün serisi aynı plan tarihlerini içermelidir."
        )

    last_history_dates = (
        history.loc[
            history["series_id"].isin(plan_series)
        ]
        .groupby("series_id")["date"]
        .max()
    )
    expected_first_dates = (
        last_history_dates
        + pd.tseries.frequencies.to_offset(
            pandas_frequency
        )
    )
    actual_first_dates = (
        plan.groupby("series_id")["date"]
        .min()
    )
    first_date_check = pd.DataFrame(
        {
            "expected": expected_first_dates,
            "actual": actual_first_dates,
        }
    )
    invalid_first = first_date_check.loc[
        first_date_check["expected"]
        .ne(first_date_check["actual"])
    ]
    if not invalid_first.empty:
        raise ValueError(
            "Gelecek planı geçmiş verinin hemen sonraki döneminde başlamalıdır. "
            f"Uyuşmayan seri sayısı: {len(invalid_first)}"
        )

    first_rows = (
        plan.sort_values("date")
        .groupby("series_id", as_index=False)
        .head(1)
    )
    if first_rows["starting_stock"].isna().any():
        raise ValueError(
            "Her mağaza–ürünün ilk gelecek tarihinde mevcut stok dolu olmalıdır."
        )

    # Başlangıç stoğu yalnızca ilk satırda kullanılacaktır.
    plan["starting_stock"] = (
        plan.sort_values(["series_id", "date"])
        .groupby("series_id")["starting_stock"]
        .transform(lambda values: values.ffill())
    )

    metadata_columns = [
        "series_id",
        "category_1",
        "category_2",
        "category_3",
        "region",
        "brand",
        "price",
        "unit_cost",
        "profit",
        "strategic_product",
        "new_product",
    ]
    available_metadata = [
        column
        for column in metadata_columns
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

    if "region_plan" in plan.columns:
        if "region" in plan.columns:
            plan["region"] = plan["region_plan"].fillna(
                plan["region"]
            )
        else:
            plan["region"] = plan["region_plan"]
        plan = plan.drop(columns=["region_plan"])

    return (
        plan.sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )


def align_forecast_with_plan(
    future_forecast_df: pd.DataFrame,
    distribution_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        future_forecast_df,
        {"series_id", "date", "predictions"},
        "future_forecast_df",
    )
    require_columns(
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

    aligned = plan.merge(
        forecast[
            [
                "series_id",
                "date",
                "predictions",
            ]
        ],
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    if aligned["predictions"].isna().any():
        raise ValueError(
            "Bazı gelecek planı tarihleri model tahminiyle eşleşmedi."
        )

    aligned["predictions"] = pd.to_numeric(
        aligned["predictions"],
        errors="coerce",
    ).clip(lower=0)

    return (
        aligned.sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )


def apply_arrival_schedule(
    aligned_plan_df: pd.DataFrame,
    *,
    arrival_delay_periods: int = 0,
    shipment_multiplier: float = 1.0,
    pandas_frequency: str = "D",
) -> pd.DataFrame:
    """
    Planlanan gönderimi beklenen giriş tarihine taşır.
    Senaryo analizinde giriş tarihine ek gecikme uygulanabilir.
    """
    if shipment_multiplier < 0:
        raise ValueError("shipment_multiplier negatif olamaz.")
    if arrival_delay_periods < 0:
        raise ValueError("arrival_delay_periods negatif olamaz.")

    plan = aligned_plan_df.copy()
    plan["date"] = pd.to_datetime(plan["date"])
    plan["expected_arrival_date"] = pd.to_datetime(
        plan.get(
            "expected_arrival_date",
            plan["date"],
        )
    )

    offset = pd.tseries.frequencies.to_offset(
        pandas_frequency
    )
    plan["effective_arrival_date"] = (
        plan["expected_arrival_date"]
        + arrival_delay_periods * offset
    )
    plan["scenario_planned_shipment"] = (
        pd.to_numeric(
            plan["planned_shipment"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        * shipment_multiplier
    )

    incoming = (
        plan.groupby(
            [
                "series_id",
                "effective_arrival_date",
            ],
            as_index=False,
        )["scenario_planned_shipment"]
        .sum()
        .rename(
            columns={
                "effective_arrival_date": "date",
                "scenario_planned_shipment": (
                    "effective_incoming_stock"
                ),
            }
        )
    )

    plan = plan.drop(
        columns=["effective_incoming_stock"],
        errors="ignore",
    ).merge(
        incoming,
        on=["series_id", "date"],
        how="left",
    )
    plan["effective_incoming_stock"] = (
        plan["effective_incoming_stock"]
        .fillna(0)
    )
    return plan


def simulate_distribution_plan(
    aligned_plan_df: pd.DataFrame,
    *,
    historical_loss_summary_df: Optional[pd.DataFrame] = None,
    safety_periods: float = 1.0,
    demand_multiplier: float = 1.0,
    shipment_multiplier: float = 1.0,
    arrival_delay_periods: int = 0,
    pandas_frequency: str = "D",
    minimum_service_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mevcut plan ve önerilen plan senaryolarını tarih bazında simüle eder.
    """
    if safety_periods < 0:
        raise ValueError("safety_periods negatif olamaz.")
    if demand_multiplier < 0:
        raise ValueError("demand_multiplier negatif olamaz.")
    if not 0 < minimum_service_level <= 1:
        raise ValueError(
            "minimum_service_level 0 ile 1 arasında olmalıdır."
        )

    plan = apply_arrival_schedule(
        aligned_plan_df,
        arrival_delay_periods=arrival_delay_periods,
        shipment_multiplier=shipment_multiplier,
        pandas_frequency=pandas_frequency,
    )

    plan["base_predictions"] = pd.to_numeric(
        plan["predictions"],
        errors="coerce",
    ).clip(lower=0)
    plan["predictions"] = (
        plan["base_predictions"]
        * demand_multiplier
    )

    safety_by_series = (
        plan.groupby("series_id")["predictions"]
        .mean()
        .mul(safety_periods)
        .rename("safety_stock")
    )
    plan = plan.merge(
        safety_by_series,
        on="series_id",
        how="left",
        validate="many_to_one",
    )

    detail_parts: list[pd.DataFrame] = []

    for _, group in plan.groupby(
        "series_id",
        sort=False,
    ):
        group = group.sort_values("date").copy()
        initial_stock = float(
            group["starting_stock"].iloc[0]
        )
        safety_stock = float(
            group["safety_stock"].iloc[0]
        )

        current_stock = initial_stock
        recommended_stock = initial_stock
        rows: list[dict[str, Any]] = []

        for row in group.itertuples(index=False):
            demand = float(row.predictions)
            incoming = float(
                row.effective_incoming_stock
            )

            opening_stock = current_stock
            available_stock = opening_stock + incoming
            fulfilled = min(demand, available_stock)
            shortage = max(demand - available_stock, 0)
            ending_stock = max(
                available_stock - demand,
                0,
            )

            recommended_opening = recommended_stock
            recommended_available = (
                recommended_opening + incoming
            )
            required_service_stock = (
                demand * minimum_service_level
                + safety_stock
            )
            extra_shipment = max(
                required_service_stock
                - recommended_available,
                0,
            )

            if (
                "store_capacity" in group.columns
                and pd.notna(
                    getattr(
                        row,
                        "store_capacity",
                        np.nan,
                    )
                )
            ):
                free_capacity = max(
                    float(row.store_capacity)
                    - recommended_available,
                    0,
                )
                extra_shipment = min(
                    extra_shipment,
                    free_capacity,
                )

            recommended_total_available = (
                recommended_available
                + extra_shipment
            )
            recommended_fulfilled = min(
                demand,
                recommended_total_available,
            )
            recommended_shortage = max(
                demand
                - recommended_total_available,
                0,
            )
            recommended_ending = max(
                recommended_total_available
                - demand,
                0,
            )

            result = row._asdict()
            result.update(
                {
                    "opening_stock": opening_stock,
                    "available_stock": available_stock,
                    "fulfilled_demand": fulfilled,
                    "period_shortage": shortage,
                    "projected_ending_stock": ending_stock,
                    "stockout_risk": shortage > 0,
                    "below_safety_stock": (
                        ending_stock < safety_stock
                    ),
                    "recommended_opening_stock": (
                        recommended_opening
                    ),
                    "recommended_extra_shipment": (
                        extra_shipment
                    ),
                    "recommended_fulfilled_demand": (
                        recommended_fulfilled
                    ),
                    "recommended_shortage": (
                        recommended_shortage
                    ),
                    "recommended_ending_stock": (
                        recommended_ending
                    ),
                }
            )

            if (
                "price" in group.columns
                and pd.notna(
                    getattr(row, "price", np.nan)
                )
            ):
                price = float(row.price)
                result["forecast_revenue"] = (
                    demand * price
                )
                result["revenue_at_risk"] = (
                    shortage * price
                )
                result["planned_shipment_value"] = (
                    incoming * price
                )
                result[
                    "recommended_extra_shipment_value"
                ] = extra_shipment * price

            if (
                "unit_cost" in group.columns
                and pd.notna(
                    getattr(
                        row,
                        "unit_cost",
                        np.nan,
                    )
                )
            ):
                cost = float(row.unit_cost)
                result["excess_stock_value"] = (
                    max(
                        ending_stock - safety_stock,
                        0,
                    )
                    * cost
                )

            rows.append(result)
            current_stock = ending_stock
            recommended_stock = recommended_ending

        detail_parts.append(pd.DataFrame(rows))

    detail = (
        pd.concat(detail_parts, ignore_index=True)
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )

    group_columns = [
        "series_id",
        "store_id",
        "product_id",
    ]
    for column in (
        "category_1",
        "category_2",
        "category_3",
        "region",
        "brand",
        "strategic_product",
        "new_product",
    ):
        if column in detail.columns:
            group_columns.append(column)

    aggregations: dict[str, tuple[str, Any]] = {
        "forecast_start": ("date", "min"),
        "forecast_end": ("date", "max"),
        "horizon_periods": ("date", "size"),
        "planned_demand": ("predictions", "sum"),
        "average_period_demand": (
            "predictions",
            "mean",
        ),
        "peak_period_demand": (
            "predictions",
            "max",
        ),
        "current_stock": (
            "starting_stock",
            "first",
        ),
        "planned_shipment_total": (
            "effective_incoming_stock",
            "sum",
        ),
        "expected_ending_stock": (
            "projected_ending_stock",
            "last",
        ),
        "expected_shortage_no_action": (
            "period_shortage",
            "sum",
        ),
        "fulfilled_demand": (
            "fulfilled_demand",
            "sum",
        ),
        "stockout_risk": (
            "stockout_risk",
            "max",
        ),
        "below_safety_stock": (
            "below_safety_stock",
            "max",
        ),
        "safety_stock": (
            "safety_stock",
            "first",
        ),
        "recommended_replenishment": (
            "recommended_extra_shipment",
            "sum",
        ),
        "recommended_fulfilled_demand": (
            "recommended_fulfilled_demand",
            "sum",
        ),
        "recommended_shortage": (
            "recommended_shortage",
            "sum",
        ),
        "recommended_ending_stock": (
            "recommended_ending_stock",
            "last",
        ),
    }

    if "forecast_revenue" in detail.columns:
        aggregations.update(
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

    if "excess_stock_value" in detail.columns:
        aggregations["excess_stock_value"] = (
            "excess_stock_value",
            "last",
        )

    if "warehouse_stock" in detail.columns:
        aggregations["warehouse_stock"] = (
            "warehouse_stock",
            "max",
        )
    if "store_capacity" in detail.columns:
        aggregations["store_capacity"] = (
            "store_capacity",
            "max",
        )

    summary = (
        detail.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(**aggregations)
    )

    summary["target_stock_for_horizon"] = (
        summary["planned_demand"]
        + summary["safety_stock"]
    )
    summary["plan_coverage_pct"] = (
        safe_divide(
            summary["fulfilled_demand"],
            summary["planned_demand"],
            default=1.0,
        )
        * 100
    )
    summary["service_level_pct"] = (
        summary["plan_coverage_pct"]
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

    summary["excess_stock_units"] = (
        summary["expected_ending_stock"]
        - summary["safety_stock"]
    ).clip(lower=0)

    summary["plan_status"] = np.select(
        [
            summary["stockout_risk"]
            .astype(bool),
            summary["below_safety_stock"]
            .astype(bool),
            summary["excess_stock_units"]
            .gt(summary["average_period_demand"] * 2),
        ],
        [
            "Kritik stok açığı",
            "Düşük stok riski",
            "Fazla stok",
        ],
        default="Güvenli",
    )

    if historical_loss_summary_df is not None:
        history_columns = [
            column
            for column in (
                "series_id",
                "stockout_rate_pct",
                "estimated_lost_demand",
                "lost_demand_share_pct",
                "estimated_lost_revenue",
                "observed_revenue",
            )
            if column
            in historical_loss_summary_df.columns
        ]
        summary = summary.merge(
            historical_loss_summary_df[
                history_columns
            ],
            on="series_id",
            how="left",
            validate="one_to_one",
        )

    demand_denominator = (
        summary["planned_demand"]
        .replace(0, np.nan)
    )
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
            pd.Series(
                0.0,
                index=summary.index,
            ),
        )
        .fillna(0)
        .div(100)
    )

    summary["priority_score"] = (
        shortage_ratio * 55
        + extra_ratio * 25
        + history_rate * 10
        + summary["stockout_risk"].astype(float) * 10
    )

    score = summary["priority_score"]
    summary["priority"] = np.select(
        [
            summary["plan_status"]
            .eq("Kritik stok açığı"),
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

    summary["recommended_action"] = np.select(
        [
            summary["plan_status"]
            .eq("Kritik stok açığı"),
            summary["plan_status"]
            .eq("Düşük stok riski"),
            summary["plan_status"]
            .eq("Fazla stok"),
        ],
        [
            (
                "Ek sevkiyat veya mağazalar arası transfer planla."
            ),
            (
                "Güvenlik stoğunu korumak için ek sevkiyatı değerlendir."
            ),
            (
                "Planlanan gönderimi azalt veya fazla stoğu riskli mağazalara aktar."
            ),
        ],
        default="Mevcut planı koru ve izlemeye devam et.",
    )

    return detail, summary.sort_values(
        [
            "priority_score",
            "expected_shortage_no_action",
        ],
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# BASELINE VE MODEL PERFORMANSI
# ---------------------------------------------------------------------

def _backtest_split(
    prepared_df: pd.DataFrame,
    *,
    horizon: int,
    min_context: int,
    max_series: Optional[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        prepared_df.groupby("series_id")
        .size()
    )
    eligible = counts.loc[
        counts >= min_context + horizon
    ].index.astype(str).tolist()

    if max_series is not None:
        eligible = eligible[:max_series]

    if not eligible:
        raise ValueError(
            "Baseline backtest için yeterli uzunlukta seri bulunamadı."
        )

    context_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    selected = prepared_df.loc[
        prepared_df["series_id"].astype(str).isin(
            eligible
        )
    ]
    for _, group in selected.groupby(
        "series_id",
        sort=False,
    ):
        group = group.sort_values("date")
        context_parts.append(
            group.iloc[:-horizon].copy()
        )
        validation_parts.append(
            group.iloc[-horizon:].copy()
        )

    return (
        pd.concat(context_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
    )


def evaluate_baselines(
    prepared_df: pd.DataFrame,
    *,
    horizon: int,
    min_context: int,
    max_series: Optional[int],
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Last Value, Seasonal Naïve ve Moving Average benchmarklarını üretir.
    """
    context, validation = _backtest_split(
        prepared_df,
        horizon=horizon,
        min_context=min_context,
        max_series=max_series,
    )

    validation = validation.rename(
        columns={"sales": "actual"}
    )

    seasonal_lag = {
        "hourly": 24,
        "daily": 7,
        "monthly": 12,
    }[frequency]

    evaluation_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for model_key, model_name in (
        ("last_value", "Geçen Dönem Değeri"),
        ("seasonal_naive", "Sezonsal Naïve"),
        ("moving_average", "Hareketli Ortalama"),
    ):
        prediction_parts: list[pd.DataFrame] = []

        for series_id, context_group in context.groupby(
            "series_id",
            sort=False,
        ):
            context_group = context_group.sort_values(
                "date"
            )
            validation_group = (
                validation.loc[
                    validation["series_id"]
                    .astype(str)
                    .eq(str(series_id))
                ]
                .sort_values("date")
                .copy()
            )

            sales = (
                pd.to_numeric(
                    context_group["sales"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy()
            )

            if model_key == "last_value":
                predictions = np.repeat(
                    sales[-1],
                    len(validation_group),
                )

            elif model_key == "seasonal_naive":
                if len(sales) >= seasonal_lag:
                    seasonal_values = sales[
                        -seasonal_lag:
                    ]
                    predictions = np.resize(
                        seasonal_values,
                        len(validation_group),
                    )
                else:
                    predictions = np.repeat(
                        sales[-1],
                        len(validation_group),
                    )

            else:
                window = min(
                    28 if frequency == "daily" else 12,
                    len(sales),
                )
                average = float(
                    np.mean(sales[-window:])
                )
                predictions = np.repeat(
                    average,
                    len(validation_group),
                )

            part = validation_group.copy()
            part["predictions"] = np.clip(
                predictions,
                0,
                None,
            )
            part["model_key"] = model_key
            part["model_name"] = model_name
            prediction_parts.append(part)

        evaluation = pd.concat(
            prediction_parts,
            ignore_index=True,
        )
        metrics = calculate_forecast_metrics(
            evaluation
        )
        metrics.update(
            {
                "model_key": model_key,
                "model_name": model_name,
                "runtime_seconds": 0.0,
                "model_type": "Benchmark",
            }
        )
        metric_rows.append(metrics)
        evaluation_frames.append(evaluation)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(
            evaluation_frames,
            ignore_index=True,
        ),
    )


def combine_model_and_baseline_metrics(
    zero_shot_metrics_df: pd.DataFrame,
    baseline_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    zero = zero_shot_metrics_df.copy()
    if not zero.empty:
        zero["model_type"] = "Zero-shot"

    combined = pd.concat(
        [baseline_metrics_df, zero],
        ignore_index=True,
        sort=False,
    )

    benchmark_wmape = (
        baseline_metrics_df["wmape_pct"].min()
        if not baseline_metrics_df.empty
        else np.nan
    )
    combined["benchmark_improvement_pct"] = np.where(
        pd.notna(benchmark_wmape)
        & combined["wmape_pct"].notna(),
        (
            benchmark_wmape
            - combined["wmape_pct"]
        )
        / benchmark_wmape
        * 100,
        np.nan,
    )

    if {
        "scored_observation_count",
        "excluded_stockout_count",
    }.issubset(combined.columns):
        total_observations = (
            combined["scored_observation_count"]
            + combined["excluded_stockout_count"]
        )
        combined["forecast_coverage_pct"] = np.where(
            total_observations > 0,
            combined["scored_observation_count"]
            / total_observations
            * 100,
            np.nan,
        )

    return combined.sort_values(
        ["wmape_pct", "mae"],
        ascending=True,
    ).reset_index(drop=True)


def add_horizon_step(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    result = evaluation_df.copy()
    result["horizon_step"] = (
        result.sort_values(
            ["model_name", "series_id", "date"]
        )
        .groupby(
            ["model_name", "series_id"]
        )
        .cumcount()
        + 1
    )
    return result


def build_horizon_performance(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    evaluation = add_horizon_step(
        evaluation_df
    )
    rows: list[dict[str, Any]] = []

    for (
        model_name,
        horizon_step,
    ), group in evaluation.groupby(
        ["model_name", "horizon_step"],
        sort=True,
    ):
        metrics = calculate_forecast_metrics(
            group
        )
        rows.append(
            {
                "model_name": model_name,
                "horizon_step": int(
                    horizon_step
                ),
                "wmape_pct": metrics[
                    "wmape_pct"
                ],
                "bias_pct": metrics[
                    "bias_pct"
                ],
                "mae": metrics["mae"],
            }
        )
    return pd.DataFrame(rows)


def build_error_heatmap_data(
    evaluation_df: pd.DataFrame,
    prepared_history_df: pd.DataFrame,
    *,
    row_dimension: str,
    column_dimension: str,
    metric: str = "wmape",
) -> pd.DataFrame:
    """
    Bölge/mağaza ile kategori/hafta ekseninde WMAPE veya Bias matrisi.
    """
    evaluation = evaluation_df.copy()
    metadata_columns = [
        "series_id",
        row_dimension,
        column_dimension,
    ]
    available = [
        column
        for column in metadata_columns
        if column in prepared_history_df.columns
    ]

    if row_dimension not in available:
        row_dimension = "store_id"
    if (
        column_dimension not in available
        and column_dimension != "week"
    ):
        column_dimension = (
            "category_1"
            if "category_1"
            in prepared_history_df.columns
            else "product_id"
        )

    metadata = (
        prepared_history_df.sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[
            list(
                dict.fromkeys(
                    [
                        "series_id",
                        row_dimension,
                        *(
                            []
                            if column_dimension == "week"
                            else [column_dimension]
                        ),
                    ]
                )
            )
        ]
        .drop_duplicates("series_id")
    )

    metadata_to_drop = [
        column
        for column in metadata.columns
        if column != "series_id"
        and column in evaluation.columns
    ]
    evaluation = evaluation.drop(
        columns=metadata_to_drop,
        errors="ignore",
    ).merge(
        metadata,
        on="series_id",
        how="left",
        validate="many_to_one",
    )

    if column_dimension == "week":
        evaluation["week"] = (
            pd.to_datetime(evaluation["date"])
            .dt.isocalendar()
            .week.astype(str)
        )

    evaluation["absolute_error"] = (
        evaluation["actual"]
        - evaluation["predictions"]
    ).abs()
    evaluation["signed_error"] = (
        evaluation["predictions"]
        - evaluation["actual"]
    )

    grouped = evaluation.groupby(
        [row_dimension, column_dimension],
        dropna=False,
    )

    if metric == "bias":
        matrix_long = grouped.agg(
            numerator=("signed_error", "sum"),
            denominator=("actual", "sum"),
        ).reset_index()
        matrix_long["metric_value"] = (
            safe_divide(
                matrix_long["numerator"],
                matrix_long["denominator"],
            )
            * 100
        )
    else:
        matrix_long = grouped.agg(
            numerator=("absolute_error", "sum"),
            denominator=("actual", "sum"),
        ).reset_index()
        matrix_long["metric_value"] = (
            safe_divide(
                matrix_long["numerator"],
                matrix_long["denominator"],
            )
            * 100
        )

    return matrix_long.pivot_table(
        index=row_dimension,
        columns=column_dimension,
        values="metric_value",
        aggfunc="mean",
    )


def build_empirical_intervals(
    future_forecast_df: pd.DataFrame,
    selected_model_evaluation_df: pd.DataFrame,
    *,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Backtest artıklarından ampirik tahmin aralığı oluşturur.
    """
    result = future_forecast_df.copy()
    evaluation = selected_model_evaluation_df.copy()

    residual = (
        evaluation["actual"]
        - evaluation["predictions"]
    )
    lower_residual = float(
        residual.quantile(lower_quantile)
    )
    upper_residual = float(
        residual.quantile(upper_quantile)
    )

    result["forecast_lower"] = (
        result["predictions"]
        + lower_residual
    ).clip(lower=0)
    result["forecast_upper"] = (
        result["predictions"]
        + upper_residual
    ).clip(lower=0)

    swap_mask = (
        result["forecast_lower"]
        > result["forecast_upper"]
    )
    if swap_mask.any():
        lower_copy = result.loc[
            swap_mask,
            "forecast_upper",
        ].copy()
        result.loc[
            swap_mask,
            "forecast_upper",
        ] = result.loc[
            swap_mask,
            "forecast_lower",
        ]
        result.loc[
            swap_mask,
            "forecast_lower",
        ] = lower_copy

    return result


# ---------------------------------------------------------------------
# ABC–XYZ
# ---------------------------------------------------------------------

def build_series_predictability(
    prepared_history_df: pd.DataFrame,
    selected_model_evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    WMAPE, Bias, sıfır satış oranı, intermittency, stokout ve oynaklığı
    birlikte kullanarak X/Y/Z öngörülebilirlik sınıfı oluşturur.
    """
    history = prepared_history_df.copy()
    evaluation = selected_model_evaluation_df.copy()

    evaluation["absolute_error"] = (
        evaluation["actual"]
        - evaluation["predictions"]
    ).abs()
    evaluation["signed_error"] = (
        evaluation["predictions"]
        - evaluation["actual"]
    )

    performance = (
        evaluation.groupby(
            "series_id",
            as_index=False,
        )
        .agg(
            absolute_error=("absolute_error", "sum"),
            actual_sum=("actual", "sum"),
            signed_error=("signed_error", "sum"),
        )
    )
    performance["wmape_pct"] = (
        safe_divide(
            performance["absolute_error"],
            performance["actual_sum"],
        )
        * 100
    )
    performance["bias_pct"] = (
        safe_divide(
            performance["signed_error"],
            performance["actual_sum"],
        )
        * 100
    )

    history_metrics = (
        history.groupby(
            ["series_id", "store_id", "product_id"],
            as_index=False,
        )
        .agg(
            mean_sales=("sales", "mean"),
            std_sales=("sales", "std"),
            zero_sales_ratio=(
                "sales",
                lambda values: values.le(0).mean(),
            ),
            stockout_rate=(
                "is_stockout",
                lambda values: (
                    values.astype("boolean")
                    .fillna(False)
                    .mean()
                ),
            ),
            positive_periods=(
                "sales",
                lambda values: values.gt(0).sum(),
            ),
            period_count=("date", "size"),
        )
    )

    history_metrics["cv"] = safe_divide(
        history_metrics["std_sales"],
        history_metrics["mean_sales"],
    )
    history_metrics["intermittency"] = (
        safe_divide(
            history_metrics["period_count"],
            history_metrics["positive_periods"],
            default=float(
                history_metrics["period_count"].max()
            ),
        )
    )

    result = history_metrics.merge(
        performance[
            [
                "series_id",
                "wmape_pct",
                "bias_pct",
            ]
        ],
        on="series_id",
        how="left",
        validate="one_to_one",
    )

    result["predictability_score"] = (
        normalise_to_unit_interval(
            result["wmape_pct"]
        )
        * 0.30
        + normalise_to_unit_interval(
            result["bias_pct"].abs()
        )
        * 0.15
        + normalise_to_unit_interval(
            result["zero_sales_ratio"]
        )
        * 0.15
        + normalise_to_unit_interval(
            result["intermittency"]
        )
        * 0.15
        + normalise_to_unit_interval(
            result["stockout_rate"]
        )
        * 0.10
        + normalise_to_unit_interval(
            result["cv"]
        )
        * 0.15
    )

    q33 = float(
        result["predictability_score"]
        .quantile(0.33)
    )
    q66 = float(
        result["predictability_score"]
        .quantile(0.66)
    )

    result["xyz_class"] = np.select(
        [
            result["predictability_score"]
            <= q33,
            result["predictability_score"]
            <= q66,
        ],
        ["X", "Y"],
        default="Z",
    )
    return result


def combine_abc_xyz(
    plan_summary_df: pd.DataFrame,
    abc_product_df: Optional[pd.DataFrame],
    predictability_df: pd.DataFrame,
) -> pd.DataFrame:
    result = plan_summary_df.copy()

    if (
        abc_product_df is not None
        and "abc_class" not in result.columns
    ):
        result = result.merge(
            abc_product_df[
                ["product_id", "abc_class"]
            ],
            on="product_id",
            how="left",
            validate="many_to_one",
        )

    if "abc_class" not in result.columns:
        result["abc_class"] = "B"

    result = result.merge(
        predictability_df[
            [
                "series_id",
                "xyz_class",
                "predictability_score",
                "wmape_pct",
                "bias_pct",
                "zero_sales_ratio",
                "intermittency",
                "stockout_rate",
            ]
        ],
        on="series_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_forecast"),
    )

    result["abc_xyz_segment"] = (
        result["abc_class"].fillna("B")
        + result["xyz_class"].fillna("Y")
    )

    segment_actions = {
        "AX": "Otomatik planlama; istisna oluşmadıkça manuel inceleme gerekmez.",
        "AY": "Otomatik planla, orta oynaklık nedeniyle periyodik kontrol et.",
        "AZ": "Yüksek iş değeri ve düşük öngörülebilirlik: insan incelemesi öncelikli.",
        "BX": "Standart otomatik planlama uygundur.",
        "BY": "Seçili risklerde planlamacı kontrolü uygula.",
        "BZ": "Yüksek sapmalı dönemleri ve promosyonları incele.",
        "CX": "Düşük iş değeri; tam otomatik yönetilebilir.",
        "CY": "Düşük dokunuşlu kontrol yeterlidir.",
        "CZ": "Düşük ekonomik değer ve zor tahmin: sınırlı manuel efor harca.",
    }
    result["segment_action"] = (
        result["abc_xyz_segment"]
        .map(segment_actions)
        .fillna("Standart inceleme uygula.")
    )

    result["human_review_priority"] = np.select(
        [
            result["abc_xyz_segment"].isin(
                ["AZ", "BZ"]
            ),
            result["abc_class"].eq("A"),
            result["xyz_class"].eq("Z"),
        ],
        [
            "1-Yüksek",
            "2-Orta",
            "3-Seçimli",
        ],
        default="4-Düşük",
    )
    return result


# ---------------------------------------------------------------------
# STOK DAĞITIM ÖNERİLERİ / TRANSFER
# ---------------------------------------------------------------------

def build_transfer_recommendations(
    plan_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aynı ürün için fazla stoklu mağazalardan açık bulunan mağazalara
    greedy transfer önerisi oluşturur. Depo stoğu varsa önce depodan tahsis eder.
    """
    require_columns(
        plan_summary_df,
        {
            "store_id",
            "product_id",
            "recommended_replenishment",
            "expected_ending_stock",
            "safety_stock",
        },
        "plan_summary_df",
    )

    summary = plan_summary_df.copy()
    summary["transferable_surplus"] = (
        summary["expected_ending_stock"]
        - summary["safety_stock"]
    ).clip(lower=0)
    summary["remaining_need"] = (
        summary["recommended_replenishment"]
        .clip(lower=0)
    )

    actions: list[dict[str, Any]] = []

    for product_id, product_group in summary.groupby(
        "product_id",
        sort=False,
    ):
        targets = (
            product_group.loc[
                product_group["remaining_need"] > 0
            ]
            .sort_values(
                [
                    "priority_score",
                    "remaining_need",
                ],
                ascending=False,
            )
            .copy()
        )
        if targets.empty:
            continue

        warehouse_stock = 0.0
        if "warehouse_stock" in product_group.columns:
            warehouse_stock = float(
                product_group["warehouse_stock"]
                .dropna()
                .max()
                if product_group[
                    "warehouse_stock"
                ].notna().any()
                else 0
            )

        # Önce depodan tahsis.
        for target_index, target in targets.iterrows():
            if warehouse_stock <= 0:
                break
            need = float(target["remaining_need"])
            quantity = min(warehouse_stock, need)
            if quantity <= 0:
                continue

            actions.append(
                {
                    "source_type": "Depo",
                    "source_location": "Merkez Depo",
                    "target_store": target["store_id"],
                    "product_id": product_id,
                    "quantity": quantity,
                    "deadline": target.get(
                        "expected_stockout_date",
                        target.get("forecast_start"),
                    ),
                    "priority": target.get(
                        "priority",
                        "Yüksek",
                    ),
                    "expected_saved_sales": quantity,
                    "expected_saved_revenue": (
                        quantity
                        * float(
                            target.get(
                                "latest_price",
                                0,
                            )
                            or 0
                        )
                    ),
                }
            )
            warehouse_stock -= quantity
            targets.at[target_index, "remaining_need"] -= (
                quantity
            )

        sources = (
            product_group.loc[
                product_group[
                    "transferable_surplus"
                ]
                > 0
            ]
            .sort_values(
                "transferable_surplus",
                ascending=False,
            )
            .copy()
        )

        for target_index, target in targets.iterrows():
            need = float(target["remaining_need"])
            if need <= 0:
                continue

            for source_index, source in sources.iterrows():
                if (
                    source["store_id"]
                    == target["store_id"]
                ):
                    continue

                surplus = float(
                    sources.at[
                        source_index,
                        "transferable_surplus",
                    ]
                )
                if surplus <= 0:
                    continue

                quantity = min(need, surplus)
                if quantity <= 0:
                    continue

                actions.append(
                    {
                        "source_type": "Mağaza Transferi",
                        "source_location": source[
                            "store_id"
                        ],
                        "target_store": target[
                            "store_id"
                        ],
                        "product_id": product_id,
                        "quantity": quantity,
                        "deadline": target.get(
                            "expected_stockout_date",
                            target.get(
                                "forecast_start"
                            ),
                        ),
                        "priority": target.get(
                            "priority",
                            "Yüksek",
                        ),
                        "expected_saved_sales": quantity,
                        "expected_saved_revenue": (
                            quantity
                            * float(
                                target.get(
                                    "latest_price",
                                    0,
                                )
                                or 0
                            )
                        ),
                    }
                )

                need -= quantity
                sources.at[
                    source_index,
                    "transferable_surplus",
                ] -= quantity

                if need <= 0:
                    break

    result = pd.DataFrame(actions)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "source_type",
                "source_location",
                "target_store",
                "product_id",
                "quantity",
                "deadline",
                "priority",
                "expected_saved_sales",
                "expected_saved_revenue",
            ]
        )

    return result.sort_values(
        [
            "priority",
            "expected_saved_revenue",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_before_after_summary(
    plan_summary_df: pd.DataFrame,
    transfer_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    current_service = (
        safe_divide(
            plan_summary_df[
                "fulfilled_demand"
            ].sum(),
            plan_summary_df[
                "planned_demand"
            ].sum(),
            default=1.0,
        )
        * 100
    )
    current_lost = float(
        plan_summary_df[
            "expected_shortage_no_action"
        ].sum()
    )
    current_excess = float(
        plan_summary_df[
            "excess_stock_units"
        ].sum()
    )

    recommended_lost = float(
        plan_summary_df.get(
            "recommended_shortage",
            pd.Series(0.0, index=plan_summary_df.index),
        ).sum()
    )
    recommended_service = (
        safe_divide(
            plan_summary_df.get(
                "recommended_fulfilled_demand",
                plan_summary_df["planned_demand"],
            ).sum(),
            plan_summary_df["planned_demand"].sum(),
            default=1.0,
        )
        * 100
    )
    recommended_excess = float(
        (
            plan_summary_df[
                "recommended_ending_stock"
            ]
            - plan_summary_df[
                "safety_stock"
            ]
        )
        .clip(lower=0)
        .sum()
    )

    transfer_quantity = (
        float(transfer_df["quantity"].sum())
        if transfer_df is not None
        and not transfer_df.empty
        else 0.0
    )

    return pd.DataFrame(
        {
            "KPI": [
                "Servis seviyesi (%)",
                "Kayıp talep (adet)",
                "Fazla stok (adet)",
                "Transfer / ek tahsis (adet)",
            ],
            "Mevcut plan": [
                current_service,
                current_lost,
                current_excess,
                0.0,
            ],
            "Önerilen plan": [
                recommended_service,
                recommended_lost,
                recommended_excess,
                transfer_quantity,
            ],
            "İyileşme": [
                recommended_service
                - current_service,
                current_lost
                - recommended_lost,
                current_excess
                - recommended_excess,
                transfer_quantity,
            ],
        }
    )


# ---------------------------------------------------------------------
# SENARYO
# ---------------------------------------------------------------------

def compare_scenarios(
    aligned_plan_df: pd.DataFrame,
    *,
    historical_loss_summary_df: Optional[pd.DataFrame],
    pandas_frequency: str,
    scenarios: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summary_rows: list[dict[str, Any]] = []
    detail_lookup: dict[str, pd.DataFrame] = {}
    summary_lookup: dict[str, pd.DataFrame] = {}

    for scenario in scenarios:
        name = str(scenario["name"])
        detail, summary = simulate_distribution_plan(
            aligned_plan_df,
            historical_loss_summary_df=(
                historical_loss_summary_df
            ),
            safety_periods=float(
                scenario.get(
                    "safety_periods",
                    1.0,
                )
            ),
            demand_multiplier=float(
                scenario.get(
                    "demand_multiplier",
                    1.0,
                )
            ),
            shipment_multiplier=float(
                scenario.get(
                    "shipment_multiplier",
                    1.0,
                )
            ),
            arrival_delay_periods=int(
                scenario.get(
                    "arrival_delay_periods",
                    0,
                )
            ),
            pandas_frequency=pandas_frequency,
            minimum_service_level=float(
                scenario.get(
                    "minimum_service_level",
                    0.95,
                )
            ),
        )

        service_level = (
            safe_divide(
                summary["fulfilled_demand"].sum(),
                summary["planned_demand"].sum(),
                default=1.0,
            )
            * 100
        )

        summary_rows.append(
            {
                "Senaryo": name,
                "Servis seviyesi (%)": service_level,
                "Kayıp talep (adet)": summary[
                    "expected_shortage_no_action"
                ].sum(),
                "Fazla stok (adet)": summary[
                    "excess_stock_units"
                ].sum(),
                "Ek sevkiyat (adet)": summary[
                    "recommended_replenishment"
                ].sum(),
                "Riskli mağaza–ürün": int(
                    summary["stockout_risk"]
                    .astype(bool)
                    .sum()
                ),
                "Kayıp satış değeri (TL)": summary.get(
                    "expected_lost_revenue_no_action",
                    pd.Series(dtype=float),
                ).sum(),
            }
        )
        detail_lookup[name] = detail
        summary_lookup[name] = summary

    return (
        pd.DataFrame(summary_rows),
        detail_lookup,
        summary_lookup,
    )


# ---------------------------------------------------------------------
# MANUEL DÜZELTME VE FVA
# ---------------------------------------------------------------------

def create_forecast_version_table(
    future_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    result = future_forecast_df[
        [
            "series_id",
            "date",
            "store_id",
            "product_id",
            "predictions",
        ]
    ].copy()
    result = result.rename(
        columns={
            "predictions": "model_forecast",
        }
    )
    result["planner_forecast"] = (
        result["model_forecast"]
    )
    result["sales_forecast"] = (
        result["planner_forecast"]
    )
    result["approved_forecast"] = (
        result["sales_forecast"]
    )
    result["change_reason"] = ""
    result["comment"] = ""
    result["changed_by"] = ""
    result["changed_at"] = pd.NaT
    return result


def build_fva_metrics(
    forecast_versions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    *,
    actual_date_column: str,
    actual_store_column: str,
    actual_product_column: str,
    actual_sales_column: str,
) -> pd.DataFrame:
    """
    Gerçekleşen veri geldikten sonra model → planlamacı → satış → final FVA.
    """
    actuals = pd.DataFrame(
        {
            "date": pd.to_datetime(
                actuals_df[
                    actual_date_column
                ],
                errors="coerce",
            ),
            "store_id": (
                actuals_df[
                    actual_store_column
                ]
                .astype(str)
                .str.strip()
            ),
            "product_id": (
                actuals_df[
                    actual_product_column
                ]
                .astype(str)
                .str.strip()
            ),
            "actual": pd.to_numeric(
                actuals_df[
                    actual_sales_column
                ],
                errors="coerce",
            ),
        }
    )
    actuals["series_id"] = (
        actuals["store_id"]
        + "||"
        + actuals["product_id"]
    )

    merged = forecast_versions_df.merge(
        actuals[
            [
                "series_id",
                "date",
                "actual",
            ]
        ],
        on=["series_id", "date"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Gerçekleşen satış dosyası tahmin tarihleriyle eşleşmedi."
        )

    stages = [
        ("Naïve", "naive_forecast"),
        ("Model", "model_forecast"),
        ("Planlamacı", "planner_forecast"),
        ("Satış", "sales_forecast"),
        ("Onaylı Final", "approved_forecast"),
    ]

    rows: list[dict[str, Any]] = []
    previous_wmape: Optional[float] = None

    for stage_name, column in stages:
        if column not in merged.columns:
            continue

        evaluation = merged[
            [
                "series_id",
                "date",
                "actual",
                column,
            ]
        ].rename(
            columns={column: "predictions"}
        )
        # Gerçekleşen gelecek satış dosyasında stokout bayrağı zorunlu
        # değildir. FVA tüm eşleşen satırlar üzerinden hesaplanır.
        evaluation["is_stockout"] = False
        metrics = calculate_forecast_metrics(
            evaluation
        )
        current_wmape = float(
            metrics["wmape_pct"]
        )
        contribution = (
            np.nan
            if previous_wmape is None
            else previous_wmape
            - current_wmape
        )
        rows.append(
            {
                "Aşama": stage_name,
                "WMAPE (%)": current_wmape,
                "Bias (%)": metrics[
                    "bias_pct"
                ],
                "Önceki aşamaya katkı (puan)": (
                    contribution
                ),
            }
        )
        previous_wmape = current_wmape

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# RAPORLAMA
# ---------------------------------------------------------------------

def dataframe_to_csv_bytes(
    df: pd.DataFrame,
) -> bytes:
    return df.to_csv(
        index=False,
    ).encode("utf-8-sig")


def build_analytics_zip(
    tables: dict[str, Optional[pd.DataFrame]],
    metadata: dict[str, Any],
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_name, table in tables.items():
            if table is None or table.empty:
                continue
            archive.writestr(
                file_name,
                dataframe_to_csv_bytes(table),
            )
        archive.writestr(
            "metadata.json",
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
    return buffer.getvalue()


def build_action_excel(
    transfer_df: pd.DataFrame,
    plan_summary_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> bytes:
    buffer = BytesIO()

    action_columns = [
        column
        for column in (
            "source_type",
            "source_location",
            "target_store",
            "product_id",
            "quantity",
            "deadline",
            "priority",
            "expected_saved_sales",
            "expected_saved_revenue",
        )
        if column in transfer_df.columns
    ]

    priority_columns = [
        column
        for column in (
            "operational_priority",
            "human_review_priority",
            "abc_xyz_segment",
            "store_id",
            "product_id",
            "plan_status",
            "current_stock",
            "planned_shipment_total",
            "planned_demand",
            "recommended_replenishment",
            "expected_stockout_date",
            "expected_lost_revenue_no_action",
            "recommended_action",
        )
        if column in plan_summary_df.columns
    ]

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        transfer_df[action_columns].to_excel(
            writer,
            sheet_name="Transfer_Aksiyonlari",
            index=False,
        )
        plan_summary_df[
            priority_columns
        ].to_excel(
            writer,
            sheet_name="Oncelikli_Riskler",
            index=False,
        )
        pd.DataFrame(
            {
                "Alan": list(metadata.keys()),
                "Değer": [
                    str(value)
                    for value in metadata.values()
                ],
            }
        ).to_excel(
            writer,
            sheet_name="Rapor_Bilgileri",
            index=False,
        )

    return buffer.getvalue()


def build_management_html(
    kpis: dict[str, Any],
    top_risks_df: pd.DataFrame,
    top_opportunities_df: pd.DataFrame,
    before_after_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> bytes:
    def table_html(
        df: pd.DataFrame,
    ) -> str:
        if df is None or df.empty:
            return "<p>Veri bulunmuyor.</p>"
        return df.to_html(
            index=False,
            border=0,
            classes="report-table",
        )

    kpi_cards = "".join(
        (
            "<div class='kpi-card'>"
            f"<div class='kpi-name'>{name}</div>"
            f"<div class='kpi-value'>{value}</div>"
            "</div>"
        )
        for name, value in kpis.items()
    )

    metadata_html = "".join(
        f"<li><strong>{key}:</strong> {value}</li>"
        for key, value in metadata.items()
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
      <meta charset="utf-8">
      <title>Demand Planning AI - Yönetim Özeti</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 32px;
          color: #172033;
        }}
        h1, h2 {{
          color: #1857b6;
        }}
        .kpi-grid {{
          display: grid;
          grid-template-columns: repeat(4, minmax(160px, 1fr));
          gap: 12px;
          margin: 20px 0;
        }}
        .kpi-card {{
          border: 1px solid #d8e1ef;
          border-radius: 10px;
          padding: 14px;
          background: #f7f9fc;
        }}
        .kpi-name {{
          font-size: 12px;
          color: #5b6578;
        }}
        .kpi-value {{
          font-size: 22px;
          font-weight: bold;
          margin-top: 6px;
        }}
        .report-table {{
          border-collapse: collapse;
          width: 100%;
          margin-bottom: 24px;
        }}
        .report-table th,
        .report-table td {{
          border: 1px solid #dde4ef;
          padding: 8px;
          text-align: left;
        }}
        .report-table th {{
          background: #eef3fa;
        }}
      </style>
    </head>
    <body>
      <h1>Demand Planning AI — Yönetim Özeti</h1>
      <div class="kpi-grid">{kpi_cards}</div>
      <h2>En Büyük Riskler</h2>
      {table_html(top_risks_df)}
      <h2>En Büyük Fırsatlar</h2>
      {table_html(top_opportunities_df)}
      <h2>Mevcut Plan / Önerilen Plan</h2>
      {table_html(before_after_df)}
      <h2>Rapor Bilgileri</h2>
      <ul>{metadata_html}</ul>
    </body>
    </html>
    """
    return html.encode("utf-8")
