from __future__ import annotations

import gc
import importlib.util
import io
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from zero_shot_demand_mvp_core_generic_v2 import (
    MODEL_CONFIGS,
    ColumnMapping,
    DataValidationError,
    DemandDataPipeline,
    DemandForecastMVP,
    compare_zero_shot_models,
    create_forecaster,
)

from demand_business_analytics_fixed import (
    add_abc_stockout_action,
    add_abc_to_demand_plan,
    apply_revenue_weighted_priority,
    build_abc_analysis,
    build_abc_management_summary,
    build_historical_abc_analysis,
    build_historical_loss_analysis,
    build_management_kpis,
)

from enterprise_analytics import (
    DistributionPlanMapping,
    HistoryExtraMapping,
    align_forecast_with_plan,
    build_action_excel,
    build_analytics_zip,
    build_before_after_summary,
    build_data_quality_report,
    build_empirical_intervals,
    build_error_heatmap_data,
    build_error_reason_summary,
    build_fva_metrics,
    build_horizon_performance,
    build_management_html,
    build_series_predictability,
    build_transfer_recommendations,
    combine_abc_xyz,
    combine_model_and_baseline_metrics,
    compare_scenarios,
    create_forecast_version_table,
    enrich_prepared_history,
    evaluate_baselines,
    prepare_distribution_plan,
    simulate_distribution_plan,
)


st.set_page_config(
    page_title="Demand Planning AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
HISTORY_SAMPLE_PATH = APP_DIR / "ornek_gecmis_satis_v4.csv"
PLAN_SAMPLE_PATH = APP_DIR / "ornek_gelecek_stok_plani_v4.csv"
NONE_OPTION = "— Yok —"

PAGES = [
    "1. Ana Sayfa",
    "2. Veri Yükleme",
    "3. Veri Kalitesi",
    "4. Talep Tahmini",
    "5. Tahmin Performansı",
    "6. Stok Riskleri",
    "7. Stok Dağıtım Önerileri",
    "8. Mağaza / Ürün Detayı",
    "9. ABC–XYZ Önceliklendirme",
    "10. Senaryo Analizi",
    "11. Manuel Düzeltme ve FVA",
    "12. Raporlar",
    "13. Model ve Veri Bilgileri",
]

ROLE_PAGE_HINTS = {
    "Üst Yönetim": [
        "1. Ana Sayfa",
        "6. Stok Riskleri",
        "7. Stok Dağıtım Önerileri",
        "12. Raporlar",
    ],
    "Talep Planlama Yöneticisi": [
        "1. Ana Sayfa",
        "4. Talep Tahmini",
        "5. Tahmin Performansı",
        "9. ABC–XYZ Önceliklendirme",
        "11. Manuel Düzeltme ve FVA",
    ],
    "Operasyon / Lojistik": [
        "6. Stok Riskleri",
        "7. Stok Dağıtım Önerileri",
        "8. Mağaza / Ürün Detayı",
        "10. Senaryo Analizi",
    ],
    "Veri Bilimci": [
        "3. Veri Kalitesi",
        "4. Talep Tahmini",
        "5. Tahmin Performansı",
        "13. Model ve Veri Bilgileri",
    ],
}

FREQUENCY_DEFAULTS = {
    "hourly": {
        "backtest_horizon": 24,
        "min_context": 168,
    },
    "daily": {
        "backtest_horizon": 7,
        "min_context": 60,
    },
    "monthly": {
        "backtest_horizon": 3,
        "min_context": 24,
    },
}

COLUMN_ALIASES = {
    "date": [
        "Tarih", "Date", "date", "dt", "timestamp",
    ],
    "store": [
        "Magaza_ID", "Mağaza_ID", "Store_ID",
        "store_id", "Store ID",
    ],
    "product": [
        "Urun_ID", "Ürün_ID", "Product_ID",
        "product_id", "Product ID", "SKU",
    ],
    "sales": [
        "Satis_Adedi", "Satış_Adedi", "Units_Sold",
        "sales", "sale_amount", "quantity",
    ],
    "stock": [
        "Stok_Miktari", "Stok_Miktarı",
        "Inventory_Level", "stock", "inventory",
    ],
    "price": [
        "Birim_Fiyat", "Birim Fiyat", "Price",
        "price", "unit_price",
    ],
    "stockout": [
        "Stokout_Flag", "Stockout_Flag",
        "stockout_flag", "is_stockout",
    ],
    "category_1": [
        "Ana_Kategori", "Category", "category",
        "category_1",
    ],
    "category_2": [
        "Alt_Kategori", "Subcategory",
        "subcategory", "category_2",
    ],
    "category_3": [
        "Marka_Grubu", "category_3", "Brand_Group",
    ],
    "promotion": [
        "Promosyon_Flag", "Promotion", "promotion",
        "promo_flag",
    ],
    "incoming_stock": [
        "Gelen_Stok", "Incoming_Stock", "incoming_stock",
    ],
    "order_quantity": [
        "Siparis_Miktari", "Order_Quantity",
        "order_quantity",
    ],
    "lead_time": [
        "Tedarik_Suresi_Gun", "Lead_Time",
        "lead_time",
    ],
    "region": [
        "Bolge", "Bölge", "Region", "region",
    ],
    "brand": [
        "Marka", "Brand", "brand", "Marka_Grubu",
    ],
    "unit_cost": [
        "Birim_Maliyet", "Unit_Cost", "unit_cost",
    ],
    "profit": [
        "Kar", "Kâr", "Profit", "profit", "margin",
    ],
    "returns": [
        "Iade_Miktari", "İade_Miktarı",
        "Returns", "returns",
    ],
    "cancellations": [
        "Iade_Iptal_Miktari", "Iptal_Miktari",
        "Cancellations", "cancellations",
    ],
    "new_product": [
        "Yeni_Urun", "New_Product", "new_product",
    ],
    "strategic_product": [
        "Stratejik_Urun", "Strategic_Product",
        "strategic_product",
    ],
    "starting_stock": [
        "Baslangic_Stoku", "Başlangıç_Stoku",
        "Mevcut_Stok", "Starting_Stock",
        "current_stock",
    ],
    "planned_shipment": [
        "Planlanan_Sevkiyat", "Planned_Shipment",
        "planned_shipment", "shipment",
    ],
    "arrival_date": [
        "Beklenen_Giris_Tarihi", "Expected_Arrival_Date",
        "arrival_date",
    ],
    "warehouse_stock": [
        "Dagitilabilir_Depo_Stogu",
        "Dağıtılabilir_Depo_Stoku",
        "Warehouse_Stock", "warehouse_stock",
    ],
    "capacity": [
        "Magaza_Kapasitesi", "Mağaza_Kapasitesi",
        "Store_Capacity", "store_capacity",
    ],
}

DISPLAY_NAMES = {
    "date": "Tarih",
    "store_id": "Mağaza",
    "product_id": "Ürün",
    "series_id": "Mağaza–Ürün",
    "category_1": "Ana Kategori",
    "category_2": "Alt Kategori",
    "category_3": "Ürün Grubu",
    "region": "Bölge",
    "brand": "Marka",
    "sales": "Gerçekleşen Satış",
    "demand_adjusted": "Düzeltilmiş Talep",
    "stock": "Stok",
    "price": "Birim Fiyat (TL)",
    "is_stockout": "Stokta Yok",
    "promotion": "Promosyon",
    "predictions": "Model Tahmini",
    "forecast_lower": "Alt Güven Sınırı",
    "forecast_upper": "Üst Güven Sınırı",
    "starting_stock": "Başlangıç Stoğu",
    "opening_stock": "Dönem Başı Stok",
    "planned_shipment": "Planlanan Gönderim",
    "effective_incoming_stock": "Gerçekleşecek Giriş",
    "available_stock": "Kullanılabilir Stok",
    "fulfilled_demand": "Karşılanan Talep",
    "period_shortage": "Karşılanamayan Talep",
    "projected_ending_stock": "Dönem Sonu Stok",
    "recommended_extra_shipment": "Önerilen Ek Gönderim",
    "recommended_ending_stock": "Öneri Sonrası Stok",
    "safety_stock": "Güvenlik Stoğu",
    "stockout_risk": "Stok Riski",
    "below_safety_stock": "Güvenlik Stoğu Altında",
    "forecast_start": "Plan Başlangıcı",
    "forecast_end": "Plan Bitişi",
    "planned_demand": "Tahmini Talep",
    "current_stock": "Mevcut Stok",
    "planned_shipment_total": "Planlanan Toplam Gönderim",
    "expected_ending_stock": "Plan Sonu Stok",
    "expected_shortage_no_action": "Karşılanamayan Talep",
    "recommended_replenishment": "Önerilen Ek Gönderim",
    "expected_stockout_date": "Tahmini Stok Tükenme Tarihi",
    "plan_coverage_pct": "Plan Karşılama Oranı (%)",
    "service_level_pct": "Servis Seviyesi (%)",
    "plan_status": "Durum",
    "excess_stock_units": "Fazla Stok",
    "excess_stock_value": "Fazla Stok Değeri (TL)",
    "planned_revenue": "Tahmini Satış Değeri (TL)",
    "expected_lost_revenue_no_action": "Kayıp Satış Riski (TL)",
    "recommended_replenishment_value": "Ek Gönderim Değeri (TL)",
    "stockout_rate_pct": "Tarihsel Stokta Yok Oranı (%)",
    "estimated_lost_demand": "Tarihsel Kayıp Talep",
    "estimated_lost_revenue": "Tarihsel Kayıp Satış (TL)",
    "priority": "Öncelik",
    "operational_priority": "Operasyon Önceliği",
    "recommended_action": "Önerilen Aksiyon",
    "abc_class": "ABC",
    "xyz_class": "XYZ",
    "abc_xyz_segment": "ABC–XYZ",
    "human_review_priority": "İnsan İnceleme Önceliği",
    "segment_action": "Segment Yaklaşımı",
    "wmape_pct": "WMAPE (%)",
    "bias_pct": "Bias (%)",
    "mae": "MAE",
    "rmse": "RMSE",
    "model_name": "Model",
    "model_type": "Tür",
    "benchmark_improvement_pct": "Benchmark İyileşmesi (%)",
    "runtime_seconds": "Çalışma Süresi (sn)",
    "forecast_coverage_pct": "Tahmin Kapsama Oranı (%)",
    "horizon_step": "Tahmin Ufku",
    "total_value": "Ticari Değer (TL)",
    "value_share_pct": "Değer Payı (%)",
    "cumulative_value_pct": "Kümülatif Değer (%)",
    "predictability_score": "Öngörülebilirlik Risk Puanı",
    "zero_sales_ratio": "Sıfır Satış Oranı",
    "intermittency": "Intermittency",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.28);
            background: rgba(248,250,252,.80);
            border-radius: 14px;
            padding: 14px 16px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 12px;
            overflow: hidden;
        }
        .hero {
            border: 1px solid rgba(148,163,184,.25);
            border-radius: 18px;
            padding: 22px 24px;
            background: linear-gradient(135deg, rgba(238,244,255,.95), rgba(248,250,252,.96));
            margin-bottom: 18px;
        }
        .hero h2 {
            margin-top: 0;
            color: #1857b6;
        }
        .small-note {
            color: #64748b;
            font-size: .88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "history_raw_df": None,
        "plan_raw_df": None,
        "pipeline": None,
        "prepared_history_df": None,
        "analysis_history_df": None,
        "future_plan_df": None,
        "quality_summary_df": None,
        "quality_issues_df": None,
        "error_reason_df": None,
        "mapping_metadata": None,
        "data_label": None,
        "zero_shot_metrics_df": None,
        "baseline_metrics_df": None,
        "all_metrics_df": None,
        "zero_shot_evaluations_df": None,
        "baseline_evaluations_df": None,
        "all_evaluations_df": None,
        "model_errors_df": None,
        "selected_model_key": None,
        "selected_model_name": None,
        "future_forecast_df": None,
        "aligned_plan_df": None,
        "historical_loss_detail_df": None,
        "historical_loss_summary_df": None,
        "plan_detail_df": None,
        "plan_summary_df": None,
        "management_kpis_df": None,
        "abc_product_df": None,
        "abc_summary_df": None,
        "predictability_df": None,
        "abc_xyz_df": None,
        "transfer_df": None,
        "before_after_df": None,
        "scenario_comparison_df": None,
        "scenario_detail_lookup": None,
        "scenario_summary_lookup": None,
        "forecast_versions_df": None,
        "fva_metrics_df": None,
        "manual_adjustment_exists": False,
        "run_metadata": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_outputs() -> None:
    for key in (
        "zero_shot_metrics_df",
        "baseline_metrics_df",
        "all_metrics_df",
        "zero_shot_evaluations_df",
        "baseline_evaluations_df",
        "all_evaluations_df",
        "model_errors_df",
        "selected_model_key",
        "selected_model_name",
        "future_forecast_df",
        "aligned_plan_df",
        "historical_loss_detail_df",
        "historical_loss_summary_df",
        "plan_detail_df",
        "plan_summary_df",
        "management_kpis_df",
        "abc_product_df",
        "abc_summary_df",
        "predictability_df",
        "abc_xyz_df",
        "transfer_df",
        "before_after_df",
        "scenario_comparison_df",
        "scenario_detail_lookup",
        "scenario_summary_lookup",
        "forecast_versions_df",
        "fva_metrics_df",
        "manual_adjustment_exists",
        "run_metadata",
    ):
        st.session_state[key] = None


def reset_everything() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialise_state()


def format_number(
    value: Any,
    decimals: int = 0,
) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = f"{float(value):,.{decimals}f}"
    return (
        text.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def format_try(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{format_number(value, 2)} TL"


def normalise_name(value: str) -> str:
    translation = str.maketrans(
        {
            "ı": "i", "İ": "i",
            "ş": "s", "Ş": "s",
            "ğ": "g", "Ğ": "g",
            "ü": "u", "Ü": "u",
            "ö": "o", "Ö": "o",
            "ç": "c", "Ç": "c",
        }
    )
    return (
        str(value)
        .translate(translation)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def guess_column(
    columns: list[str],
    field: str,
) -> Optional[str]:
    lookup = {
        normalise_name(column): column
        for column in columns
    }
    for alias in COLUMN_ALIASES[field]:
        match = lookup.get(normalise_name(alias))
        if match is not None:
            return match
    return None


def select_column(
    label: str,
    columns: list[str],
    field: str,
    *,
    optional: bool,
    key: str,
) -> Optional[str]:
    options = (
        [NONE_OPTION, *columns]
        if optional
        else columns
    )
    guessed = guess_column(columns, field)
    default = (
        guessed if guessed in options else options[0]
    )
    selected = st.selectbox(
        label,
        options,
        index=options.index(default),
        key=key,
    )
    if optional and selected == NONE_OPTION:
        return None
    return selected


@st.cache_data(show_spinner=False)
def read_uploaded_table(
    file_bytes: bytes,
    file_name: str,
) -> pd.DataFrame:
    extension = (
        file_name.lower().rsplit(".", maxsplit=1)[-1]
    )
    stream = io.BytesIO(file_bytes)
    if extension == "csv":
        return pd.read_csv(
            stream,
            sep=None,
            engine="python",
        )
    if extension == "xlsx":
        return pd.read_excel(stream)
    if extension == "parquet":
        return pd.read_parquet(stream)
    raise ValueError(
        "Desteklenen dosyalar CSV, XLSX ve Parquet."
    )


def prepare_display_table(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    *,
    rows: Optional[int] = None,
) -> pd.DataFrame:
    selected = (
        [
            column
            for column in columns
            if column in df.columns
        ]
        if columns is not None
        else list(df.columns)
    )
    result = df[selected].copy()

    for column in result.columns:
        if (
            "date" in column.lower()
            or column
            in {
                "forecast_start",
                "forecast_end",
                "expected_stockout_date",
                "deadline",
            }
        ):
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).dt.strftime("%d.%m.%Y")
        elif (
            pd.api.types.is_bool_dtype(result[column])
            or str(result[column].dtype)
            == "boolean"
        ):
            result[column] = (
                result[column]
                .map({True: "Evet", False: "Hayır"})
                .fillna("—")
            )
        elif pd.api.types.is_numeric_dtype(
            result[column]
        ):
            result[column] = result[column].round(2)

    result = result.rename(columns=DISPLAY_NAMES)
    return (
        result.head(rows)
        if rows is not None
        else result
    )


def data_ready() -> bool:
    return (
        st.session_state.prepared_history_df
        is not None
        and st.session_state.future_plan_df
        is not None
    )


def analysis_ready() -> bool:
    return (
        st.session_state.plan_summary_df
        is not None
    )


def add_metadata_to_evaluation(
    evaluation_df: pd.DataFrame,
    prepared_history_df: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "series_id",
        "store_id",
        "product_id",
        "category_1",
        "category_2",
        "category_3",
        "region",
        "brand",
    ]
    available = [
        column
        for column in metadata_columns
        if column in prepared_history_df.columns
    ]
    metadata = (
        prepared_history_df.sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[available]
        .drop_duplicates("series_id")
    )
    drop_columns = [
        column
        for column in available
        if column != "series_id"
        and column in evaluation_df.columns
    ]
    result = evaluation_df.drop(
        columns=drop_columns,
        errors="ignore",
    )
    return result.merge(
        metadata,
        on="series_id",
        how="left",
        validate="many_to_one",
    )


def create_future_naive_forecast(
    prepared_history_df: pd.DataFrame,
    future_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    latest_sales = (
        prepared_history_df.sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[["series_id", "sales"]]
        .rename(columns={"sales": "naive_forecast"})
    )
    return future_forecast_df.merge(
        latest_sales,
        on="series_id",
        how="left",
        validate="many_to_one",
    )


def run_zero_shot_pipeline(
    *,
    model_keys: list[str],
    backtest_horizon: int,
    min_context: int,
    safety_periods: float,
    minimum_service_level: float,
) -> None:
    prepared = (
        st.session_state.prepared_history_df
    )
    plan = st.session_state.future_plan_df
    pipeline = st.session_state.pipeline

    if prepared is None or plan is None or pipeline is None:
        raise ValueError(
            "Önce geçmiş veri ve gelecek planı hazırlayın."
        )

    plan_series_ids = (
        plan["series_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    model_history = prepared.loc[
        prepared["series_id"]
        .astype(str)
        .isin(plan_series_ids)
    ].copy()

    max_series = len(plan_series_ids)
    plan_horizon = int(plan["date"].nunique())

    baseline_metrics, baseline_evaluations = (
        evaluate_baselines(
            model_history,
            horizon=backtest_horizon,
            min_context=min_context,
            max_series=max_series,
            frequency=pipeline.frequency,
        )
    )

    (
        zero_metrics,
        zero_evaluations,
        model_errors,
    ) = compare_zero_shot_models(
        prepared_df=model_history,
        data_pipeline=pipeline,
        model_keys=model_keys,
        horizon=backtest_horizon,
        min_context=min_context,
        max_series=max_series,
    )

    if zero_metrics.empty:
        raise RuntimeError(
            "Seçilen zero-shot modeller çalışmadı."
        )

    zero_metrics = zero_metrics.sort_values(
        ["wmape_pct", "mae"],
        ascending=True,
    ).reset_index(drop=True)

    selected_model_key = str(
        zero_metrics.iloc[0]["model_key"]
    )
    selected_model_name = str(
        zero_metrics.iloc[0]["model_name"]
    )

    forecaster = None
    mvp = None
    try:
        forecaster = create_forecaster(
            selected_model_key,
            freq=pipeline.pandas_freq,
        )
        mvp = DemandForecastMVP(
            pipeline,
            forecaster,
        )
        future_forecast = mvp.forecast(
            prepared_df=model_history,
            horizon=plan_horizon,
            min_context=min_context,
            max_series=max_series,
        )
    finally:
        if mvp is not None:
            del mvp
        if forecaster is not None:
            del forecaster
        gc.collect()

    zero_evaluations = add_metadata_to_evaluation(
        zero_evaluations,
        model_history,
    )
    baseline_evaluations = add_metadata_to_evaluation(
        baseline_evaluations,
        model_history,
    )
    all_evaluations = pd.concat(
        [
            baseline_evaluations,
            zero_evaluations,
        ],
        ignore_index=True,
        sort=False,
    )

    all_metrics = combine_model_and_baseline_metrics(
        zero_metrics,
        baseline_metrics,
    )

    selected_evaluation = zero_evaluations.loc[
        zero_evaluations["model_key"]
        .astype(str)
        .eq(selected_model_key)
    ].copy()

    future_forecast = build_empirical_intervals(
        future_forecast,
        selected_evaluation,
    )
    future_forecast = create_future_naive_forecast(
        model_history,
        future_forecast,
    )

    aligned = align_forecast_with_plan(
        future_forecast,
        plan,
    )
    aligned = aligned.merge(
        future_forecast[
            [
                "series_id",
                "date",
                "forecast_lower",
                "forecast_upper",
                "naive_forecast",
            ]
        ],
        on=["series_id", "date"],
        how="left",
        validate="one_to_one",
    )

    (
        historical_loss_detail,
        historical_loss_summary,
    ) = build_historical_loss_analysis(
        st.session_state.analysis_history_df
    )

    plan_detail, plan_summary = (
        simulate_distribution_plan(
            aligned,
            historical_loss_summary_df=(
                historical_loss_summary
            ),
            safety_periods=safety_periods,
            minimum_service_level=(
                minimum_service_level
            ),
            pandas_frequency=(
                pipeline.pandas_freq
            ),
        )
    )

    has_revenue = (
        "planned_revenue" in plan_summary.columns
        and "expected_lost_revenue_no_action"
        in plan_summary.columns
    )

    abc_product = None
    abc_summary = None

    if has_revenue:
        original_action = plan_summary[
            ["series_id", "recommended_action"]
        ].rename(
            columns={
                "recommended_action":
                    "plan_recommended_action"
            }
        )
        plan_summary = (
            apply_revenue_weighted_priority(
                plan_summary
            )
        )
        plan_summary = plan_summary.merge(
            original_action,
            on="series_id",
            how="left",
            validate="one_to_one",
        )
        plan_summary["recommended_action"] = (
            plan_summary[
                "plan_recommended_action"
            ]
        )
        plan_summary = plan_summary.drop(
            columns=["plan_recommended_action"]
        )

        if {
            "observed_revenue",
            "estimated_lost_revenue",
        }.issubset(
            historical_loss_summary.columns
        ):
            abc_product = (
                build_historical_abc_analysis(
                    historical_loss_summary
                )
            )
        else:
            abc_input = plan_summary[
                ["product_id", "planned_revenue"]
            ].rename(
                columns={
                    "planned_revenue":
                        "commercial_value"
                }
            )
            abc_product = build_abc_analysis(
                abc_input,
                item_column="product_id",
                value_column="commercial_value",
            )

        plan_summary = add_abc_to_demand_plan(
            plan_summary,
            abc_product,
        )
        plan_summary = add_abc_stockout_action(
            plan_summary
        )
        abc_summary = (
            build_abc_management_summary(
                plan_summary
            )
        )

    predictability = build_series_predictability(
        model_history,
        selected_evaluation,
    )
    abc_xyz = combine_abc_xyz(
        plan_summary,
        abc_product,
        predictability,
    )

    transfer_df = build_transfer_recommendations(
        abc_xyz
    )
    before_after = build_before_after_summary(
        abc_xyz,
        transfer_df,
    )

    management_kpis = build_management_kpis(
        historical_loss_summary,
        abc_xyz,
    )

    forecast_versions = create_forecast_version_table(
        future_forecast
    )
    forecast_versions["naive_forecast"] = (
        future_forecast[
            "naive_forecast"
        ].to_numpy()
    )

    st.session_state.zero_shot_metrics_df = (
        zero_metrics
    )
    st.session_state.baseline_metrics_df = (
        baseline_metrics
    )
    st.session_state.all_metrics_df = all_metrics
    st.session_state.zero_shot_evaluations_df = (
        zero_evaluations
    )
    st.session_state.baseline_evaluations_df = (
        baseline_evaluations
    )
    st.session_state.all_evaluations_df = (
        all_evaluations
    )
    st.session_state.model_errors_df = (
        model_errors
    )
    st.session_state.selected_model_key = (
        selected_model_key
    )
    st.session_state.selected_model_name = (
        selected_model_name
    )
    st.session_state.future_forecast_df = (
        future_forecast
    )
    st.session_state.aligned_plan_df = aligned
    st.session_state.historical_loss_detail_df = (
        historical_loss_detail
    )
    st.session_state.historical_loss_summary_df = (
        historical_loss_summary
    )
    st.session_state.plan_detail_df = plan_detail
    st.session_state.plan_summary_df = abc_xyz
    st.session_state.management_kpis_df = (
        management_kpis
    )
    st.session_state.abc_product_df = abc_product
    st.session_state.abc_summary_df = abc_summary
    st.session_state.predictability_df = (
        predictability
    )
    st.session_state.abc_xyz_df = abc_xyz
    st.session_state.transfer_df = transfer_df
    st.session_state.before_after_df = (
        before_after
    )
    st.session_state.forecast_versions_df = (
        forecast_versions
    )
    st.session_state.manual_adjustment_exists = False
    st.session_state.run_metadata = {
        "data_cutoff_date": (
            model_history["date"].max()
        ),
        "forecast_created_at": datetime.now(),
        "forecast_start": (
            future_forecast["date"].min()
        ),
        "forecast_end": (
            future_forecast["date"].max()
        ),
        "forecast_horizon": plan_horizon,
        "model_key": selected_model_key,
        "model_name": selected_model_name,
        "model_id": MODEL_CONFIGS[
            selected_model_key
        ]["model_id"],
        "data_version": (
            st.session_state.data_label
        ),
        "manual_adjustment": False,
    }


def executive_kpis(
    summary: pd.DataFrame,
) -> dict[str, Any]:
    total_demand = float(
        summary["planned_demand"].sum()
    )
    total_available = float(
        (
            summary["current_stock"]
            + summary["planned_shipment_total"]
        ).sum()
    )
    unmet = float(
        summary[
            "expected_shortage_no_action"
        ].sum()
    )
    lost_revenue = float(
        summary.get(
            "expected_lost_revenue_no_action",
            pd.Series(dtype=float),
        ).sum()
    )
    excess_value = float(
        summary.get(
            "excess_stock_value",
            pd.Series(dtype=float),
        ).sum()
    )
    service = (
        summary["fulfilled_demand"].sum()
        / max(summary["planned_demand"].sum(), 1)
        * 100
    )
    risky = int(
        summary["stockout_risk"]
        .astype(bool)
        .sum()
    )
    metrics = st.session_state.all_metrics_df
    bias = (
        float(
            metrics.loc[
                metrics["model_name"]
                .eq(
                    st.session_state[
                        "selected_model_name"
                    ]
                ),
                "bias_pct",
            ].iloc[0]
        )
        if metrics is not None
        and not metrics.empty
        else np.nan
    )
    return {
        "Gelecek dönem talebi": format_number(
            total_demand
        ),
        "Toplam kullanılabilir stok": format_number(
            total_available
        ),
        "Karşılanamayan talep": format_number(unmet),
        "Kayıp satış riski": format_try(lost_revenue),
        "Fazla stok değeri": format_try(excess_value),
        "Beklenen servis seviyesi": (
            f"%{format_number(service, 2)}"
        ),
        "Riskli mağaza–ürün": format_number(risky),
        "Bias": f"%{format_number(bias, 2)}",
    }


def global_filter_controls(
    df: pd.DataFrame,
    *,
    key_prefix: str,
) -> pd.DataFrame:
    filtered = df.copy()
    columns = st.columns(4)

    if "region" in df.columns:
        regions = sorted(
            df["region"].dropna().astype(str).unique()
        )
        selected_regions = columns[0].multiselect(
            "Bölge",
            regions,
            key=f"{key_prefix}_region",
        )
        if selected_regions:
            filtered = filtered.loc[
                filtered["region"]
                .astype(str)
                .isin(selected_regions)
            ]

    stores = sorted(
        filtered["store_id"].astype(str).unique()
    )
    selected_stores = columns[1].multiselect(
        "Mağaza",
        stores,
        key=f"{key_prefix}_store",
    )
    if selected_stores:
        filtered = filtered.loc[
            filtered["store_id"]
            .astype(str)
            .isin(selected_stores)
        ]

    if "category_1" in filtered.columns:
        categories = sorted(
            filtered["category_1"]
            .dropna()
            .astype(str)
            .unique()
        )
        selected_categories = columns[2].multiselect(
            "Kategori",
            categories,
            key=f"{key_prefix}_category",
        )
        if selected_categories:
            filtered = filtered.loc[
                filtered["category_1"]
                .astype(str)
                .isin(selected_categories)
            ]

    products = sorted(
        filtered["product_id"]
        .astype(str)
        .unique()
    )
    selected_products = columns[3].multiselect(
        "Ürün",
        products,
        key=f"{key_prefix}_product",
    )
    if selected_products:
        filtered = filtered.loc[
            filtered["product_id"]
            .astype(str)
            .isin(selected_products)
        ]

    return filtered


def page_home() -> None:
    st.markdown(
        """
        <div class="hero">
          <h2>Demand Planning AI</h2>
          <p>
            Geçmiş satış verisini zero-shot zaman serisi modelleriyle tahmin eder;
            gelecekteki stok dağıtım planını talep, servis seviyesi, kayıp satış,
            fazla stok ve transfer kararları açısından test eder.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analysis_ready():
        st.subheader("Başlangıç")
        col1, col2, col3 = st.columns(3)
        col1.info(
            "1. Veri Yükleme sayfasından geçmiş satış ve gelecek stok planını yükleyin."
        )
        col2.info(
            "2. Talep Tahmini sayfasında yalnızca zero-shot modelleri çalıştırın."
        )
        col3.info(
            "3. Yönetici, operasyon ve model performansı ekranlarını inceleyin."
        )
        return

    summary = st.session_state.plan_summary_df
    kpis = executive_kpis(summary)
    metric_columns = st.columns(8)
    for column, (name, value) in zip(
        metric_columns,
        kpis.items(),
    ):
        column.metric(name, value)

    filtered = global_filter_controls(
        summary,
        key_prefix="home",
    )

    st.markdown("### Talep–stok karşılaştırması")
    detail = st.session_state.plan_detail_df
    filtered_series = set(
        filtered["series_id"].astype(str)
    )
    detail_filtered = detail.loc[
        detail["series_id"]
        .astype(str)
        .isin(filtered_series)
    ].copy()
    timeline = (
        detail_filtered.groupby(
            "date",
            as_index=False,
        )
        .agg(
            Tahmini_Talep=(
                "predictions",
                "sum",
            ),
            Kullanilabilir_Stok=(
                "available_stock",
                "sum",
            ),
            Karsilanan_Talep=(
                "fulfilled_demand",
                "sum",
            ),
            Karsilanamayan_Talep=(
                "period_shortage",
                "sum",
            ),
        )
    )
    timeline_long = timeline.melt(
        id_vars="date",
        var_name="Gösterge",
        value_name="Adet",
    )
    st.plotly_chart(
        px.area(
            timeline_long,
            x="date",
            y="Adet",
            color="Gösterge",
            title=(
                "Plan dönemi boyunca tahmini talep ve stok yeterliliği"
            ),
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        risk_distribution = (
            filtered["plan_status"]
            .value_counts()
            .rename_axis("Durum")
            .reset_index(name="Kombinasyon")
        )
        st.plotly_chart(
            px.bar(
                risk_distribution,
                x="Durum",
                y="Kombinasyon",
                color="Durum",
                title="Risk dağılımı",
            ),
            use_container_width=True,
        )
    with col2:
        waterfall_values = {
            "Toplam talep": float(
                detail_filtered[
                    "predictions"
                ].sum()
            ),
            "Başlangıç + planlı girişle karşılanan": float(
                detail_filtered[
                    "fulfilled_demand"
                ].sum()
            ),
            "Ek tahsisle kurtarılabilir": float(
                filtered[
                    "recommended_replenishment"
                ].sum()
            ),
            "Karşılanamayan": float(
                filtered[
                    "expected_shortage_no_action"
                ].sum()
            ),
        }
        waterfall = go.Figure(
            go.Waterfall(
                x=list(waterfall_values.keys()),
                y=[
                    waterfall_values["Toplam talep"],
                    -waterfall_values[
                        "Başlangıç + planlı girişle karşılanan"
                    ],
                    -waterfall_values[
                        "Ek tahsisle kurtarılabilir"
                    ],
                    waterfall_values[
                        "Karşılanamayan"
                    ],
                ],
                measure=[
                    "absolute",
                    "relative",
                    "relative",
                    "total",
                ],
            )
        )
        waterfall.update_layout(
            title="Kayıp satış waterfall görünümü",
            yaxis_title="Adet",
        )
        st.plotly_chart(
            waterfall,
            use_container_width=True,
        )

    st.markdown("### En önemli 10 aksiyon")
    action_columns = [
        "operational_priority",
        "human_review_priority",
        "region",
        "store_id",
        "product_id",
        "abc_xyz_segment",
        "plan_status",
        "recommended_replenishment",
        "expected_lost_revenue_no_action",
        "recommended_action",
    ]
    st.dataframe(
        prepare_display_table(
            filtered,
            action_columns,
            rows=10,
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_data_upload() -> None:
    st.title("Veri Yükleme")
    st.write(
        "Geçmiş satış verisini ve geleceğe ait stok dağıtım planını ayrı dosyalar olarak yükleyin."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Örnek geçmiş veriyi indir",
            data=HISTORY_SAMPLE_PATH.read_bytes(),
            file_name="ornek_gecmis_satis_v4.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Örnek gelecek stok planını indir",
            data=PLAN_SAMPLE_PATH.read_bytes(),
            file_name="ornek_gelecek_stok_plani_v4.csv",
            mime="text/csv",
            use_container_width=True,
        )

    upload1, upload2 = st.columns(2)
    with upload1:
        history_file = st.file_uploader(
            "Geçmiş veri",
            type=["csv", "xlsx", "parquet"],
            key="history_file",
        )
    with upload2:
        plan_file = st.file_uploader(
            "Gelecek stok planı",
            type=["csv", "xlsx", "parquet"],
            key="plan_file",
        )

    if history_file is None or plan_file is None:
        st.info(
            "İki dosya da yüklendiğinde sütun eşleme ekranı açılır."
        )
        return

    try:
        raw_history = read_uploaded_table(
            history_file.getvalue(),
            history_file.name,
        )
        raw_plan = read_uploaded_table(
            plan_file.getvalue(),
            plan_file.name,
        )
    except Exception as error:
        st.error(str(error))
        return

    preview1, preview2 = st.columns(2)
    with preview1:
        st.subheader("Geçmiş veri önizleme")
        st.dataframe(
            raw_history.head(30),
            use_container_width=True,
            hide_index=True,
        )
    with preview2:
        st.subheader("Gelecek plan önizleme")
        st.dataframe(
            raw_plan.head(30),
            use_container_width=True,
            hide_index=True,
        )

    history_columns = (
        raw_history.columns.astype(str).tolist()
    )
    plan_columns = (
        raw_plan.columns.astype(str).tolist()
    )

    with st.form("mapping_form"):
        st.markdown("## Geçmiş veri — zorunlu alanlar")
        row = st.columns(5)
        history_date = select_column(
            "Tarih",
            history_columns,
            "date",
            optional=False,
            key="history_date",
        )
        history_store = select_column(
            "Mağaza ID",
            history_columns,
            "store",
            optional=False,
            key="history_store",
        )
        history_product = select_column(
            "Ürün ID",
            history_columns,
            "product",
            optional=False,
            key="history_product",
        )
        history_sales = select_column(
            "Satılan miktar",
            history_columns,
            "sales",
            optional=False,
            key="history_sales",
        )
        history_stock = select_column(
            "Dönem başı / güncel stok",
            history_columns,
            "stock",
            optional=False,
            key="history_stock",
        )

        st.markdown("## Geçmiş veri — önerilen alanlar")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            history_price = select_column(
                "Fiyat",
                history_columns,
                "price",
                optional=True,
                key="history_price",
            )
            history_promotion = select_column(
                "Promosyon",
                history_columns,
                "promotion",
                optional=True,
                key="history_promotion",
            )
            history_incoming = select_column(
                "Gelen stok",
                history_columns,
                "incoming_stock",
                optional=True,
                key="history_incoming",
            )
        with col2:
            history_order = select_column(
                "Sipariş miktarı",
                history_columns,
                "order_quantity",
                optional=True,
                key="history_order",
            )
            history_lead = select_column(
                "Tedarik süresi",
                history_columns,
                "lead_time",
                optional=True,
                key="history_lead",
            )
            history_stockout = select_column(
                "Stokta yok bilgisi",
                history_columns,
                "stockout",
                optional=True,
                key="history_stockout",
            )
        with col3:
            history_category_1 = select_column(
                "Ürün kategorisi",
                history_columns,
                "category_1",
                optional=True,
                key="history_category_1",
            )
            history_category_2 = select_column(
                "Alt kategori",
                history_columns,
                "category_2",
                optional=True,
                key="history_category_2",
            )
            history_brand = select_column(
                "Marka",
                history_columns,
                "brand",
                optional=True,
                key="history_brand",
            )
        with col4:
            history_region = select_column(
                "Bölge",
                history_columns,
                "region",
                optional=True,
                key="history_region",
            )
            history_cost = select_column(
                "Birim maliyet",
                history_columns,
                "unit_cost",
                optional=True,
                key="history_cost",
            )
            history_profit = select_column(
                "Birim kâr / marj",
                history_columns,
                "profit",
                optional=True,
                key="history_profit",
            )

        with st.expander("İade, iptal ve ürün durumu alanları"):
            col1, col2, col3, col4 = st.columns(4)
            history_returns = select_column(
                "İade",
                history_columns,
                "returns",
                optional=True,
                key="history_returns",
            )
            history_cancellations = select_column(
                "İptal",
                history_columns,
                "cancellations",
                optional=True,
                key="history_cancellations",
            )
            history_new_product = select_column(
                "Yeni ürün",
                history_columns,
                "new_product",
                optional=True,
                key="history_new_product",
            )
            history_strategic = select_column(
                "Stratejik ürün",
                history_columns,
                "strategic_product",
                optional=True,
                key="history_strategic",
            )

        st.markdown("## Gelecek stok planı")
        col1, col2, col3, col4 = st.columns(4)
        plan_date = select_column(
            "Tarih",
            plan_columns,
            "date",
            optional=False,
            key="plan_date",
        )
        plan_store = select_column(
            "Mağaza ID",
            plan_columns,
            "store",
            optional=False,
            key="plan_store",
        )
        plan_product = select_column(
            "Ürün ID",
            plan_columns,
            "product",
            optional=False,
            key="plan_product",
        )
        plan_shipment = select_column(
            "Planlanan gönderim",
            plan_columns,
            "planned_shipment",
            optional=False,
            key="plan_shipment",
        )

        col1, col2, col3, col4 = st.columns(4)
        plan_stock = select_column(
            "Mevcut stok",
            plan_columns,
            "starting_stock",
            optional=False,
            key="plan_stock",
        )
        plan_arrival = select_column(
            "Beklenen giriş tarihi",
            plan_columns,
            "arrival_date",
            optional=True,
            key="plan_arrival",
        )
        plan_warehouse = select_column(
            "Dağıtılabilir depo stoğu",
            plan_columns,
            "warehouse_stock",
            optional=True,
            key="plan_warehouse",
        )
        plan_capacity = select_column(
            "Mağaza kapasitesi",
            plan_columns,
            "capacity",
            optional=True,
            key="plan_capacity",
        )

        col1, col2 = st.columns(2)
        plan_price = select_column(
            "Plan dönemi fiyatı",
            plan_columns,
            "price",
            optional=True,
            key="plan_price",
        )
        plan_region = select_column(
            "Plan bölgesi",
            plan_columns,
            "region",
            optional=True,
            key="plan_region",
        )

        with st.expander("Pipeline ayarları"):
            col1, col2, col3, col4 = st.columns(4)
            frequency = col1.selectbox(
                "Frekans",
                ["daily", "hourly", "monthly"],
                format_func={
                    "daily": "Günlük",
                    "hourly": "Saatlik",
                    "monthly": "Aylık",
                }.get,
            )
            duplicate_policy = col2.selectbox(
                "Tekrarlı kayıt",
                ["aggregate", "error"],
                format_func={
                    "aggregate": "Birleştir",
                    "error": "Hata ver",
                }.get,
            )
            date_gap_policy = col3.selectbox(
                "Eksik tarih",
                ["warn", "error", "ignore"],
                format_func={
                    "warn": "Uyar",
                    "error": "Hata ver",
                    "ignore": "Yoksay",
                }.get,
            )
            stock_timing = col4.selectbox(
                "Stok zamanı",
                [
                    "end_of_period",
                    "start_of_period",
                ],
                format_func={
                    "end_of_period": "Dönem sonu",
                    "start_of_period": "Dönem başı",
                }.get,
            )
            dayfirst = st.checkbox(
                "Tarih biçimi gün/ay/yıl",
                value=False,
            )

        submitted = st.form_submit_button(
            "Verileri doğrula ve hazırla",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    try:
        mapping = ColumnMapping(
            date=str(history_date),
            store=str(history_store),
            product=str(history_product),
            sales=str(history_sales),
            stock=str(history_stock),
            price=history_price,
            category_1=history_category_1,
            category_2=history_category_2,
            category_3=history_brand,
            stockout_flag=history_stockout,
        )
        pipeline = DemandDataPipeline(
            mapping=mapping,
            frequency=frequency,
            dayfirst=dayfirst,
            duplicate_policy=duplicate_policy,
            date_gap_policy=date_gap_policy,
            negative_value_policy="clip",
            use_sales_equals_stock_rule=False,
            stockout_tolerance=0.0,
            stockout_stock_threshold=0.0,
            combine_provided_flag_with_inferred=True,
            stock_timing=stock_timing,
            imputation_window=28,
            min_history=7,
            imputation_statistic="median",
        )
        prepared = pipeline.prepare(raw_history)

        extra_mapping = HistoryExtraMapping(
            promotion=history_promotion,
            incoming_stock=history_incoming,
            order_quantity=history_order,
            lead_time=history_lead,
            region=history_region,
            brand=history_brand,
            unit_cost=history_cost,
            profit=history_profit,
            returns=history_returns,
            cancellations=history_cancellations,
            new_product=history_new_product,
            strategic_product=history_strategic,
        )
        prepared = enrich_prepared_history(
            prepared,
            raw_history,
            raw_date_column=str(history_date),
            raw_store_column=str(history_store),
            raw_product_column=str(
                history_product
            ),
            extra_mapping=extra_mapping,
        )
        adjusted = pipeline.impute_stockouts(
            prepared
        )

        plan_mapping = DistributionPlanMapping(
            date=str(plan_date),
            store=str(plan_store),
            product=str(plan_product),
            starting_stock=str(plan_stock),
            planned_shipment=str(plan_shipment),
            expected_arrival_date=plan_arrival,
            warehouse_stock=plan_warehouse,
            store_capacity=plan_capacity,
            price=plan_price,
            region=plan_region,
        )
        future_plan = prepare_distribution_plan(
            raw_plan,
            mapping=plan_mapping,
            prepared_history_df=prepared,
            pandas_frequency=pipeline.pandas_freq,
        )

        quality_summary, quality_issues = (
            build_data_quality_report(
                raw_history,
                prepared,
                date_column=str(history_date),
                sales_column=str(history_sales),
                stock_column=str(history_stock),
                stockout_column=history_stockout,
                new_product_column=history_new_product,
            )
        )
        error_reason = build_error_reason_summary(
            prepared,
            quality_issues,
        )

        st.session_state.history_raw_df = (
            raw_history
        )
        st.session_state.plan_raw_df = raw_plan
        st.session_state.pipeline = pipeline
        st.session_state.prepared_history_df = (
            prepared
        )
        st.session_state.analysis_history_df = (
            adjusted
        )
        st.session_state.future_plan_df = (
            future_plan
        )
        st.session_state.quality_summary_df = (
            quality_summary
        )
        st.session_state.quality_issues_df = (
            quality_issues
        )
        st.session_state.error_reason_df = (
            error_reason
        )
        st.session_state.mapping_metadata = {
            "history_date": history_date,
            "history_store": history_store,
            "history_product": history_product,
            "history_sales": history_sales,
            "history_stock": history_stock,
            "frequency": frequency,
            "stock_timing": stock_timing,
        }
        st.session_state.data_label = (
            f"{history_file.name} + {plan_file.name}"
        )
        reset_analysis_outputs()
        st.success(
            "Veriler hazırlandı. Şimdi Veri Kalitesi ve Talep Tahmini sayfalarına geçebilirsiniz."
        )
    except (
        DataValidationError,
        ValueError,
    ) as error:
        st.error(str(error))
    except Exception as error:
        st.error(
            f"{type(error).__name__}: {error}"
        )
        with st.expander("Teknik hata"):
            st.code(traceback.format_exc())


def page_data_quality() -> None:
    st.title("Veri Kalitesi")
    if not data_ready():
        st.info(
            "Önce Veri Yükleme sayfasında iki dosyayı hazırlayın."
        )
        return

    summary = st.session_state.quality_summary_df
    issues = st.session_state.quality_issues_df
    error_reasons = st.session_state.error_reason_df

    st.subheader("Veri Kalitesi Raporu")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    prepared = st.session_state.prepared_history_df
    col1.metric(
        "Tarih Aralığı",
        f"{prepared['date'].min():%d.%m.%Y}",
        f"{prepared['date'].max():%d.%m.%Y}",
    )
    col2.metric(
        "Mağaza–Ürün",
        format_number(
            prepared["series_id"].nunique()
        ),
    )
    col3.metric(
        "Stokta Yok Dönemi",
        format_number(
            prepared["is_stockout"]
            .astype(bool)
            .sum()
        ),
    )
    col4.metric(
        "Düzeltilen Kayıp Talep",
        format_number(
            (
                st.session_state[
                    "analysis_history_df"
                ]["demand_adjusted"]
                - st.session_state[
                    "analysis_history_df"
                ]["sales"]
            )
            .clip(lower=0)
            .sum()
        ),
    )

    st.subheader(
        "Satış ile gerçek talebi ayırma"
    )
    st.warning(
        "Stokta olmayan dönemlerde satışın sıfır olması talebin sıfır olduğu anlamına gelmez. "
        "Bu dönemler model bağlamında düzeltilmiş talep ile ele alınır; backtest metriğinde stokout satırları ayrıca izlenir."
    )

    if issues is not None and not issues.empty:
        st.subheader("Sorunlu kayıtlar")
        st.dataframe(
            issues.head(1000),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success(
            "Satır seviyesinde kritik tarih veya satış sorunu bulunmadı."
        )

    st.subheader("Tahmin hatasının olası nedenleri")
    st.dataframe(
        error_reasons,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Pipeline uyarıları ve istatistikleri"):
        report = st.session_state.pipeline.report
        for warning in report.get(
            "warnings",
            [],
        ):
            st.warning(str(warning))
        st.json(report.get("stats", {}))


def page_demand_forecast() -> None:
    st.title("Talep Tahmini")
    if not data_ready():
        st.info(
            "Önce Veri Yükleme sayfasında verileri hazırlayın."
        )
        return

    pipeline = st.session_state.pipeline
    defaults = FREQUENCY_DEFAULTS[
        pipeline.frequency
    ]
    plan_horizon = int(
        st.session_state.future_plan_df[
            "date"
        ].nunique()
    )
    plan_series_count = int(
        st.session_state.future_plan_df[
            "series_id"
        ].nunique()
    )

    st.info(
        f"Gerçek gelecek tahmin ufku yüklenen stok planından alınır: "
        f"{plan_horizon} dönem ve {plan_series_count} mağaza–ürün."
    )

    available_models = {
        "chronos_bolt": "Chronos Bolt Small",
        "chronos_2": "Chronos 2",
    }
    if importlib.util.find_spec("timesfm") is not None:
        available_models[
            "timesfm_2_5"
        ] = "TimesFM 2.5"

    selected_models = st.multiselect(
        "Zero-shot modeller",
        options=list(available_models),
        default=["chronos_bolt"],
        format_func=available_models.get,
        help=(
            "Final tahmin modeli yalnızca bu zero-shot modeller arasından seçilir. "
            "Naïve yöntemler benchmark olarak kullanılır."
        ),
    )

    col1, col2, col3, col4 = st.columns(4)
    backtest_horizon = int(
        col1.number_input(
            "Backtest ufku",
            min_value=1,
            max_value=90,
            value=min(
                defaults["backtest_horizon"],
                plan_horizon,
            ),
        )
    )
    min_context = int(
        col2.number_input(
            "Minimum geçmiş",
            min_value=2,
            max_value=5000,
            value=defaults["min_context"],
        )
    )
    safety_periods = float(
        col3.number_input(
            "Güvenlik stoğu dönemi",
            min_value=0.0,
            max_value=30.0,
            value=1.0,
            step=0.5,
        )
    )
    service_level = float(
        col4.slider(
            "Minimum servis seviyesi",
            min_value=0.80,
            max_value=1.00,
            value=0.95,
            step=0.01,
        )
    )

    if st.button(
        "Zero-shot tahmin ve plan analizini çalıştır",
        type="primary",
        disabled=not selected_models,
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Benchmarklar, zero-shot modeller, gelecek tahmini ve stok planı analiz ediliyor..."
            ):
                run_zero_shot_pipeline(
                    model_keys=selected_models,
                    backtest_horizon=(
                        backtest_horizon
                    ),
                    min_context=min_context,
                    safety_periods=safety_periods,
                    minimum_service_level=(
                        service_level
                    ),
                )
            st.success(
                "Tahmin ve stok planı analizi tamamlandı."
            )
        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )
            with st.expander("Teknik hata"):
                st.code(traceback.format_exc())

    if not analysis_ready():
        return

    history = st.session_state.analysis_history_df
    future = st.session_state.future_forecast_df
    detail = st.session_state.plan_detail_df

    filtered_plan = global_filter_controls(
        st.session_state.plan_summary_df,
        key_prefix="forecast_filters",
    )
    series_options = (
        filtered_plan[
            ["series_id", "store_id", "product_id"]
        ]
        .drop_duplicates()
        .assign(
            label=lambda table: (
                table["store_id"].astype(str)
                + " / "
                + table["product_id"].astype(str)
            )
        )
    )
    selected_label = st.selectbox(
        "Mağaza–ürün",
        series_options["label"].tolist(),
    )
    selected_series = str(
        series_options.loc[
            series_options["label"].eq(
                selected_label
            ),
            "series_id",
        ].iloc[0]
    )

    history_series = (
        history.loc[
            history["series_id"]
            .astype(str)
            .eq(selected_series)
        ]
        .sort_values("date")
        .tail(120)
    )
    future_series = (
        future.loc[
            future["series_id"]
            .astype(str)
            .eq(selected_series)
        ]
        .sort_values("date")
    )
    detail_series = (
        detail.loc[
            detail["series_id"]
            .astype(str)
            .eq(selected_series)
        ]
        .sort_values("date")
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history_series["date"],
            y=history_series["sales"],
            name="Gerçek satış",
            mode="lines",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history_series["date"],
            y=history_series[
                "demand_adjusted"
            ],
            name="Düzeltilmiş talep",
            mode="lines",
            line={"dash": "dot"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_series["date"],
            y=future_series["forecast_upper"],
            name="Üst güven sınırı",
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_series["date"],
            y=future_series["forecast_lower"],
            name="Ampirik güven aralığı",
            mode="lines",
            fill="tonexty",
            line={"width": 0},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_series["date"],
            y=future_series["predictions"],
            name=st.session_state.selected_model_name,
            mode="lines+markers",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future_series["date"],
            y=future_series["naive_forecast"],
            name="Geçen dönem benchmark",
            mode="lines",
            line={"dash": "dash"},
        )
    )

    stockout_points = history_series.loc[
        history_series["is_stockout"]
        .astype(bool)
    ]
    if not stockout_points.empty:
        figure.add_trace(
            go.Scatter(
                x=stockout_points["date"],
                y=stockout_points["sales"],
                name="Stokta yok",
                mode="markers",
                marker={
                    "symbol": "x",
                    "size": 9,
                },
            )
        )

    if "promotion" in history_series.columns:
        promotion_points = history_series.loc[
            history_series["promotion"]
            .astype("boolean")
            .fillna(False)
        ]
        if not promotion_points.empty:
            figure.add_trace(
                go.Scatter(
                    x=promotion_points["date"],
                    y=promotion_points["sales"],
                    name="Promosyon",
                    mode="markers",
                    marker={
                        "symbol": "diamond",
                        "size": 8,
                    },
                )
            )

    figure.update_layout(
        title="Geçmiş satış, benchmark ve gelecek zero-shot talep tahmini",
        xaxis_title="Tarih",
        yaxis_title="Adet",
        hovermode="x unified",
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    stock_figure = go.Figure()
    stock_figure.add_trace(
        go.Scatter(
            x=detail_series["date"],
            y=detail_series[
                "projected_ending_stock"
            ],
            name="Mevcut planla stok",
            mode="lines+markers",
            fill="tozeroy",
        )
    )
    stock_figure.add_trace(
        go.Scatter(
            x=detail_series["date"],
            y=detail_series[
                "recommended_ending_stock"
            ],
            name="Önerilen planla stok",
            mode="lines+markers",
        )
    )
    stock_figure.add_trace(
        go.Scatter(
            x=detail_series["date"],
            y=detail_series["safety_stock"],
            name="Güvenlik stoğu",
            mode="lines",
            line={"dash": "dash"},
        )
    )
    stock_figure.update_layout(
        title="Stok projeksiyonu",
        xaxis_title="Tarih",
        yaxis_title="Stok adedi",
        hovermode="x unified",
    )
    st.plotly_chart(
        stock_figure,
        use_container_width=True,
    )

    st.dataframe(
        prepare_display_table(
            detail_series,
            [
                "date",
                "opening_stock",
                "effective_incoming_stock",
                "predictions",
                "available_stock",
                "projected_ending_stock",
                "period_shortage",
                "recommended_extra_shipment",
                "recommended_ending_stock",
                "stockout_risk",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_forecast_performance() -> None:
    st.title("Tahmin Performansı ve Model Kalitesi")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında modelleri çalıştırın."
        )
        return

    metrics = st.session_state.all_metrics_df
    evaluations = (
        st.session_state.all_evaluations_df
    )

    selected_row = metrics.loc[
        metrics["model_name"].eq(
            st.session_state.selected_model_name
        )
    ].iloc[0]
    benchmark_wmape = float(
        st.session_state.baseline_metrics_df[
            "wmape_pct"
        ].min()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "WMAPE",
        f"%{format_number(selected_row['wmape_pct'], 2)}",
    )
    col2.metric(
        "Bias",
        f"%{format_number(selected_row['bias_pct'], 2)}",
    )
    col3.metric(
        "MAE",
        format_number(selected_row["mae"], 2),
    )
    col4.metric(
        "Benchmark İyileşmesi",
        f"%{format_number((benchmark_wmape - selected_row['wmape_pct']) / benchmark_wmape * 100, 2)}",
    )

    metric_columns = [
        "model_name",
        "model_type",
        "wmape_pct",
        "bias_pct",
        "mae",
        "rmse",
        "benchmark_improvement_pct",
        "forecast_coverage_pct",
        "runtime_seconds",
    ]
    st.dataframe(
        prepare_display_table(
            metrics,
            metric_columns,
        ),
        use_container_width=True,
        hide_index=True,
    )

    horizon_performance = (
        build_horizon_performance(
            evaluations
        )
    )
    selected_models = st.multiselect(
        "Ufuk grafiğinde gösterilecek modeller",
        options=sorted(
            horizon_performance[
                "model_name"
            ].unique()
        ),
        default=[
            st.session_state.selected_model_name,
            "Sezonsal Naïve",
        ],
    )
    horizon_filtered = (
        horizon_performance.loc[
            horizon_performance[
                "model_name"
            ].isin(selected_models)
        ]
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.line(
                horizon_filtered,
                x="horizon_step",
                y="wmape_pct",
                color="model_name",
                markers=True,
                title="Tahmin ufkuna göre WMAPE",
                labels={
                    "horizon_step": "Ufuk",
                    "wmape_pct": "WMAPE (%)",
                    "model_name": "Model",
                },
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.line(
                horizon_filtered,
                x="horizon_step",
                y="bias_pct",
                color="model_name",
                markers=True,
                title="Tahmin ufkuna göre Bias",
                labels={
                    "horizon_step": "Ufuk",
                    "bias_pct": "Bias (%)",
                    "model_name": "Model",
                },
            ),
            use_container_width=True,
        )

    st.subheader("Hata ısı haritası")
    selected_model_eval = evaluations.loc[
        evaluations["model_name"].eq(
            st.session_state.selected_model_name
        )
    ]
    row_dimension = (
        "region"
        if "region"
        in st.session_state.prepared_history_df.columns
        else "store_id"
    )
    column_dimension = (
        "category_1"
        if "category_1"
        in st.session_state.prepared_history_df.columns
        else "week"
    )
    heatmap_metric = st.radio(
        "Isı haritası metriği",
        ["WMAPE", "Bias"],
        horizontal=True,
    )
    matrix = build_error_heatmap_data(
        selected_model_eval,
        st.session_state.prepared_history_df,
        row_dimension=row_dimension,
        column_dimension=column_dimension,
        metric=(
            "bias"
            if heatmap_metric == "Bias"
            else "wmape"
        ),
    )
    st.plotly_chart(
        px.imshow(
            matrix,
            aspect="auto",
            labels={
                "x": DISPLAY_NAMES.get(
                    column_dimension,
                    column_dimension,
                ),
                "y": DISPLAY_NAMES.get(
                    row_dimension,
                    row_dimension,
                ),
                "color": heatmap_metric,
            },
            title=f"{heatmap_metric} hata ısı haritası",
        ),
        use_container_width=True,
    )

    st.subheader("Hata nedenleri")
    st.dataframe(
        st.session_state.error_reason_df,
        use_container_width=True,
        hide_index=True,
    )

    if (
        st.session_state.model_errors_df
        is not None
        and not st.session_state.model_errors_df.empty
    ):
        with st.expander("Çalışmayan modeller"):
            st.dataframe(
                st.session_state.model_errors_df,
                use_container_width=True,
                hide_index=True,
            )


def page_stock_risks() -> None:
    st.title("Stok Riskleri")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    summary = global_filter_controls(
        st.session_state.plan_summary_df,
        key_prefix="stock_risk",
    )
    detail = st.session_state.plan_detail_df
    detail = detail.loc[
        detail["series_id"]
        .astype(str)
        .isin(
            summary["series_id"].astype(str)
        )
    ]

    total_need = float(
        summary["recommended_replenishment"].sum()
    )
    available_warehouse = float(
        summary.get(
            "warehouse_stock",
            pd.Series(dtype=float),
        )
        .groupby(
            summary["product_id"]
            if "product_id" in summary
            else pd.Series(dtype=str)
        )
        .max()
        .sum()
        if "warehouse_stock" in summary.columns
        else 0
    )
    shortage = float(
        summary["expected_shortage_no_action"].sum()
    )
    excess = float(
        summary["excess_stock_units"].sum()
    )
    service = (
        summary["fulfilled_demand"].sum()
        / max(summary["planned_demand"].sum(), 1)
        * 100
    )
    recoverable_revenue = float(
        summary.get(
            "expected_lost_revenue_no_action",
            pd.Series(dtype=float),
        ).sum()
    )

    cards = st.columns(6)
    cards[0].metric(
        "Toplam ihtiyaç",
        format_number(total_need),
    )
    cards[1].metric(
        "Dağıtılabilir depo stoğu",
        format_number(available_warehouse),
    )
    cards[2].metric(
        "Stok açığı",
        format_number(shortage),
    )
    cards[3].metric(
        "Fazla stok",
        format_number(excess),
    )
    cards[4].metric(
        "Servis seviyesi",
        f"%{format_number(service, 2)}",
    )
    cards[5].metric(
        "Kurtarılabilir satış",
        format_try(recoverable_revenue),
    )

    timeline = (
        detail.groupby("date", as_index=False)
        .agg(
            Stok_Seviyesi=(
                "projected_ending_stock",
                "sum",
            ),
            Guvenlik_Stogu=(
                "safety_stock",
                "sum",
            ),
            Tahmini_Talep=(
                "predictions",
                "sum",
            ),
            Planlanan_Gonderim=(
                "effective_incoming_stock",
                "sum",
            ),
        )
    )
    timeline_long = timeline.melt(
        id_vars="date",
        var_name="Gösterge",
        value_name="Adet",
    )
    st.plotly_chart(
        px.line(
            timeline_long,
            x="date",
            y="Adet",
            color="Gösterge",
            markers=True,
            title="Toplam stok projeksiyonu",
        ),
        use_container_width=True,
    )

    risk_by_dimension = (
        summary.groupby(
            "region"
            if "region" in summary.columns
            else "store_id",
            as_index=False,
        )["plan_status"]
        .value_counts()
    )
    dimension = (
        "region"
        if "region" in summary.columns
        else "store_id"
    )
    st.plotly_chart(
        px.bar(
            risk_by_dimension,
            x=dimension,
            y="count",
            color="plan_status",
            barmode="stack",
            title="Bölge / mağaza bazında stok riskleri",
            labels={
                dimension: DISPLAY_NAMES.get(
                    dimension,
                    dimension,
                ),
                "count": "Mağaza–ürün",
                "plan_status": "Durum",
            },
        ),
        use_container_width=True,
    )

    risk_columns = [
        "region",
        "store_id",
        "product_id",
        "current_stock",
        "planned_demand",
        "planned_shipment_total",
        "expected_shortage_no_action",
        "expected_stockout_date",
        "recommended_replenishment",
        "plan_status",
    ]
    st.dataframe(
        prepare_display_table(
            summary,
            risk_columns,
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_distribution_recommendations() -> None:
    st.title("Stok Dağıtım Önerileri")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    transfer = st.session_state.transfer_df
    before_after = st.session_state.before_after_df
    summary = st.session_state.plan_summary_df

    st.subheader("Mevcut plan / önerilen plan")
    st.dataframe(
        before_after,
        use_container_width=True,
        hide_index=True,
    )

    comparison_long = before_after.melt(
        id_vars="KPI",
        value_vars=[
            "Mevcut plan",
            "Önerilen plan",
        ],
        var_name="Plan",
        value_name="Değer",
    )
    st.plotly_chart(
        px.bar(
            comparison_long,
            x="KPI",
            y="Değer",
            color="Plan",
            barmode="group",
            title="Önce / sonra karşılaştırması",
        ),
        use_container_width=True,
    )

    st.subheader("Operasyon aksiyon dosyası")
    if transfer is None or transfer.empty:
        st.info(
            "Mevcut depo ve mağaza fazlasıyla oluşturulabilecek transfer önerisi bulunmadı."
        )
    else:
        display_transfer = transfer.rename(
            columns={
                "source_type": "Kaynak Tipi",
                "source_location": "Kaynak Mağaza / Depo",
                "target_store": "Hedef Mağaza",
                "product_id": "Ürün",
                "quantity": "Gönderilecek Miktar",
                "deadline": "Son Tarih",
                "priority": "Öncelik",
                "expected_saved_sales": "Korunan Satış",
                "expected_saved_revenue": "Korunan Satış Değeri (TL)",
            }
        )
        st.dataframe(
            display_transfer,
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            px.bar(
                transfer.nlargest(
                    20,
                    "expected_saved_revenue",
                ),
                x="expected_saved_revenue",
                y="target_store",
                color="source_type",
                orientation="h",
                title="En yüksek satış koruma etkisine sahip transferler",
                labels={
                    "expected_saved_revenue": (
                        "Korunan satış değeri (TL)"
                    ),
                    "target_store": "Hedef mağaza",
                    "source_type": "Kaynak",
                },
            ),
            use_container_width=True,
        )

    st.subheader("En önemli kararlar")
    st.dataframe(
        prepare_display_table(
            summary,
            [
                "operational_priority",
                "human_review_priority",
                "region",
                "store_id",
                "product_id",
                "abc_xyz_segment",
                "plan_status",
                "recommended_replenishment",
                "expected_lost_revenue_no_action",
                "recommended_action",
            ],
            rows=50,
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_store_product_detail() -> None:
    st.title("Mağaza / Ürün Detayı")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    mode = st.radio(
        "Detay türü",
        ["Mağaza Detayı", "Ürün Detayı"],
        horizontal=True,
    )
    summary = st.session_state.plan_summary_df
    detail = st.session_state.plan_detail_df
    evaluation = (
        st.session_state.zero_shot_evaluations_df
    )

    if mode == "Mağaza Detayı":
        store = st.selectbox(
            "Mağaza",
            sorted(summary["store_id"].astype(str).unique()),
        )
        selected = summary.loc[
            summary["store_id"].astype(str).eq(store)
        ]
        selected_detail = detail.loc[
            detail["store_id"].astype(str).eq(store)
        ]

        cards = st.columns(5)
        cards[0].metric(
            "Toplam talep",
            format_number(
                selected["planned_demand"].sum()
            ),
        )
        cards[1].metric(
            "Riskli ürün",
            format_number(
                selected["stockout_risk"].sum()
            ),
        )
        cards[2].metric(
            "Fazla stoklu ürün",
            format_number(
                selected["plan_status"]
                .eq("Fazla stok")
                .sum()
            ),
        )
        cards[3].metric(
            "Ek gönderim",
            format_number(
                selected[
                    "recommended_replenishment"
                ].sum()
            ),
        )
        cards[4].metric(
            "Servis seviyesi",
            f"%{format_number(selected['fulfilled_demand'].sum() / max(selected['planned_demand'].sum(), 1) * 100, 2)}",
        )

        if "category_1" in selected.columns:
            category = (
                selected.groupby(
                    "category_1",
                    as_index=False,
                )
                .agg(
                    Tahmini_Talep=(
                        "planned_demand",
                        "sum",
                    ),
                    Ek_Gonderim=(
                        "recommended_replenishment",
                        "sum",
                    ),
                )
            )
            st.plotly_chart(
                px.bar(
                    category,
                    x="category_1",
                    y=[
                        "Tahmini_Talep",
                        "Ek_Gonderim",
                    ],
                    barmode="group",
                    title="Kategori bazında talep ve ek gönderim",
                    labels={
                        "category_1": "Kategori",
                        "value": "Adet",
                        "variable": "Gösterge",
                    },
                ),
                use_container_width=True,
            )

        st.dataframe(
            prepare_display_table(
                selected,
                [
                    "product_id",
                    "category_1",
                    "planned_demand",
                    "current_stock",
                    "planned_shipment_total",
                    "recommended_replenishment",
                    "plan_status",
                    "abc_xyz_segment",
                    "expected_stockout_date",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:
        product = st.selectbox(
            "Ürün",
            sorted(
                summary["product_id"]
                .astype(str)
                .unique()
            ),
        )
        selected = summary.loc[
            summary["product_id"]
            .astype(str)
            .eq(product)
        ]
        selected_detail = detail.loc[
            detail["product_id"]
            .astype(str)
            .eq(product)
        ]

        cards = st.columns(5)
        cards[0].metric(
            "Toplam talep",
            format_number(
                selected["planned_demand"].sum()
            ),
        )
        cards[1].metric(
            "Toplam stok",
            format_number(
                selected["current_stock"].sum()
            ),
        )
        cards[2].metric(
            "Riskli mağaza",
            format_number(
                selected["stockout_risk"].sum()
            ),
        )
        cards[3].metric(
            "Fazla stok",
            format_number(
                selected["excess_stock_units"].sum()
            ),
        )
        cards[4].metric(
            "Ek gönderim",
            format_number(
                selected[
                    "recommended_replenishment"
                ].sum()
            ),
        )

        st.plotly_chart(
            px.scatter(
                selected,
                x="planned_demand",
                y="current_stock",
                size="recommended_replenishment",
                color="plan_status",
                hover_name="store_id",
                title="Mağaza talep–stok matrisi",
                labels={
                    "planned_demand": "Tahmini talep",
                    "current_stock": "Mevcut stok",
                    "plan_status": "Durum",
                    "recommended_replenishment": (
                        "Ek gönderim"
                    ),
                },
            ),
            use_container_width=True,
        )

        if "region" in selected.columns:
            region = (
                selected.groupby(
                    "region",
                    as_index=False,
                )
                .agg(
                    Tahmini_Talep=(
                        "planned_demand",
                        "sum",
                    ),
                    Mevcut_Stok=(
                        "current_stock",
                        "sum",
                    ),
                    Ek_Gonderim=(
                        "recommended_replenishment",
                        "sum",
                    ),
                )
            )
            st.plotly_chart(
                px.bar(
                    region,
                    x="region",
                    y=[
                        "Tahmini_Talep",
                        "Mevcut_Stok",
                        "Ek_Gonderim",
                    ],
                    barmode="group",
                    title="Bölge bazında ürün görünümü",
                    labels={
                        "region": "Bölge",
                        "value": "Adet",
                        "variable": "Gösterge",
                    },
                ),
                use_container_width=True,
            )

        st.dataframe(
            prepare_display_table(
                selected,
                [
                    "region",
                    "store_id",
                    "planned_demand",
                    "current_stock",
                    "planned_shipment_total",
                    "recommended_replenishment",
                    "plan_status",
                    "expected_stockout_date",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )


def page_abc_xyz() -> None:
    st.title("ABC–XYZ Önceliklendirme")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    abc_xyz = st.session_state.abc_xyz_df
    if abc_xyz is None:
        st.warning(
            "ABC–XYZ tablosu oluşturulamadı."
        )
        return

    segment_summary = (
        abc_xyz.groupby(
            ["abc_class", "xyz_class"],
            as_index=False,
        )
        .agg(
            Kombinasyon=(
                "series_id",
                "nunique",
            ),
            Tahmini_Talep=(
                "planned_demand",
                "sum",
            ),
            Ek_Gonderim=(
                "recommended_replenishment",
                "sum",
            ),
            Ciro_Riski=(
                "expected_lost_revenue_no_action",
                "sum",
            )
            if "expected_lost_revenue_no_action"
            in abc_xyz.columns
            else (
                "expected_shortage_no_action",
                "sum",
            ),
        )
    )
    pivot = segment_summary.pivot_table(
        index="abc_class",
        columns="xyz_class",
        values="Kombinasyon",
        fill_value=0,
    )
    st.plotly_chart(
        px.imshow(
            pivot,
            aspect="auto",
            text_auto=True,
            labels={
                "x": "XYZ — öngörülebilirlik",
                "y": "ABC — ekonomik önem",
                "color": "Mağaza–ürün",
            },
            title="ABC–XYZ segment matrisi",
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.bar(
                segment_summary,
                x="abc_class",
                y="Kombinasyon",
                color="xyz_class",
                barmode="stack",
                title="Segment dağılımı",
                labels={
                    "abc_class": "ABC",
                    "xyz_class": "XYZ",
                },
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.scatter(
                abc_xyz,
                x="predictability_score",
                y=(
                    "expected_lost_revenue_no_action"
                    if "expected_lost_revenue_no_action"
                    in abc_xyz.columns
                    else "expected_shortage_no_action"
                ),
                color="abc_xyz_segment",
                size="recommended_replenishment",
                hover_name="series_id",
                title="İş değeri ve öngörülebilirlik önceliği",
                labels={
                    "predictability_score": (
                        "Öngörülemezlik risk puanı"
                    ),
                    "abc_xyz_segment": "Segment",
                },
            ),
            use_container_width=True,
        )

    st.dataframe(
        prepare_display_table(
            abc_xyz,
            [
                "human_review_priority",
                "abc_xyz_segment",
                "region",
                "store_id",
                "product_id",
                "wmape_pct",
                "bias_pct",
                "zero_sales_ratio",
                "intermittency",
                "stockout_rate",
                "recommended_replenishment",
                "segment_action",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_scenario() -> None:
    st.title("Senaryo Analizi")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    demand_change = col1.slider(
        "Talep değişimi (%)",
        -30,
        50,
        10,
        1,
    )
    shipment_change = col2.slider(
        "Gelen stok değişimi (%)",
        -50,
        50,
        0,
        1,
    )
    delay = col3.number_input(
        "Tedarik gecikmesi (dönem)",
        min_value=0,
        max_value=30,
        value=0,
    )
    minimum_service = col4.slider(
        "Minimum servis seviyesi",
        0.80,
        1.00,
        0.95,
        0.01,
    )
    safety = st.slider(
        "Güvenlik stoğu dönemi",
        0.0,
        6.0,
        1.0,
        0.5,
    )

    if st.button(
        "Senaryoları karşılaştır",
        type="primary",
    ):
        scenarios = [
            {
                "name": "Mevcut plan",
                "demand_multiplier": 1.0,
                "shipment_multiplier": 1.0,
                "arrival_delay_periods": 0,
                "minimum_service_level": 0.95,
                "safety_periods": 1.0,
            },
            {
                "name": "Dengeli dağıtım",
                "demand_multiplier": 1.0,
                "shipment_multiplier": 1.0,
                "arrival_delay_periods": 0,
                "minimum_service_level": 0.95,
                "safety_periods": 0.5,
            },
            {
                "name": "Yüksek servis",
                "demand_multiplier": 1.0,
                "shipment_multiplier": 1.0,
                "arrival_delay_periods": 0,
                "minimum_service_level": 0.98,
                "safety_periods": 2.0,
            },
            {
                "name": "Özel senaryo",
                "demand_multiplier": (
                    1 + demand_change / 100
                ),
                "shipment_multiplier": (
                    1 + shipment_change / 100
                ),
                "arrival_delay_periods": int(delay),
                "minimum_service_level": (
                    minimum_service
                ),
                "safety_periods": safety,
            },
        ]
        try:
            comparison, detail_lookup, summary_lookup = (
                compare_scenarios(
                    st.session_state.aligned_plan_df,
                    historical_loss_summary_df=(
                        st.session_state[
                            "historical_loss_summary_df"
                        ]
                    ),
                    pandas_frequency=(
                        st.session_state.pipeline.pandas_freq
                    ),
                    scenarios=scenarios,
                )
            )
            st.session_state.scenario_comparison_df = (
                comparison
            )
            st.session_state.scenario_detail_lookup = (
                detail_lookup
            )
            st.session_state.scenario_summary_lookup = (
                summary_lookup
            )
        except Exception as error:
            st.error(str(error))

    comparison = (
        st.session_state.scenario_comparison_df
    )
    if comparison is None:
        return

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    metric = st.selectbox(
        "Grafik metriği",
        [
            "Servis seviyesi (%)",
            "Kayıp talep (adet)",
            "Fazla stok (adet)",
            "Ek sevkiyat (adet)",
            "Kayıp satış değeri (TL)",
        ],
    )
    st.plotly_chart(
        px.bar(
            comparison,
            x="Senaryo",
            y=metric,
            color="Senaryo",
            title=f"Senaryo karşılaştırması — {metric}",
        ),
        use_container_width=True,
    )


def page_manual_fva() -> None:
    st.title("Manuel Düzeltme ve FVA")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    versions = (
        st.session_state.forecast_versions_df
    )
    st.info(
        "Model tahminini değiştirebilir, değişiklik nedenini kaydedebilir ve gerçek satışlar geldiğinde FVA hesaplayabilirsiniz."
    )

    editable_columns = [
        "date",
        "store_id",
        "product_id",
        "naive_forecast",
        "model_forecast",
        "planner_forecast",
        "sales_forecast",
        "approved_forecast",
        "change_reason",
        "comment",
        "changed_by",
    ]
    edited = st.data_editor(
        versions[editable_columns],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=[
            "date",
            "store_id",
            "product_id",
            "naive_forecast",
            "model_forecast",
        ],
        key="forecast_editor",
    )

    if st.button(
        "Manuel tahmin versiyonlarını kaydet",
        type="primary",
    ):
        updated = versions.copy()
        for column in (
            "planner_forecast",
            "sales_forecast",
            "approved_forecast",
            "change_reason",
            "comment",
            "changed_by",
        ):
            updated[column] = edited[column].to_numpy()
        changed_mask = (
            updated["approved_forecast"]
            .ne(updated["model_forecast"])
        )
        updated.loc[
            changed_mask,
            "changed_at",
        ] = pd.Timestamp.now()
        st.session_state.forecast_versions_df = (
            updated
        )
        st.session_state.manual_adjustment_exists = bool(
            changed_mask.any()
        )
        if st.session_state.run_metadata:
            st.session_state.run_metadata[
                "manual_adjustment"
            ] = bool(changed_mask.any())
        st.success("Tahmin versiyonları kaydedildi.")

    st.subheader("Gerçekleşen satışlarla FVA")
    actual_file = st.file_uploader(
        "Tahmin dönemi gerçekleşen satışları",
        type=["csv", "xlsx", "parquet"],
        key="actual_file",
    )
    if actual_file is None:
        st.warning(
            "FVA, tahmin dönemi gerçekleşen satışları gelmeden hesaplanamaz."
        )
        return

    actuals = read_uploaded_table(
        actual_file.getvalue(),
        actual_file.name,
    )
    actual_columns = (
        actuals.columns.astype(str).tolist()
    )

    with st.form("actual_mapping"):
        col1, col2, col3, col4 = st.columns(4)
        actual_date = select_column(
            "Tarih",
            actual_columns,
            "date",
            optional=False,
            key="actual_date",
        )
        actual_store = select_column(
            "Mağaza",
            actual_columns,
            "store",
            optional=False,
            key="actual_store",
        )
        actual_product = select_column(
            "Ürün",
            actual_columns,
            "product",
            optional=False,
            key="actual_product",
        )
        actual_sales = select_column(
            "Gerçek satış",
            actual_columns,
            "sales",
            optional=False,
            key="actual_sales",
        )
        calculate = st.form_submit_button(
            "FVA hesapla",
            type="primary",
        )

    if calculate:
        try:
            fva = build_fva_metrics(
                st.session_state[
                    "forecast_versions_df"
                ],
                actuals,
                actual_date_column=str(actual_date),
                actual_store_column=str(actual_store),
                actual_product_column=str(
                    actual_product
                ),
                actual_sales_column=str(actual_sales),
            )
            st.session_state.fva_metrics_df = fva
        except Exception as error:
            st.error(str(error))

    if st.session_state.fva_metrics_df is not None:
        fva = st.session_state.fva_metrics_df
        st.dataframe(
            fva,
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(
            px.bar(
                fva,
                x="Aşama",
                y="Önceki aşamaya katkı (puan)",
                color="Aşama",
                title="Forecast Value Added",
            ),
            use_container_width=True,
        )


def page_reports() -> None:
    st.title("Raporlar ve Dışa Aktarım")
    if not analysis_ready():
        st.info(
            "Önce Talep Tahmini sayfasında analizi çalıştırın."
        )
        return

    summary = st.session_state.plan_summary_df
    transfer = st.session_state.transfer_df
    before_after = st.session_state.before_after_df
    metadata = st.session_state.run_metadata or {}

    kpis = executive_kpis(summary)
    top_risks = prepare_display_table(
        summary,
        [
            "region",
            "store_id",
            "product_id",
            "plan_status",
            "recommended_replenishment",
            "expected_stockout_date",
            "expected_lost_revenue_no_action",
            "recommended_action",
        ],
        rows=5,
    )
    top_opportunities = (
        prepare_display_table(
            summary.sort_values(
                "excess_stock_units",
                ascending=False,
            ),
            [
                "region",
                "store_id",
                "product_id",
                "excess_stock_units",
                "excess_stock_value",
                "recommended_action",
            ],
            rows=5,
        )
    )

    management_html = build_management_html(
        kpis,
        top_risks,
        top_opportunities,
        before_after,
        metadata,
    )
    action_excel = build_action_excel(
        transfer,
        summary,
        metadata,
    )
    analytics_zip = build_analytics_zip(
        {
            "zero_shot_model_metrics.csv": (
                st.session_state.zero_shot_metrics_df
            ),
            "benchmark_metrics.csv": (
                st.session_state.baseline_metrics_df
            ),
            "future_forecasts.csv": (
                st.session_state.future_forecast_df
            ),
            "stock_projection.csv": (
                st.session_state.plan_detail_df
            ),
            "risk_scores.csv": summary,
            "transfer_actions.csv": transfer,
            "abc_xyz.csv": (
                st.session_state.abc_xyz_df
            ),
            "forecast_versions.csv": (
                st.session_state.forecast_versions_df
            ),
        },
        metadata,
    )

    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Yönetim özetini HTML indir",
        data=management_html,
        file_name="demand_planning_yonetim_ozeti.html",
        mime="text/html",
        use_container_width=True,
    )
    col2.download_button(
        "Operasyon aksiyon Excel'i indir",
        data=action_excel,
        file_name="stok_dagitim_aksiyonlari.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True,
    )
    col3.download_button(
        "Analitik çıktıları ZIP indir",
        data=analytics_zip,
        file_name="demand_planning_analitik_ciktilar.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.subheader("Rapor kapsamı")
    st.write(
        "Her raporda veri kesim tarihi, tahmin oluşturma tarihi, tahmin ufku, "
        "model ve veri versiyonu ile manuel düzenleme bilgisi bulunur."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Alan": list(metadata.keys()),
                "Değer": [
                    str(value)
                    for value in metadata.values()
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_model_data_info() -> None:
    st.title("Model ve Veri Bilgileri")
    if not data_ready():
        st.info(
            "Önce Veri Yükleme sayfasında verileri hazırlayın."
        )
        return

    prepared = st.session_state.prepared_history_df
    plan = st.session_state.future_plan_df
    pipeline = st.session_state.pipeline

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Veri bilgileri")
        data_info = {
            "Veri versiyonu": st.session_state.data_label,
            "Geçmiş başlangıcı": prepared["date"].min(),
            "Veri kesim tarihi": prepared["date"].max(),
            "Gelecek plan başlangıcı": plan["date"].min(),
            "Gelecek plan bitişi": plan["date"].max(),
            "Tahmin ufku": plan["date"].nunique(),
            "Mağaza sayısı": plan["store_id"].nunique(),
            "Ürün sayısı": plan["product_id"].nunique(),
            "Mağaza–ürün sayısı": plan["series_id"].nunique(),
            "Frekans": pipeline.frequency_label_tr,
            "Stok zamanı": pipeline.stock_timing,
        }
        st.dataframe(
            pd.DataFrame(
                {
                    "Alan": list(data_info.keys()),
                    "Değer": [
                        str(value)
                        for value in data_info.values()
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.subheader("Zero-shot model kataloğu")
        model_catalog = pd.DataFrame(
            [
                {
                    "Model Anahtarı": key,
                    "Model": value["display_name"],
                    "Model ID": value["model_id"],
                    "Aile": value["family"],
                    "Final model adayı": "Evet",
                }
                for key, value in MODEL_CONFIGS.items()
            ]
        )
        st.dataframe(
            model_catalog,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Çalıştırılan model")
    if st.session_state.run_metadata:
        st.dataframe(
            pd.DataFrame(
                {
                    "Alan": list(
                        st.session_state.run_metadata.keys()
                    ),
                    "Değer": [
                        str(value)
                        for value in st.session_state.run_metadata.values()
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Henüz tahmin çalıştırılmadı."
        )

    st.warning(
        "Naïve ve hareketli ortalama yöntemleri yalnızca benchmarktır. "
        "Final gelecek tahmini Chronos Bolt, Chronos 2 veya kuruluysa TimesFM zero-shot modelinden üretilir."
    )


inject_css()
initialise_state()

with st.sidebar:
    st.title("Demand Planning AI")
    role = st.selectbox(
        "Rol",
        list(ROLE_PAGE_HINTS.keys()),
    )
    page = st.radio(
        "Menü",
        PAGES,
        index=0,
    )
    st.caption(
        "Bu rol için öncelikli ekranlar: "
        + ", ".join(
            ROLE_PAGE_HINTS[role]
        )
    )
    st.divider()
    if st.button(
        "Oturumu sıfırla",
        use_container_width=True,
    ):
        reset_everything()
        st.rerun()

page_function_map = {
    "1. Ana Sayfa": page_home,
    "2. Veri Yükleme": page_data_upload,
    "3. Veri Kalitesi": page_data_quality,
    "4. Talep Tahmini": page_demand_forecast,
    "5. Tahmin Performansı": page_forecast_performance,
    "6. Stok Riskleri": page_stock_risks,
    "7. Stok Dağıtım Önerileri": page_distribution_recommendations,
    "8. Mağaza / Ürün Detayı": page_store_product_detail,
    "9. ABC–XYZ Önceliklendirme": page_abc_xyz,
    "10. Senaryo Analizi": page_scenario,
    "11. Manuel Düzeltme ve FVA": page_manual_fva,
    "12. Raporlar": page_reports,
    "13. Model ve Veri Bilgileri": page_model_data_info,
}

page_function_map[page]()
