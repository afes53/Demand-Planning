from __future__ import annotations

import gc
import importlib.util
import io
import traceback
import zipfile
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

from distribution_plan_analytics import (
    DistributionPlanMapping,
    align_forecast_with_distribution_plan,
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
HISTORY_SAMPLE_PATH = APP_DIR / "ornek_gecmis_satis.csv"
PLAN_SAMPLE_PATH = APP_DIR / "ornek_gelecek_dagitim_plani.csv"
NONE_OPTION = "— Yok —"

FREQUENCY_DEFAULTS = {
    "hourly": {"backtest_horizon": 24, "min_context": 168},
    "daily": {"backtest_horizon": 7, "min_context": 60},
    "monthly": {"backtest_horizon": 3, "min_context": 24},
}

COLUMN_ALIASES = {
    "date": [
        "Tarih", "Date", "date", "dt", "timestamp",
    ],
    "store": [
        "Magaza_ID", "Mağaza_ID", "Store_ID", "store_id",
        "Store ID",
    ],
    "product": [
        "Urun_ID", "Ürün_ID", "Product_ID", "product_id",
        "Product ID", "SKU",
    ],
    "sales": [
        "Satis_Adedi", "Satış_Adedi", "Units_Sold",
        "sales", "sale_amount", "quantity",
    ],
    "stock": [
        "Stok_Miktari", "Stok_Miktarı", "Inventory_Level",
        "stock", "inventory",
    ],
    "price": [
        "Birim_Fiyat", "Birim Fiyat", "Price", "price",
        "unit_price",
    ],
    "stockout": [
        "Stokout_Flag", "Stockout_Flag", "stockout_flag",
        "is_stockout",
    ],
    "category_1": [
        "Ana_Kategori", "Category", "category", "category_1",
    ],
    "category_2": [
        "Alt_Kategori", "Subcategory", "subcategory", "category_2",
    ],
    "category_3": [
        "Marka_Grubu", "category_3", "Brand_Group",
    ],
    "starting_stock": [
        "Baslangic_Stoku", "Başlangıç_Stoku",
        "Starting_Stock", "starting_stock",
        "current_stock",
    ],
    "planned_shipment": [
        "Planlanan_Sevkiyat", "Planned_Shipment",
        "planned_shipment", "planned_replenishment",
        "shipment",
    ],
}

DISPLAY_NAMES = {
    "date": "Tarih",
    "store_id": "Mağaza",
    "product_id": "Ürün",
    "category_1": "Ana Kategori",
    "category_2": "Alt Kategori",
    "category_3": "Ürün Grubu",
    "sales": "Gerçekleşen Satış",
    "demand_adjusted": "Düzeltilmiş Talep",
    "stock": "Geçmiş Dönem Sonu Stok",
    "price": "Birim Fiyat (TL)",
    "is_stockout": "Stokout",
    "predictions": "Tahmini Talep",
    "starting_stock": "Başlangıç Stoğu",
    "opening_stock": "Dönem Başı Stok",
    "planned_shipment": "Planlanan Sevkiyat",
    "available_stock": "Kullanılabilir Stok",
    "fulfilled_demand": "Karşılanan Talep",
    "period_shortage": "Karşılanamayan Talep",
    "projected_ending_stock": "Plan Sonu Tahmini Stok",
    "safety_stock": "Güvenlik Stoğu",
    "below_safety_stock": "Güvenlik Stoğu Altında",
    "stockout_risk": "Stokout Riski",
    "recommended_extra_shipment": "Önerilen Ek Sevkiyat",
    "recommended_ending_stock": "Öneri Sonrası Tahmini Stok",
    "forecast_revenue": "Tahmini Ciro (TL)",
    "revenue_at_risk": "Risk Altındaki Ciro (TL)",
    "planned_shipment_value": "Planlanan Sevkiyat Değeri (TL)",
    "recommended_extra_shipment_value": "Ek Sevkiyat Değeri (TL)",
    "forecast_start": "Plan Başlangıcı",
    "forecast_end": "Plan Bitişi",
    "horizon_periods": "Plan Dönemi",
    "planned_demand": "Toplam Tahmini Talep",
    "average_period_demand": "Ortalama Dönem Talebi",
    "peak_period_demand": "En Yüksek Dönem Talebi",
    "current_stock": "Başlangıç Stoğu",
    "planned_shipment_total": "Toplam Planlanan Sevkiyat",
    "expected_ending_stock": "Plan Sonu Tahmini Stok",
    "expected_shortage_no_action": "Planla Karşılanamayan Talep",
    "recommended_replenishment": "Önerilen Ek Sevkiyat",
    "expected_stockout_date": "İlk Tahmini Stokout Tarihi",
    "plan_coverage_pct": "Plan Karşılama Oranı (%)",
    "plan_status": "Plan Durumu",
    "latest_price": "Güncel Fiyat (TL)",
    "planned_revenue": "Tahmini Ciro (TL)",
    "expected_lost_revenue_no_action": "Risk Altındaki Ciro (TL)",
    "recommended_replenishment_value": "Önerilen Ek Sevkiyat Değeri (TL)",
    "planned_shipment_value": "Planlanan Sevkiyat Değeri (TL)",
    "stockout_rate_pct": "Tarihsel Stokout Oranı (%)",
    "estimated_lost_demand": "Tarihsel Tahmini Kayıp Talep",
    "estimated_lost_revenue": "Tarihsel Tahmini Kayıp Ciro (TL)",
    "priority": "Risk Seviyesi",
    "operational_priority": "Operasyon Önceliği",
    "abc_class": "ABC Sınıfı",
    "revenue_priority_score": "Ciro Risk Puanı",
    "abc_adjusted_priority_score": "ABC Düzeltilmiş Puan",
    "recommended_action": "Önerilen Aksiyon",
    "abc_stock_action": "ABC Stok Aksiyonu",
    "model_name": "Model",
    "wmape_pct": "WMAPE (%)",
    "mae": "MAE",
    "rmse": "RMSE",
    "bias_pct": "Bias (%)",
    "runtime_seconds": "Süre (sn)",
    "total_value": "Toplam Ticari Değer (TL)",
    "value_share_pct": "Değer Payı (%)",
    "cumulative_value_pct": "Kümülatif Değer (%)",
    "abc_description": "Açıklama",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.28);
            background: rgba(248,250,252,.78);
            border-radius: 14px;
            padding: 14px 16px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,.23);
            border-radius: 12px;
            overflow: hidden;
        }
        .flow-card {
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 14px;
            padding: 15px 17px;
            background: rgba(248,250,252,.72);
            min-height: 120px;
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
        "history_has_price": False,
        "history_has_stock": False,
        "data_label": None,
        "metrics_df": None,
        "evaluations_df": None,
        "model_errors_df": None,
        "future_forecast_df": None,
        "aligned_plan_df": None,
        "plan_detail_df": None,
        "plan_summary_df": None,
        "historical_loss_detail_df": None,
        "historical_loss_summary_df": None,
        "management_kpis_df": None,
        "abc_product_df": None,
        "abc_summary_df": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_outputs() -> None:
    for key in (
        "metrics_df",
        "evaluations_df",
        "model_errors_df",
        "future_forecast_df",
        "aligned_plan_df",
        "plan_detail_df",
        "plan_summary_df",
        "historical_loss_detail_df",
        "historical_loss_summary_df",
        "management_kpis_df",
        "abc_product_df",
        "abc_summary_df",
    ):
        st.session_state[key] = None


def reset_everything() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialise_state()


def format_number(value: Any, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_try(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{format_number(value, 2)} TL"


@st.cache_data(show_spinner=False)
def read_uploaded_table(
    file_bytes: bytes,
    file_name: str,
) -> pd.DataFrame:
    extension = file_name.lower().rsplit(".", maxsplit=1)[-1]
    stream = io.BytesIO(file_bytes)

    if extension == "csv":
        return pd.read_csv(stream, sep=None, engine="python")
    if extension == "xlsx":
        return pd.read_excel(stream)
    if extension == "parquet":
        return pd.read_parquet(stream)

    raise ValueError("Desteklenen dosyalar: CSV, XLSX ve Parquet.")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def zip_tables(
    tables: dict[str, Optional[pd.DataFrame]],
) -> bytes:
    buffer = io.BytesIO()
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
                csv_bytes(table),
            )
    return buffer.getvalue()


def normalise_name(value: str) -> str:
    translation = str.maketrans(
        {
            "ı": "i",
            "İ": "i",
            "ş": "s",
            "Ş": "s",
            "ğ": "g",
            "Ğ": "g",
            "ü": "u",
            "Ü": "u",
            "ö": "o",
            "Ö": "o",
            "ç": "c",
            "Ç": "c",
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
    normalised_columns = {
        normalise_name(column): column
        for column in columns
    }
    for alias in COLUMN_ALIASES[field]:
        match = normalised_columns.get(
            normalise_name(alias)
        )
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
    default_value = (
        guessed if guessed in options else options[0]
    )
    selected = st.selectbox(
        label,
        options,
        index=options.index(default_value),
        key=key,
    )
    if optional and selected == NONE_OPTION:
        return None
    return selected


def prepare_display_table(
    df: pd.DataFrame,
    columns: list[str],
    *,
    rows: Optional[int] = None,
) -> pd.DataFrame:
    available = [
        column
        for column in columns
        if column in df.columns
    ]
    result = df[available].copy()

    for column in result.columns:
        if (
            "date" in column.lower()
            or column in {"forecast_start", "forecast_end"}
        ):
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).dt.strftime("%d.%m.%Y")
        elif (
            pd.api.types.is_bool_dtype(result[column])
            or str(result[column].dtype) == "boolean"
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

    return result.rename(columns=DISPLAY_NAMES).head(rows)


def build_pipeline(
    *,
    working_history_df: pd.DataFrame,
    date_column: str,
    store_column: str,
    product_column: str,
    sales_column: str,
    stock_column: Optional[str],
    price_column: Optional[str],
    stockout_column: Optional[str],
    category_1: Optional[str],
    category_2: Optional[str],
    category_3: Optional[str],
    frequency: str,
    dayfirst: bool,
    duplicate_policy: str,
    date_gap_policy: str,
    min_history: int,
) -> tuple[pd.DataFrame, DemandDataPipeline, bool]:
    history = working_history_df.copy()
    history_has_stock = stock_column is not None

    if stock_column is None:
        stock_column = "__history_stock_proxy__"
        history[stock_column] = 1.0

    if stockout_column is None:
        stockout_column = "__history_stockout_proxy__"
        history[stockout_column] = False

    mapping = ColumnMapping(
        date=date_column,
        store=store_column,
        product=product_column,
        sales=sales_column,
        stock=stock_column,
        price=price_column,
        category_1=category_1,
        category_2=category_2,
        category_3=category_3,
        stockout_flag=stockout_column,
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
        combine_provided_flag_with_inferred=(
            history_has_stock
        ),
        stock_timing="end_of_period",
        imputation_window=28,
        min_history=min_history,
        imputation_statistic="median",
    )

    prepared = pipeline.prepare(history)
    return prepared, pipeline, history_has_stock


def save_prepared_bundle(
    *,
    history_raw_df: pd.DataFrame,
    plan_raw_df: pd.DataFrame,
    pipeline: DemandDataPipeline,
    prepared_history_df: pd.DataFrame,
    analysis_history_df: pd.DataFrame,
    future_plan_df: pd.DataFrame,
    history_has_price: bool,
    history_has_stock: bool,
    label: str,
) -> None:
    st.session_state.history_raw_df = history_raw_df
    st.session_state.plan_raw_df = plan_raw_df
    st.session_state.pipeline = pipeline
    st.session_state.prepared_history_df = (
        prepared_history_df
    )
    st.session_state.analysis_history_df = (
        analysis_history_df
    )
    st.session_state.future_plan_df = future_plan_df
    st.session_state.history_has_price = (
        history_has_price
    )
    st.session_state.history_has_stock = (
        history_has_stock
    )
    st.session_state.data_label = label
    reset_outputs()


def run_forecast(
    *,
    model_keys: list[str],
    backtest_horizon: int,
    min_context: int,
) -> None:
    prepared_history = (
        st.session_state.prepared_history_df
    )
    future_plan = st.session_state.future_plan_df
    pipeline = st.session_state.pipeline

    if (
        prepared_history is None
        or future_plan is None
        or pipeline is None
    ):
        raise ValueError(
            "Önce geçmiş satış ve gelecek dağıtım planını hazırlayın."
        )

    plan_series_ids = (
        future_plan["series_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    model_history = prepared_history.loc[
        prepared_history["series_id"].isin(
            plan_series_ids
        )
    ].copy()

    plan_horizon = int(
        future_plan["date"].nunique()
    )
    max_series = len(plan_series_ids)

    if len(model_keys) == 1:
        model_key = model_keys[0]
        forecaster = None
        mvp = None

        try:
            forecaster = create_forecaster(
                model_key,
                freq=pipeline.pandas_freq,
            )
            mvp = DemandForecastMVP(
                pipeline,
                forecaster,
            )

            evaluations_df, metrics = mvp.backtest(
                prepared_df=model_history,
                horizon=backtest_horizon,
                min_context=min_context,
                max_series=max_series,
            )
            metrics.update(
                {
                    "model_key": model_key,
                    "model_name": MODEL_CONFIGS[
                        model_key
                    ]["display_name"],
                }
            )
            metrics_df = pd.DataFrame([metrics])
            evaluations_df["model_key"] = model_key
            evaluations_df["model_name"] = (
                metrics["model_name"]
            )
            model_errors_df = pd.DataFrame()

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

    else:
        (
            metrics_df,
            evaluations_df,
            model_errors_df,
        ) = compare_zero_shot_models(
            prepared_df=model_history,
            data_pipeline=pipeline,
            model_keys=model_keys,
            horizon=backtest_horizon,
            min_context=min_context,
            max_series=max_series,
        )

        if metrics_df.empty:
            raise RuntimeError(
                "Seçilen modeller başarıyla tamamlanamadı."
            )

        metrics_df = (
            metrics_df.sort_values(
                ["wmape_pct", "mae"],
                ascending=True,
            )
            .reset_index(drop=True)
        )

        best_model_key = str(
            metrics_df.iloc[0]["model_key"]
        )
        forecaster = None
        mvp = None

        try:
            forecaster = create_forecaster(
                best_model_key,
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

    future_forecast["predictions"] = (
        pd.to_numeric(
            future_forecast["predictions"],
            errors="coerce",
        )
        .clip(lower=0)
    )

    aligned_plan = (
        align_forecast_with_distribution_plan(
            future_forecast,
            future_plan,
        )
    )

    st.session_state.metrics_df = metrics_df
    st.session_state.evaluations_df = evaluations_df
    st.session_state.model_errors_df = (
        model_errors_df
    )
    st.session_state.future_forecast_df = (
        future_forecast
    )
    st.session_state.aligned_plan_df = aligned_plan


def run_plan_analysis(
    *,
    safety_periods: float,
) -> None:
    history_analysis = (
        st.session_state.analysis_history_df
    )
    aligned_plan = st.session_state.aligned_plan_df

    if history_analysis is None or aligned_plan is None:
        raise ValueError(
            "Önce tahminleri dağıtım planıyla eşleştirin."
        )

    (
        historical_loss_detail,
        historical_loss_summary,
    ) = build_historical_loss_analysis(
        history_analysis
    )

    plan_detail, plan_summary = simulate_distribution_plan(
        aligned_plan,
        historical_loss_summary_df=(
            historical_loss_summary
        ),
        safety_periods=safety_periods,
    )

    has_future_revenue = (
        "expected_lost_revenue_no_action"
        in plan_summary.columns
        and "latest_price" in plan_summary.columns
    )

    abc_product = None
    abc_summary = None

    if has_future_revenue:
        # Dağıtım planına özel açıklamayı fiyat bazlı puanlamadan önce koru.
        plan_action_lookup = plan_summary[
            ["series_id", "recommended_action"]
        ].rename(
            columns={
                "recommended_action":
                    "distribution_plan_action"
            }
        )

        plan_summary = apply_revenue_weighted_priority(
            plan_summary
        )

        plan_summary = plan_summary.merge(
            plan_action_lookup,
            on="series_id",
            how="left",
            validate="one_to_one",
        )
        plan_summary["recommended_action"] = (
            plan_summary[
                "distribution_plan_action"
            ]
        )
        plan_summary = plan_summary.drop(
            columns=["distribution_plan_action"]
        )

        if (
            "observed_revenue"
            in historical_loss_summary.columns
            and "estimated_lost_revenue"
            in historical_loss_summary.columns
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
        abc_summary = build_abc_management_summary(
            plan_summary
        )

    st.session_state.historical_loss_detail_df = (
        historical_loss_detail
    )
    st.session_state.historical_loss_summary_df = (
        historical_loss_summary
    )
    st.session_state.plan_detail_df = plan_detail
    st.session_state.plan_summary_df = plan_summary
    st.session_state.management_kpis_df = (
        build_management_kpis(
            historical_loss_summary,
            plan_summary,
        )
    )
    st.session_state.abc_product_df = abc_product
    st.session_state.abc_summary_df = abc_summary


def render_data_overview() -> None:
    history = st.session_state.prepared_history_df
    plan = st.session_state.future_plan_df

    if history is None or plan is None:
        return

    history_start = pd.to_datetime(
        history["date"]
    ).min()
    history_end = pd.to_datetime(
        history["date"]
    ).max()
    plan_start = pd.to_datetime(
        plan["date"]
    ).min()
    plan_end = pd.to_datetime(
        plan["date"]
    ).max()

    st.success(
        f"Veriler hazır: {st.session_state.data_label}"
    )

    cards = st.columns(6)
    cards[0].metric(
        "Geçmiş Aralığı",
        f"{history_start:%d.%m.%Y}",
        f"{history_end:%d.%m.%Y}",
    )
    cards[1].metric(
        "Plan Aralığı",
        f"{plan_start:%d.%m.%Y}",
        f"{plan_end:%d.%m.%Y}",
    )
    cards[2].metric(
        "Plan Dönemi",
        format_number(plan["date"].nunique()),
    )
    cards[3].metric(
        "Mağaza",
        format_number(plan["store_id"].nunique()),
    )
    cards[4].metric(
        "Ürün",
        format_number(plan["product_id"].nunique()),
    )
    cards[5].metric(
        "Planlanan Sevkiyat",
        format_number(plan["planned_shipment"].sum()),
    )

    history_tab, plan_tab = st.tabs(
        ["Geçmiş satış verisi", "Gelecek dağıtım planı"]
    )

    with history_tab:
        st.dataframe(
            prepare_display_table(
                history,
                [
                    "date",
                    "store_id",
                    "product_id",
                    "sales",
                    "stock",
                    "price",
                    "is_stockout",
                    "category_1",
                    "category_2",
                    "category_3",
                ],
                rows=100,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with plan_tab:
        st.dataframe(
            prepare_display_table(
                plan,
                [
                    "date",
                    "store_id",
                    "product_id",
                    "starting_stock",
                    "planned_shipment",
                    "price",
                    "category_1",
                    "category_2",
                    "category_3",
                ],
                rows=100,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Veri doğrulama raporu"):
        report = st.session_state.pipeline.report
        if report.get("warnings"):
            for warning in report["warnings"]:
                st.warning(str(warning))
        else:
            st.info("Kritik veri uyarısı bulunmuyor.")
        st.json(report.get("stats", {}))


def series_selector(
    key_prefix: str,
) -> Optional[str]:
    plan = st.session_state.future_plan_df
    if plan is None:
        return None

    metadata = (
        plan[
            ["series_id", "store_id", "product_id"]
        ]
        .drop_duplicates()
        .copy()
    )

    stores = sorted(
        metadata["store_id"].astype(str).unique()
    )
    selected_store = st.selectbox(
        "Mağaza",
        stores,
        key=f"{key_prefix}_store",
    )

    products = sorted(
        metadata.loc[
            metadata["store_id"]
            .astype(str)
            .eq(selected_store),
            "product_id",
        ]
        .astype(str)
        .unique()
    )
    selected_product = st.selectbox(
        "Ürün",
        products,
        key=f"{key_prefix}_product",
    )

    matched = metadata.loc[
        metadata["store_id"]
        .astype(str)
        .eq(selected_store)
        & metadata["product_id"]
        .astype(str)
        .eq(selected_product),
        "series_id",
    ]

    return (
        None
        if matched.empty
        else str(matched.iloc[0])
    )


def render_forecast_and_stock_view() -> None:
    metrics = st.session_state.metrics_df
    evaluations = st.session_state.evaluations_df
    plan_detail = st.session_state.plan_detail_df
    history = st.session_state.analysis_history_df

    if (
        metrics is None
        or plan_detail is None
        or history is None
    ):
        return

    best = metrics.sort_values("wmape_pct").iloc[0]
    cards = st.columns(5)
    cards[0].metric(
        "Seçilen Model",
        str(best.get("model_name", "—")),
    )
    cards[1].metric(
        "WMAPE",
        f"%{format_number(best.get('wmape_pct'), 2)}",
    )
    cards[2].metric(
        "MAE",
        format_number(best.get("mae"), 2),
    )
    cards[3].metric(
        "Plan Başlangıcı",
        f"{pd.to_datetime(plan_detail['date']).min():%d.%m.%Y}",
    )
    cards[4].metric(
        "Plan Bitişi",
        f"{pd.to_datetime(plan_detail['date']).max():%d.%m.%Y}",
    )

    st.dataframe(
        prepare_display_table(
            metrics,
            [
                "model_name",
                "wmape_pct",
                "mae",
                "rmse",
                "bias_pct",
                "runtime_seconds",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Mağaza–ürün plan kontrolü")
    selected_series = series_selector("series_view")
    if selected_series is None:
        return

    selected_history = (
        history.loc[
            history["series_id"]
            .astype(str)
            .eq(selected_series)
        ]
        .sort_values("date")
        .tail(70)
        .copy()
    )
    selected_plan = (
        plan_detail.loc[
            plan_detail["series_id"]
            .astype(str)
            .eq(selected_series)
        ]
        .sort_values("date")
        .copy()
    )

    demand_figure = go.Figure()
    demand_figure.add_trace(
        go.Scatter(
            x=selected_history["date"],
            y=selected_history["sales"],
            name="Gerçekleşen satış",
            mode="lines",
        )
    )
    demand_figure.add_trace(
        go.Scatter(
            x=selected_history["date"],
            y=selected_history["demand_adjusted"],
            name="Stokout düzeltilmiş talep",
            mode="lines",
            line={"dash": "dot"},
        )
    )
    demand_figure.add_trace(
        go.Scatter(
            x=selected_plan["date"],
            y=selected_plan["predictions"],
            name="Plan tarihleri için tahmini talep",
            mode="lines+markers",
        )
    )
    demand_figure.update_layout(
        title="Geçmiş satış ve plan tarihleri için talep tahmini",
        xaxis_title="Tarih",
        yaxis_title="Adet",
        hovermode="x unified",
    )
    st.plotly_chart(
        demand_figure,
        use_container_width=True,
    )
    st.caption(
        "Tahmin çizgisi, doğrudan yüklenen gelecek dağıtım planındaki "
        "tarihler için üretilir."
    )

    shipment_figure = go.Figure()
    shipment_figure.add_trace(
        go.Bar(
            x=selected_plan["date"],
            y=selected_plan["planned_shipment"],
            name="Planlanan sevkiyat",
        )
    )
    shipment_figure.add_trace(
        go.Bar(
            x=selected_plan["date"],
            y=selected_plan[
                "recommended_extra_shipment"
            ],
            name="Önerilen ek sevkiyat",
        )
    )
    shipment_figure.add_trace(
        go.Scatter(
            x=selected_plan["date"],
            y=selected_plan["predictions"],
            name="Tahmini talep",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    shipment_figure.update_layout(
        title="Planlanan dağıtım, ek sevkiyat ve tahmini talep",
        xaxis_title="Tarih",
        yaxis={
            "title": "Sevkiyat adedi",
        },
        yaxis2={
            "title": "Tahmini talep",
            "overlaying": "y",
            "side": "right",
        },
        barmode="group",
        hovermode="x unified",
    )
    st.plotly_chart(
        shipment_figure,
        use_container_width=True,
    )
    st.caption(
        "Barlar mevcut dağıtım planını ve sistemin önerdiği ek miktarı; "
        "çizgi ise aynı tarihte beklenen talebi gösterir."
    )

    stock_figure = go.Figure()
    stock_figure.add_trace(
        go.Scatter(
            x=selected_plan["date"],
            y=selected_plan[
                "projected_ending_stock"
            ],
            name="Mevcut planla kalan stok",
            mode="lines+markers",
            fill="tozeroy",
        )
    )
    stock_figure.add_trace(
        go.Scatter(
            x=selected_plan["date"],
            y=selected_plan[
                "recommended_ending_stock"
            ],
            name="Öneri uygulanırsa kalan stok",
            mode="lines+markers",
        )
    )
    stock_figure.add_trace(
        go.Scatter(
            x=selected_plan["date"],
            y=selected_plan["safety_stock"],
            name="Güvenlik stoğu",
            mode="lines",
            line={"dash": "dash"},
        )
    )
    stock_figure.update_layout(
        title="Dağıtım planına göre stok seyri",
        xaxis_title="Tarih",
        yaxis_title="Stok adedi",
        hovermode="x unified",
    )
    st.plotly_chart(
        stock_figure,
        use_container_width=True,
    )
    st.caption(
        "Mevcut planın stok seviyesini ve önerilen ek sevkiyatların "
        "uygulanması durumundaki stok seviyesini karşılaştırır."
    )

    detail_columns = [
        "date",
        "store_id",
        "product_id",
        "opening_stock",
        "planned_shipment",
        "predictions",
        "available_stock",
        "projected_ending_stock",
        "period_shortage",
        "recommended_extra_shipment",
        "recommended_ending_stock",
        "stockout_risk",
        "revenue_at_risk",
    ]
    st.dataframe(
        prepare_display_table(
            selected_plan,
            detail_columns,
        ),
        use_container_width=True,
        hide_index=True,
    )

    if evaluations is not None and not evaluations.empty:
        with st.expander("Backtest ayrıntısı"):
            st.dataframe(
                prepare_display_table(
                    evaluations,
                    [
                        "date",
                        "store_id",
                        "product_id",
                        "actual",
                        "predictions",
                        "is_stockout",
                    ],
                    rows=500,
                ),
                use_container_width=True,
                hide_index=True,
            )


def plan_status_donut(
    summary: pd.DataFrame,
) -> go.Figure:
    status = (
        summary["plan_status"]
        .value_counts()
        .rename_axis("Plan Durumu")
        .reset_index(name="Mağaza–Ürün Sayısı")
    )
    return px.pie(
        status,
        names="Plan Durumu",
        values="Mağaza–Ürün Sayısı",
        hole=0.55,
        title="Dağıtım planı yeterlilik dağılımı",
    )


def plan_waterfall(
    detail: pd.DataFrame,
) -> go.Figure:
    initial_stock = (
        detail.sort_values("date")
        .groupby("series_id")["starting_stock"]
        .first()
        .sum()
    )
    planned_shipments = detail[
        "planned_shipment"
    ].sum()
    forecast_demand = detail[
        "predictions"
    ].sum()
    ending_stock = (
        detail.sort_values("date")
        .groupby("series_id")[
            "projected_ending_stock"
        ]
        .last()
        .sum()
    )

    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=[
                "absolute",
                "relative",
                "relative",
                "total",
            ],
            x=[
                "Başlangıç stoğu",
                "Planlanan sevkiyat",
                "Tahmini talep",
                "Plan sonu stok",
            ],
            y=[
                initial_stock,
                planned_shipments,
                -forecast_demand,
                ending_stock,
            ],
            connector={
                "line": {"dash": "dot"},
            },
        )
    )
    figure.update_layout(
        title="Toplam stok planı denge analizi",
        yaxis_title="Adet",
    )
    return figure


def store_distribution_bar(
    summary: pd.DataFrame,
) -> go.Figure:
    store_summary = (
        summary.groupby(
            "store_id",
            as_index=False,
        )
        .agg(
            Planlanan_Sevkiyat=(
                "planned_shipment_total",
                "sum",
            ),
            Onerilen_Ek_Sevkiyat=(
                "recommended_replenishment",
                "sum",
            ),
            Tahmini_Talep=(
                "planned_demand",
                "sum",
            ),
        )
        .sort_values(
            "Onerilen_Ek_Sevkiyat",
            ascending=False,
        )
        .head(20)
    )
    melted = store_summary.melt(
        id_vars="store_id",
        var_name="Gösterge",
        value_name="Adet",
    )
    return px.bar(
        melted,
        x="store_id",
        y="Adet",
        color="Gösterge",
        barmode="group",
        title="Mağaza bazında talep ve dağıtım karşılaştırması",
        labels={"store_id": "Mağaza"},
    )


def extra_shipment_heatmap(
    summary: pd.DataFrame,
) -> go.Figure:
    top_stores = (
        summary.groupby("store_id")[
            "recommended_replenishment"
        ]
        .sum()
        .nlargest(12)
        .index
    )
    top_products = (
        summary.groupby("product_id")[
            "recommended_replenishment"
        ]
        .sum()
        .nlargest(18)
        .index
    )

    filtered = summary.loc[
        summary["store_id"].isin(top_stores)
        & summary["product_id"].isin(top_products)
    ]
    pivot = filtered.pivot_table(
        index="store_id",
        columns="product_id",
        values="recommended_replenishment",
        aggfunc="sum",
        fill_value=0,
    )
    return px.imshow(
        pivot,
        aspect="auto",
        labels={
            "x": "Ürün",
            "y": "Mağaza",
            "color": "Ek sevkiyat",
        },
        title="Önerilen ek sevkiyat heatmap'i",
    )


def risk_bubble(
    summary: pd.DataFrame,
) -> go.Figure:
    y_column = (
        "expected_lost_revenue_no_action"
        if "expected_lost_revenue_no_action"
        in summary.columns
        else "expected_shortage_no_action"
    )
    plot_df = summary.copy()
    plot_df["Konum"] = (
        plot_df["store_id"].astype(str)
        + " / "
        + plot_df["product_id"].astype(str)
    )

    return px.scatter(
        plot_df,
        x="plan_coverage_pct",
        y=y_column,
        size="recommended_replenishment",
        color=(
            "abc_class"
            if "abc_class" in plot_df.columns
            else "plan_status"
        ),
        hover_name="Konum",
        labels={
            "plan_coverage_pct": "Plan karşılama oranı (%)",
            y_column: DISPLAY_NAMES.get(
                y_column,
                y_column,
            ),
            "recommended_replenishment": (
                "Önerilen ek sevkiyat"
            ),
        },
        title="Plan karşılama oranı ve iş etkisi risk matrisi",
    )


def revenue_risk_bar(
    summary: pd.DataFrame,
) -> go.Figure:
    top = (
        summary.nlargest(
            15,
            "expected_lost_revenue_no_action",
        )
        .copy()
    )
    top["Konum"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values(
        "expected_lost_revenue_no_action"
    )
    return px.bar(
        top,
        x="expected_lost_revenue_no_action",
        y="Konum",
        orientation="h",
        color=(
            "abc_class"
            if "abc_class" in top.columns
            else "plan_status"
        ),
        labels={
            "expected_lost_revenue_no_action": (
                "Risk altındaki ciro (TL)"
            ),
        },
        title="Mevcut planla en yüksek ciro riski",
    )


def abc_treemap(
    abc_product: pd.DataFrame,
) -> go.Figure:
    plot_df = abc_product.copy()
    plot_df["Ürün"] = (
        plot_df["product_id"].astype(str)
    )
    return px.treemap(
        plot_df,
        path=["abc_class", "Ürün"],
        values="total_value",
        color="abc_class",
        title="ABC ürün portföyü",
    )


def render_dashboard() -> None:
    summary = st.session_state.plan_summary_df
    detail = st.session_state.plan_detail_df
    history = (
        st.session_state.historical_loss_summary_df
    )
    abc_product = st.session_state.abc_product_df
    abc_summary = st.session_state.abc_summary_df

    if summary is None or detail is None:
        st.info(
            "Tahmin ve Stok sekmesinden analizleri çalıştırın."
        )
        return

    insufficient = int(
        summary["plan_status"].eq("Yetersiz").sum()
    )
    below_safety = int(
        summary["plan_status"]
        .eq("Güvenlik stoğu altında")
        .sum()
    )
    coverage = (
        summary["fulfilled_demand"].sum()
        / summary["planned_demand"].sum()
        * 100
        if summary["planned_demand"].sum() > 0
        else 100
    )

    cards = st.columns(7)
    cards[0].metric(
        "Tahmini Talep",
        format_number(summary["planned_demand"].sum()),
    )
    cards[1].metric(
        "Planlanan Sevkiyat",
        format_number(
            summary["planned_shipment_total"].sum()
        ),
    )
    cards[2].metric(
        "Önerilen Ek Sevkiyat",
        format_number(
            summary["recommended_replenishment"].sum()
        ),
    )
    cards[3].metric(
        "Yetersiz Plan",
        format_number(insufficient),
    )
    cards[4].metric(
        "Güvenlik Stoğu Altında",
        format_number(below_safety),
    )
    cards[5].metric(
        "Talep Karşılama",
        f"%{format_number(coverage, 2)}",
    )
    cards[6].metric(
        "Ciro Riski",
        format_try(
            summary.get(
                "expected_lost_revenue_no_action",
                pd.Series(dtype=float),
            ).sum()
        ),
    )

    tabs = st.tabs(
        [
            "Yönetici Özeti",
            "Dağıtım ve Stok",
            "Risk ve Ciro",
            "ABC Portföyü",
            "Detay ve İndirme",
        ]
    )

    with tabs[0]:
        st.markdown("#### Öncelikli aksiyonlar")
        action_columns = [
            "operational_priority",
            "abc_class",
            "store_id",
            "product_id",
            "plan_status",
            "current_stock",
            "planned_shipment_total",
            "planned_demand",
            "expected_ending_stock",
            "expected_shortage_no_action",
            "recommended_replenishment",
            "expected_stockout_date",
            "expected_lost_revenue_no_action",
            "recommended_action",
        ]
        st.dataframe(
            prepare_display_table(
                summary,
                action_columns,
                rows=30,
            ),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plan_status_donut(summary),
                use_container_width=True,
            )
            st.caption(
                "Dağıtım planının kaç mağaza–üründe yeterli, "
                "riskli veya yetersiz olduğunu gösterir."
            )
        with col2:
            st.plotly_chart(
                risk_bubble(summary),
                use_container_width=True,
            )
            st.caption(
                "Düşük karşılama oranı ve yüksek iş etkisine sahip "
                "noktaları önceliklendirir."
            )

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plan_waterfall(detail),
                use_container_width=True,
            )
            st.caption(
                "Başlangıç stoğu, planlanan sevkiyat ve tahmini "
                "talebin toplam stok dengesine etkisini gösterir."
            )
        with col2:
            st.plotly_chart(
                store_distribution_bar(summary),
                use_container_width=True,
            )
            st.caption(
                "Mağaza bazında tahmini talebi, mevcut planı ve "
                "önerilen ek sevkiyatı karşılaştırır."
            )

        st.plotly_chart(
            extra_shipment_heatmap(summary),
            use_container_width=True,
        )
        st.caption(
            "Ek sevkiyat ihtiyacının mağaza ve ürünlerde nerede "
            "yoğunlaştığını gösterir."
        )

        stock_columns = [
            "plan_status",
            "store_id",
            "product_id",
            "current_stock",
            "planned_shipment_total",
            "planned_demand",
            "safety_stock",
            "expected_ending_stock",
            "expected_shortage_no_action",
            "recommended_replenishment",
            "recommended_ending_stock",
            "expected_stockout_date",
            "plan_coverage_pct",
        ]
        st.dataframe(
            prepare_display_table(
                summary,
                stock_columns,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        if (
            "expected_lost_revenue_no_action"
            not in summary.columns
        ):
            st.warning(
                "Ciro analizi için geçmiş satış veya gelecek plan "
                "dosyasında fiyat sütunu bulunmalıdır."
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    revenue_risk_bar(summary),
                    use_container_width=True,
                )
                st.caption(
                    "Mevcut dağıtım planı değiştirilmezse en yüksek "
                    "ciro riski yaratabilecek noktaları gösterir."
                )
            with col2:
                historical_loss = float(
                    history[
                        "estimated_lost_revenue"
                    ].sum()
                )
                future_risk = float(
                    summary[
                        "expected_lost_revenue_no_action"
                    ].sum()
                )
                planned_revenue = float(
                    summary["planned_revenue"].sum()
                )
                revenue_df = pd.DataFrame(
                    {
                        "Gösterge": [
                            "Tarihsel stokout kaybı",
                            "Gelecek plan ciro riski",
                            "Plan dönemi tahmini ciro",
                        ],
                        "Tutar": [
                            historical_loss,
                            future_risk,
                            planned_revenue,
                        ],
                    }
                )
                st.plotly_chart(
                    px.bar(
                        revenue_df,
                        x="Gösterge",
                        y="Tutar",
                        title=(
                            "Tarihsel kayıp ve gelecek planın "
                            "parasal görünümü"
                        ),
                        labels={"Tutar": "Tutar (TL)"},
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Geçmiş stokout kaybını, gelecek planın riskini "
                    "ve tahmini ciroyu aynı ölçekte karşılaştırır."
                )

            revenue_columns = [
                "abc_class",
                "store_id",
                "product_id",
                "latest_price",
                "planned_demand",
                "planned_revenue",
                "planned_shipment_total",
                "planned_shipment_value",
                "recommended_replenishment",
                "recommended_replenishment_value",
                "expected_lost_revenue_no_action",
                "revenue_priority_score",
            ]
            st.dataframe(
                prepare_display_table(
                    summary,
                    revenue_columns,
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        if abc_product is None:
            st.warning(
                "ABC analizi için fiyat bilgisi gerekir."
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    abc_treemap(abc_product),
                    use_container_width=True,
                )
                st.caption(
                    "Ürünlerin ticari değerini ve ABC sınıfını "
                    "portföy görünümünde gösterir."
                )
            with col2:
                class_summary = (
                    abc_product.groupby(
                        "abc_class",
                        as_index=False,
                    )
                    .agg(
                        total_value=(
                            "total_value",
                            "sum",
                        ),
                        product_count=(
                            "product_id",
                            "nunique",
                        ),
                    )
                )
                st.plotly_chart(
                    px.pie(
                        class_summary,
                        names="abc_class",
                        values="total_value",
                        hole=0.55,
                        title="ABC sınıflarının ticari değer payı",
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "A, B ve C sınıflarının toplam ürün değerindeki "
                    "payını gösterir."
                )

            st.dataframe(
                prepare_display_table(
                    abc_product,
                    [
                        "product_id",
                        "abc_class",
                        "total_value",
                        "value_share_pct",
                        "cumulative_value_pct",
                        "abc_description",
                    ],
                ),
                use_container_width=True,
                hide_index=True,
            )

            if abc_summary is not None:
                st.markdown("#### ABC sınıfı operasyon özeti")
                st.dataframe(
                    prepare_display_table(
                        abc_summary,
                        list(abc_summary.columns),
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[4]:
        table_choice = st.selectbox(
            "Gösterilecek detay",
            [
                "Mağaza–ürün plan özeti",
                "Tarih bazlı stok planı",
                "Tarihsel stokout analizi",
                "Model metrikleri",
            ],
        )

        table_map = {
            "Mağaza–ürün plan özeti": summary,
            "Tarih bazlı stok planı": detail,
            "Tarihsel stokout analizi": history,
            "Model metrikleri": (
                st.session_state.metrics_df
            ),
        }
        selected_table = table_map[table_choice]

        st.dataframe(
            prepare_display_table(
                selected_table,
                list(selected_table.columns),
                rows=3000,
            ),
            use_container_width=True,
            hide_index=True,
        )

        downloads = {
            "model_metrikleri.csv": (
                st.session_state.metrics_df
            ),
            "gelecek_talep_tahminleri.csv": (
                st.session_state.future_forecast_df
            ),
            "tarih_bazli_dagitim_plani_analizi.csv": (
                detail
            ),
            "magaza_urun_plan_ozeti.csv": summary,
            "tarihsel_stokout_analizi.csv": history,
            "yonetici_kpi.csv": (
                st.session_state.management_kpis_df
            ),
            "abc_urun_analizi.csv": abc_product,
            "abc_sinif_ozeti.csv": abc_summary,
        }

        st.download_button(
            "Tüm sonuçları ZIP olarak indir",
            data=zip_tables(downloads),
            file_name="demand_planning_sonuclari.zip",
            mime="application/zip",
        )


inject_css()
initialise_state()

st.title("📦 Demand Planning AI")
st.caption(
    "Geçmiş satışlardan talep tahmini üretir ve yüklenen gelecek "
    "stok dağıtım planının yeterliliğini test eder."
)

with st.sidebar:
    st.header("Uygulama Akışı")
    st.write("1. Geçmiş satış verisini yükle")
    st.write("2. Gelecek dağıtım planını yükle")
    st.write("3. Plan tarihleri için talep tahmini üret")
    st.write("4. Plan yeterliliği ve ek sevkiyatı analiz et")
    st.divider()
    if st.button(
        "Oturumu sıfırla",
        use_container_width=True,
    ):
        reset_everything()
        st.rerun()

data_tab, forecast_tab, dashboard_tab = st.tabs(
    [
        "1 · Veri ve Dağıtım Planı",
        "2 · Tahmin ve Stok",
        "3 · Karar Destek Dashboard",
    ]
)

with data_tab:
    st.subheader("Dosyaları yükle")

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Örnek geçmiş satış verisini indir",
            data=HISTORY_SAMPLE_PATH.read_bytes(),
            file_name="ornek_gecmis_satis.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            "Örnek gelecek dağıtım planını indir",
            data=PLAN_SAMPLE_PATH.read_bytes(),
            file_name="ornek_gelecek_dagitim_plani.csv",
            mime="text/csv",
            use_container_width=True,
        )

    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        history_file = st.file_uploader(
            "Geçmiş satış verisi",
            type=["csv", "xlsx", "parquet"],
            key="history_uploader",
        )
    with upload_col2:
        plan_file = st.file_uploader(
            "Gelecek stok dağıtım planı",
            type=["csv", "xlsx", "parquet"],
            key="plan_uploader",
        )

    if history_file is not None and plan_file is not None:
        try:
            history_raw = read_uploaded_table(
                history_file.getvalue(),
                history_file.name,
            )
            plan_raw = read_uploaded_table(
                plan_file.getvalue(),
                plan_file.name,
            )
        except Exception as error:
            st.error(str(error))
            history_raw = None
            plan_raw = None

        if history_raw is not None and plan_raw is not None:
            preview1, preview2 = st.columns(2)
            with preview1:
                st.markdown("#### Geçmiş satış önizleme")
                st.dataframe(
                    history_raw.head(30),
                    use_container_width=True,
                    hide_index=True,
                )
            with preview2:
                st.markdown("#### Gelecek plan önizleme")
                st.dataframe(
                    plan_raw.head(30),
                    use_container_width=True,
                    hide_index=True,
                )

            history_columns = (
                history_raw.columns.astype(str).tolist()
            )
            plan_columns = (
                plan_raw.columns.astype(str).tolist()
            )

            with st.form("mapping_form"):
                st.markdown("### Geçmiş satış sütunları")
                col1, col2, col3, col4 = st.columns(4)
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
                    "Satış adedi",
                    history_columns,
                    "sales",
                    optional=False,
                    key="history_sales",
                )

                col1, col2, col3, col4 = st.columns(4)
                history_stock = select_column(
                    "Geçmiş stok",
                    history_columns,
                    "stock",
                    optional=True,
                    key="history_stock",
                )
                history_price = select_column(
                    "Birim fiyat",
                    history_columns,
                    "price",
                    optional=True,
                    key="history_price",
                )
                history_stockout = select_column(
                    "Stokout bayrağı",
                    history_columns,
                    "stockout",
                    optional=True,
                    key="history_stockout",
                )
                history_category_1 = select_column(
                    "Ana kategori",
                    history_columns,
                    "category_1",
                    optional=True,
                    key="history_category_1",
                )

                with st.expander(
                    "Geçmiş veri gelişmiş ayarları"
                ):
                    col1, col2 = st.columns(2)
                    history_category_2 = select_column(
                        "Alt kategori",
                        history_columns,
                        "category_2",
                        optional=True,
                        key="history_category_2",
                    )
                    history_category_3 = select_column(
                        "Ürün grubu",
                        history_columns,
                        "category_3",
                        optional=True,
                        key="history_category_3",
                    )

                    col1, col2, col3 = st.columns(3)
                    frequency = col1.selectbox(
                        "Veri frekansı",
                        ["daily", "hourly", "monthly"],
                        format_func={
                            "daily": "Günlük",
                            "hourly": "Saatlik",
                            "monthly": "Aylık",
                        }.get,
                    )
                    duplicate_policy = col2.selectbox(
                        "Tekrarlı kayıtlar",
                        ["aggregate", "error"],
                        format_func={
                            "aggregate": "Birleştir",
                            "error": "Hata ver",
                        }.get,
                    )
                    date_gap_policy = col3.selectbox(
                        "Eksik tarihler",
                        ["warn", "error", "ignore"],
                        format_func={
                            "warn": "Uyar",
                            "error": "Hata ver",
                            "ignore": "Yoksay",
                        }.get,
                    )

                    col1, col2 = st.columns(2)
                    dayfirst = col1.checkbox(
                        "Tarih biçimi gün/ay/yıl",
                        value=False,
                    )
                    imputation_min_history = (
                        col2.number_input(
                            "Talep düzeltme minimum geçmişi",
                            min_value=1,
                            max_value=100,
                            value=7,
                        )
                    )

                st.markdown("### Gelecek dağıtım planı sütunları")
                col1, col2, col3 = st.columns(3)
                plan_date = select_column(
                    "Plan tarihi",
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

                col1, col2, col3 = st.columns(3)
                plan_starting_stock = select_column(
                    "Başlangıç stoğu",
                    plan_columns,
                    "starting_stock",
                    optional=False,
                    key="plan_starting_stock",
                )
                plan_shipment = select_column(
                    "Planlanan sevkiyat",
                    plan_columns,
                    "planned_shipment",
                    optional=False,
                    key="plan_shipment",
                )
                plan_price = select_column(
                    "Plan dönemi fiyatı",
                    plan_columns,
                    "price",
                    optional=True,
                    key="plan_price",
                )

                submit_data = st.form_submit_button(
                    "Verileri hazırla",
                    type="primary",
                    use_container_width=True,
                )

            if submit_data:
                try:
                    (
                        prepared_history,
                        pipeline,
                        history_has_stock,
                    ) = build_pipeline(
                        working_history_df=history_raw,
                        date_column=str(history_date),
                        store_column=str(history_store),
                        product_column=str(
                            history_product
                        ),
                        sales_column=str(history_sales),
                        stock_column=history_stock,
                        price_column=history_price,
                        stockout_column=(
                            history_stockout
                        ),
                        category_1=(
                            history_category_1
                        ),
                        category_2=(
                            history_category_2
                        ),
                        category_3=(
                            history_category_3
                        ),
                        frequency=frequency,
                        dayfirst=dayfirst,
                        duplicate_policy=(
                            duplicate_policy
                        ),
                        date_gap_policy=(
                            date_gap_policy
                        ),
                        min_history=int(
                            imputation_min_history
                        ),
                    )

                    analysis_history = (
                        pipeline.impute_stockouts(
                            prepared_history
                        )
                    )

                    plan_mapping = (
                        DistributionPlanMapping(
                            date=str(plan_date),
                            store=str(plan_store),
                            product=str(plan_product),
                            starting_stock=str(
                                plan_starting_stock
                            ),
                            planned_shipment=str(
                                plan_shipment
                            ),
                            price=plan_price,
                        )
                    )

                    future_plan = prepare_distribution_plan(
                        plan_raw,
                        mapping=plan_mapping,
                        prepared_history_df=(
                            prepared_history
                        ),
                        pandas_frequency=(
                            pipeline.pandas_freq
                        ),
                    )

                    save_prepared_bundle(
                        history_raw_df=history_raw,
                        plan_raw_df=plan_raw,
                        pipeline=pipeline,
                        prepared_history_df=(
                            prepared_history
                        ),
                        analysis_history_df=(
                            analysis_history
                        ),
                        future_plan_df=future_plan,
                        history_has_price=(
                            history_price is not None
                            or "price"
                            in future_plan.columns
                        ),
                        history_has_stock=(
                            history_has_stock
                        ),
                        label=(
                            f"{history_file.name} + "
                            f"{plan_file.name}"
                        ),
                    )
                    st.success(
                        "Geçmiş veri ve gelecek dağıtım planı hazırlandı."
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

    render_data_overview()

with forecast_tab:
    if (
        st.session_state.prepared_history_df is None
        or st.session_state.future_plan_df is None
    ):
        st.info(
            "Önce Veri ve Dağıtım Planı sekmesinde iki dosyayı hazırlayın."
        )
    else:
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

        st.subheader("Tahmin ayarları")
        st.info(
            f"Gelecek tahmin ufku dağıtım planından otomatik alındı: "
            f"{plan_horizon} dönem, {plan_series_count} mağaza–ürün serisi."
        )

        available_models = {
            "chronos_bolt": "Chronos Bolt Small",
            "chronos_2": "Chronos 2",
        }
        if (
            importlib.util.find_spec("timesfm")
            is not None
        ):
            available_models[
                "timesfm_2_5"
            ] = "TimesFM 2.5"

        selected_models = st.multiselect(
            "Zero-shot modeller",
            options=list(available_models),
            default=["chronos_bolt"],
            format_func=available_models.get,
        )

        col1, col2, col3 = st.columns(3)
        backtest_horizon = int(
            col1.number_input(
                "Backtest dönemi",
                min_value=1,
                max_value=max(
                    1,
                    int(
                        st.session_state[
                            "prepared_history_df"
                        ]
                        .groupby("series_id")
                        .size()
                        .min()
                        - 2
                    ),
                ),
                value=min(
                    int(
                        defaults[
                            "backtest_horizon"
                        ]
                    ),
                    plan_horizon,
                ),
            )
        )
        min_context = int(
            col2.number_input(
                "Minimum geçmiş",
                min_value=2,
                max_value=5000,
                value=int(
                    defaults["min_context"]
                ),
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

        if st.button(
            "Tahmin ve plan analizini çalıştır",
            type="primary",
            disabled=not selected_models,
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Backtest, plan tarihleri için tahmin ve stok dağıtım analizi çalışıyor..."
                ):
                    run_forecast(
                        model_keys=selected_models,
                        backtest_horizon=(
                            backtest_horizon
                        ),
                        min_context=min_context,
                    )
                    run_plan_analysis(
                        safety_periods=(
                            safety_periods
                        )
                    )
                st.success(
                    "Tahmin ve dağıtım planı analizi tamamlandı."
                )
            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )
                with st.expander("Teknik hata"):
                    st.code(traceback.format_exc())

        render_forecast_and_stock_view()

with dashboard_tab:
    render_dashboard()
