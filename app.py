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
    build_abc_management_summary,
    build_future_demand_plan,
    build_historical_abc_analysis,
    build_historical_loss_analysis,
    build_management_kpis,
)


st.set_page_config(
    page_title="Demand Planning AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = APP_DIR / "ornek_talep_verisi.csv"
NONE_OPTION = "— Yok —"

FREQUENCY_DEFAULTS = {
    "hourly": {"horizon": 24, "min_context": 168},
    "daily": {"horizon": 7, "min_context": 60},
    "monthly": {"horizon": 3, "min_context": 24},
}

COLUMN_ALIASES = {
    "date": [
        "Tarih", "Date", "date", "dt", "timestamp", "Datetime",
    ],
    "store": [
        "Magaza_ID", "Mağaza_ID", "Store_ID", "store_id",
        "Store ID", "shop_id",
    ],
    "product": [
        "Urun_ID", "Ürün_ID", "Product_ID", "product_id",
        "Product ID", "sku_id", "SKU",
    ],
    "sales": [
        "Satis_Adedi", "Satış_Adedi", "Units_Sold", "Units Sold",
        "sales", "sale_amount", "quantity",
    ],
    "stock": [
        "Stok_Miktari", "Stok_Miktarı", "Inventory_Level",
        "Inventory Level", "stock", "inventory", "current_stock",
    ],
    "price": [
        "Birim_Fiyat", "Birim Fiyat", "Price", "price",
        "unit_price", "selling_price",
    ],
    "stockout": [
        "Stokout_Flag", "Stockout_Flag", "stockout_flag",
        "is_stockout", "OOS_Flag",
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
}

DISPLAY_NAMES = {
    "date": "Tarih",
    "series_id": "Seri",
    "store_id": "Mağaza",
    "product_id": "Ürün",
    "category_1": "Ana Kategori",
    "category_2": "Alt Kategori",
    "category_3": "Ürün Grubu",
    "sales": "Gerçekleşen Satış",
    "demand_adjusted": "Düzeltilmiş Talep",
    "stock": "Dönem Sonu Stok",
    "price": "Birim Fiyat (TL)",
    "is_stockout": "Stokout",
    "predictions": "Tahmini Satış",
    "actual": "Gerçek Satış",
    "current_stock": "Güncel Stok",
    "cumulative_demand": "Kümülatif Tahmin",
    "projected_stock": "Tahmini Kalan Stok",
    "period_shortage": "Tahmini Açık",
    "stockout_risk": "Stokout Riski",
    "forecast_start": "Tahmin Başlangıcı",
    "forecast_end": "Tahmin Bitişi",
    "planned_demand": "Toplam Tahmini Talep",
    "average_period_demand": "Ortalama Dönem Talebi",
    "peak_period_demand": "En Yüksek Dönem Talebi",
    "expected_ending_stock": "Beklenen Dönem Sonu Stok",
    "expected_shortage_no_action": "Aksiyon Alınmazsa Açık",
    "safety_stock": "Güvenlik Stoğu",
    "target_stock_for_horizon": "Hedef Stok",
    "recommended_replenishment": "Önerilen İkmal",
    "expected_stockout_date": "Tahmini Stok Bitiş Tarihi",
    "latest_price": "Güncel Fiyat (TL)",
    "planned_revenue": "Planlanan Ciro (TL)",
    "recommended_replenishment_value": "İkmal Değeri (TL)",
    "expected_lost_revenue_no_action": "Aksiyon Alınmazsa Ciro Riski (TL)",
    "stockout_rate_pct": "Tarihsel Stokout Oranı (%)",
    "estimated_lost_demand": "Tahmini Kayıp Talep",
    "estimated_lost_revenue": "Tahmini Kayıp Ciro (TL)",
    "lost_demand_share_pct": "Kayıp Talep Payı (%)",
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
    "scored_rows": "Değerlendirilen Satır",
    "excluded_stockout_rows": "Hariç Tutulan Stokout",
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
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 14px 16px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            overflow: hidden;
        }
        .dashboard-note {
            padding: 12px 14px;
            border-left: 4px solid #64748b;
            background: rgba(241, 245, 249, 0.75);
            border-radius: 8px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "raw_df": None,
        "data_pipeline": None,
        "prepared_df": None,
        "analysis_df": None,
        "current_stock_df": None,
        "stock_is_real": False,
        "has_price": False,
        "data_label": None,
        "metrics_df": None,
        "evaluations_df": None,
        "model_errors_df": None,
        "future_forecast_df": None,
        "historical_loss_detail_df": None,
        "historical_loss_summary_df": None,
        "future_demand_detail_df": None,
        "demand_plan_df": None,
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
        "historical_loss_detail_df",
        "historical_loss_summary_df",
        "future_demand_detail_df",
        "demand_plan_df",
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
    return "—" if value is None or pd.isna(value) else f"{format_number(value, 2)} TL"


def read_sample_bytes() -> bytes:
    return SAMPLE_DATA_PATH.read_bytes()


@st.cache_data(show_spinner=False)
def read_uploaded_table(file_bytes: bytes, file_name: str) -> pd.DataFrame:
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


def zip_tables(tables: dict[str, Optional[pd.DataFrame]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, table in tables.items():
            if table is not None and not table.empty:
                archive.writestr(filename, csv_bytes(table))
    return buffer.getvalue()


def normalise_name(value: str) -> str:
    translation = str.maketrans(
        {
            "ı": "i", "İ": "i", "ş": "s", "Ş": "s",
            "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
            "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
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


def guess_column(columns: list[str], field: str) -> Optional[str]:
    normalised_columns = {
        normalise_name(column): column
        for column in columns
    }
    for alias in COLUMN_ALIASES[field]:
        result = normalised_columns.get(normalise_name(alias))
        if result is not None:
            return result
    return None


def select_column(
    label: str,
    columns: list[str],
    field: str,
    *,
    optional: bool,
    key: str,
) -> Optional[str]:
    options = ([NONE_OPTION] if optional else []) + columns
    guessed = guess_column(columns, field)
    default_value = guessed if guessed in options else options[0]
    index = options.index(default_value)

    selected = st.selectbox(
        label,
        options,
        index=index,
        key=key,
    )
    if optional and selected == NONE_OPTION:
        return None
    return selected


def derive_current_stock(
    prepared_df: pd.DataFrame,
    stock_timing: str,
) -> pd.DataFrame:
    latest = (
        prepared_df.sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[
            [
                "series_id",
                "store_id",
                "product_id",
                "stock",
                "sales",
            ]
        ]
        .copy()
    )

    if stock_timing == "start_of_period":
        latest["current_stock"] = (
            pd.to_numeric(latest["stock"], errors="coerce")
            - pd.to_numeric(latest["sales"], errors="coerce")
        ).clip(lower=0)
    else:
        latest["current_stock"] = pd.to_numeric(
            latest["stock"],
            errors="coerce",
        ).clip(lower=0)

    return latest[
        ["series_id", "store_id", "product_id", "current_stock"]
    ].reset_index(drop=True)


def prepare_display_table(
    df: pd.DataFrame,
    columns: list[str],
    *,
    rows: Optional[int] = None,
) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    result = df[available].copy()

    for column in result.columns:
        if "date" in column.lower() or column in {
            "forecast_start", "forecast_end",
        }:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).dt.strftime("%d.%m.%Y")
        elif pd.api.types.is_bool_dtype(result[column]) or str(
            result[column].dtype
        ) == "boolean":
            result[column] = result[column].map(
                {True: "Evet", False: "Hayır"}
            ).fillna("—")
        elif pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].round(2)

    result = result.rename(columns=DISPLAY_NAMES)
    return result.head(rows) if rows is not None else result


def store_data_bundle(
    *,
    raw_df: pd.DataFrame,
    pipeline: DemandDataPipeline,
    prepared_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    current_stock_df: Optional[pd.DataFrame],
    stock_is_real: bool,
    has_price: bool,
    label: str,
) -> None:
    st.session_state.raw_df = raw_df
    st.session_state.data_pipeline = pipeline
    st.session_state.prepared_df = prepared_df
    st.session_state.analysis_df = analysis_df
    st.session_state.current_stock_df = current_stock_df
    st.session_state.stock_is_real = stock_is_real
    st.session_state.has_price = has_price
    st.session_state.data_label = label
    reset_outputs()


def run_forecast(
    model_keys: list[str],
    horizon: int,
    min_context: int,
    max_series: int,
) -> None:
    prepared_df = st.session_state.prepared_df
    pipeline = st.session_state.data_pipeline

    if prepared_df is None or pipeline is None:
        raise ValueError("Önce veri setini hazırlayın.")

    if len(model_keys) == 1:
        model_key = model_keys[0]
        forecaster = None
        mvp = None
        try:
            forecaster = create_forecaster(
                model_key,
                freq=pipeline.pandas_freq,
            )
            mvp = DemandForecastMVP(pipeline, forecaster)

            evaluations_df, metrics = mvp.backtest(
                prepared_df=prepared_df,
                horizon=horizon,
                min_context=min_context,
                max_series=max_series,
            )
            metrics.update(
                {
                    "model_key": model_key,
                    "model_name": MODEL_CONFIGS[model_key]["display_name"],
                }
            )
            metrics_df = pd.DataFrame([metrics])
            evaluations_df["model_key"] = model_key
            evaluations_df["model_name"] = metrics["model_name"]
            model_errors_df = pd.DataFrame()

            future_forecast_df = mvp.forecast(
                prepared_df=prepared_df,
                horizon=horizon,
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
        metrics_df, evaluations_df, model_errors_df = compare_zero_shot_models(
            prepared_df=prepared_df,
            data_pipeline=pipeline,
            model_keys=model_keys,
            horizon=horizon,
            min_context=min_context,
            max_series=max_series,
        )

        if metrics_df.empty:
            raise RuntimeError("Seçilen modeller başarıyla tamamlanamadı.")

        metrics_df = metrics_df.sort_values(
            ["wmape_pct", "mae"],
            ascending=True,
        ).reset_index(drop=True)

        best_model_key = str(metrics_df.iloc[0]["model_key"])
        forecaster = None
        mvp = None
        try:
            forecaster = create_forecaster(
                best_model_key,
                freq=pipeline.pandas_freq,
            )
            mvp = DemandForecastMVP(pipeline, forecaster)
            future_forecast_df = mvp.forecast(
                prepared_df=prepared_df,
                horizon=horizon,
                min_context=min_context,
                max_series=max_series,
            )
        finally:
            if mvp is not None:
                del mvp
            if forecaster is not None:
                del forecaster
            gc.collect()

    future_forecast_df["predictions"] = pd.to_numeric(
        future_forecast_df["predictions"],
        errors="coerce",
    ).clip(lower=0)

    st.session_state.metrics_df = metrics_df
    st.session_state.evaluations_df = evaluations_df
    st.session_state.model_errors_df = model_errors_df
    st.session_state.future_forecast_df = future_forecast_df


def run_analysis(safety_periods: float) -> None:
    analysis_df = st.session_state.analysis_df
    prepared_df = st.session_state.prepared_df
    forecast_df = st.session_state.future_forecast_df
    pipeline = st.session_state.data_pipeline

    if any(value is None for value in (
        analysis_df,
        prepared_df,
        forecast_df,
        pipeline,
    )):
        raise ValueError("Analiz için veri ve tahmin gereklidir.")

    historical_detail, historical_summary = build_historical_loss_analysis(
        analysis_df
    )

    forecast_input = forecast_df[
        ["series_id", "date", "predictions"]
    ].copy()

    future_detail, demand_plan = build_future_demand_plan(
        future_forecast_df=forecast_input,
        prepared_df=prepared_df,
        historical_loss_summary_df=historical_summary,
        current_stock_df=st.session_state.current_stock_df,
        stock_is_real=st.session_state.stock_is_real,
        stock_timing=pipeline.stock_timing,
        safety_periods=safety_periods,
    )

    has_revenue = (
        st.session_state.has_price
        and "estimated_lost_revenue" in historical_summary.columns
        and "latest_price" in demand_plan.columns
    )

    if has_revenue:
        demand_plan = apply_revenue_weighted_priority(demand_plan)
        abc_product = build_historical_abc_analysis(historical_summary)
        demand_plan = add_abc_to_demand_plan(
            demand_plan,
            abc_product,
        )
        demand_plan = add_abc_stockout_action(demand_plan)
        abc_summary = build_abc_management_summary(demand_plan)
    else:
        abc_product = None
        abc_summary = None

    st.session_state.historical_loss_detail_df = historical_detail
    st.session_state.historical_loss_summary_df = historical_summary
    st.session_state.future_demand_detail_df = future_detail
    st.session_state.demand_plan_df = demand_plan
    st.session_state.management_kpis_df = build_management_kpis(
        historical_summary,
        demand_plan,
    )
    st.session_state.abc_product_df = abc_product
    st.session_state.abc_summary_df = abc_summary


def series_selector(key_prefix: str) -> Optional[str]:
    prepared_df = st.session_state.prepared_df
    if prepared_df is None:
        return None

    metadata = (
        prepared_df[["series_id", "store_id", "product_id"]]
        .drop_duplicates()
        .copy()
    )
    stores = sorted(metadata["store_id"].astype(str).unique())
    selected_store = st.selectbox(
        "Mağaza",
        stores,
        key=f"{key_prefix}_store",
    )

    products = sorted(
        metadata.loc[
            metadata["store_id"].astype(str).eq(selected_store),
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
        metadata["store_id"].astype(str).eq(selected_store)
        & metadata["product_id"].astype(str).eq(selected_product),
        "series_id",
    ]

    return None if matched.empty else str(matched.iloc[0])


def render_data_summary() -> None:
    prepared_df = st.session_state.prepared_df
    if prepared_df is None:
        return

    st.success(f"Veri hazır: {st.session_state.data_label}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Satır", format_number(len(prepared_df)))
    col2.metric("Mağaza", format_number(prepared_df["store_id"].nunique()))
    col3.metric("Ürün", format_number(prepared_df["product_id"].nunique()))
    col4.metric("Seri", format_number(prepared_df["series_id"].nunique()))
    col5.metric(
        "Stokout Oranı",
        f"%{format_number(prepared_df['is_stockout'].mean() * 100, 2)}",
    )

    preview_columns = [
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
    ]
    st.dataframe(
        prepare_display_table(prepared_df, preview_columns, rows=100),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Veri doğrulama raporu"):
        report = st.session_state.data_pipeline.report
        if report.get("warnings"):
            for warning in report["warnings"]:
                st.warning(str(warning))
        else:
            st.info("Kritik veri uyarısı bulunmuyor.")
        st.json(report.get("stats", {}))


def render_forecast_dashboard() -> None:
    metrics_df = st.session_state.metrics_df
    forecast_df = st.session_state.future_forecast_df
    evaluations_df = st.session_state.evaluations_df
    future_detail = st.session_state.future_demand_detail_df
    prepared_df = st.session_state.prepared_df
    analysis_df = st.session_state.analysis_df

    if metrics_df is None or forecast_df is None:
        return

    best = metrics_df.sort_values("wmape_pct").iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Seçilen Model", str(best.get("model_name", "—")))
    col2.metric("WMAPE", f"%{format_number(best.get('wmape_pct'), 2)}")
    col3.metric("MAE", format_number(best.get("mae"), 2))
    col4.metric("Bias", f"%{format_number(best.get('bias_pct'), 2)}")

    st.dataframe(
        prepare_display_table(
            metrics_df,
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

    st.markdown("### Mağaza–ürün tahmin ve stok görünümü")
    st.info(
        "Gelecek tahminleri yüklediğiniz veri setindeki satış geçmişinden "
        "üretilir. Stok sütunu varsa aynı veri setindeki son stok seviyesi "
        "kullanılarak gelecekteki kalan stok hesaplanır."
    )

    selected_series = series_selector("forecast")
    if selected_series is None:
        return

    history = (
        analysis_df.loc[
            analysis_df["series_id"].astype(str).eq(selected_series)
        ]
        .sort_values("date")
        .tail(70)
        .copy()
    )
    future = (
        forecast_df.loc[
            forecast_df["series_id"].astype(str).eq(selected_series)
        ]
        .sort_values("date")
        .copy()
    )

    sales_figure = go.Figure()
    sales_figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["sales"],
            name="Gerçekleşen satış",
            mode="lines",
        )
    )
    if "demand_adjusted" in history.columns:
        sales_figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["demand_adjusted"],
                name="Stokout düzeltilmiş talep",
                mode="lines",
                line={"dash": "dot"},
            )
        )
    sales_figure.add_trace(
        go.Scatter(
            x=future["date"],
            y=future["predictions"],
            name="Gelecek tahmini",
            mode="lines+markers",
        )
    )
    stockout_history = history.loc[history["is_stockout"].astype(bool)]
    if not stockout_history.empty:
        sales_figure.add_trace(
            go.Scatter(
                x=stockout_history["date"],
                y=stockout_history["sales"],
                name="Stokout günü",
                mode="markers",
                marker={"symbol": "x", "size": 9},
            )
        )
    sales_figure.update_layout(
        title="Satış geçmişi ve gelecek talep tahmini",
        xaxis_title="Tarih",
        yaxis_title="Adet",
        hovermode="x unified",
        legend_title_text="Gösterge",
    )
    st.plotly_chart(sales_figure, use_container_width=True)
    st.caption(
        "Amaç: Modelin geçmiş satış eğrisini nasıl devam ettirdiğini ve "
        "stokout günlerinde düzeltilmiş talebin satıştan ne kadar ayrıldığını gösterir."
    )

    if future_detail is not None:
        stock_future = (
            future_detail.loc[
                future_detail["series_id"].astype(str).eq(selected_series)
            ]
            .sort_values("date")
            .copy()
        )
    else:
        stock_future = pd.DataFrame()

    stock_history = (
        prepared_df.loc[
            prepared_df["series_id"].astype(str).eq(selected_series)
        ]
        .sort_values("date")
        .tail(70)
    )

    stock_figure = go.Figure()
    stock_figure.add_trace(
        go.Scatter(
            x=stock_history["date"],
            y=stock_history["stock"],
            name="Geçmiş stok",
            mode="lines",
        )
    )

    if (
        not stock_future.empty
        and stock_future["current_stock"].notna().any()
    ):
        stock_figure.add_trace(
            go.Scatter(
                x=stock_future["date"],
                y=stock_future["projected_stock"],
                name="Tahmini kalan stok",
                mode="lines+markers",
                fill="tozeroy",
            )
        )
        stock_figure.add_hline(
            y=0,
            line_dash="dash",
            annotation_text="Stokout sınırı",
        )
    stock_figure.update_layout(
        title="Geçmiş stok ve tahmin dönemindeki stok seyri",
        xaxis_title="Tarih",
        yaxis_title="Stok adedi",
        hovermode="x unified",
        legend_title_text="Gösterge",
    )
    st.plotly_chart(stock_figure, use_container_width=True)
    st.caption(
        "Amaç: Mevcut stoktan kümülatif tahmin düşülerek stokun hangi tarihte "
        "tükenebileceğini gösterir."
    )

    if not stock_future.empty:
        forecast_table = prepare_display_table(
            stock_future,
            [
                "date",
                "store_id",
                "product_id",
                "predictions",
                "current_stock",
                "cumulative_demand",
                "projected_stock",
                "period_shortage",
                "stockout_risk",
            ],
        )
    else:
        forecast_table = prepare_display_table(
            future,
            ["date", "store_id", "product_id", "predictions"],
        )

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True,
    )


def build_priority_donut(demand_plan: pd.DataFrame) -> go.Figure:
    counts = (
        demand_plan["priority"]
        .fillna("Belirsiz")
        .value_counts()
        .rename_axis("Risk Seviyesi")
        .reset_index(name="Mağaza–Ürün Sayısı")
    )
    return px.pie(
        counts,
        names="Risk Seviyesi",
        values="Mağaza–Ürün Sayısı",
        hole=0.55,
        title="Risk seviyelerinin dağılımı",
    )


def build_risk_bubble(demand_plan: pd.DataFrame) -> go.Figure:
    y_column = (
        "expected_lost_revenue_no_action"
        if "expected_lost_revenue_no_action" in demand_plan.columns
        else "expected_shortage_no_action"
    )
    size_column = (
        "recommended_replenishment"
        if "recommended_replenishment" in demand_plan.columns
        else "planned_demand"
    )
    color_column = (
        "abc_class"
        if "abc_class" in demand_plan.columns
        else "priority"
    )

    plot_df = demand_plan.copy()
    plot_df["Konum"] = (
        plot_df["store_id"].astype(str)
        + " / "
        + plot_df["product_id"].astype(str)
    )

    figure = px.scatter(
        plot_df,
        x="stockout_rate_pct",
        y=y_column,
        size=size_column,
        color=color_column,
        hover_name="Konum",
        hover_data={
            "planned_demand": ":.1f",
            "current_stock": ":.1f",
            "recommended_replenishment": ":.1f",
        },
        labels={
            "stockout_rate_pct": "Tarihsel stokout oranı (%)",
            y_column: DISPLAY_NAMES.get(y_column, y_column),
            size_column: DISPLAY_NAMES.get(size_column, size_column),
            color_column: DISPLAY_NAMES.get(color_column, color_column),
        },
        title="Stokout sıklığı ve iş etkisi risk matrisi",
    )
    return figure


def build_top_replenishment_bar(demand_plan: pd.DataFrame) -> go.Figure:
    top = (
        demand_plan.nlargest(15, "recommended_replenishment")
        .copy()
    )
    top["Konum"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values("recommended_replenishment")
    return px.bar(
        top,
        x="recommended_replenishment",
        y="Konum",
        orientation="h",
        color="priority",
        labels={
            "recommended_replenishment": "Önerilen ikmal (adet)",
            "priority": "Risk seviyesi",
        },
        title="En yüksek ikmal ihtiyacı bulunan mağaza–ürünler",
    )


def build_replenishment_heatmap(demand_plan: pd.DataFrame) -> go.Figure:
    store_totals = (
        demand_plan.groupby("store_id")["recommended_replenishment"]
        .sum()
        .nlargest(10)
        .index
    )
    product_totals = (
        demand_plan.groupby("product_id")["recommended_replenishment"]
        .sum()
        .nlargest(15)
        .index
    )
    filtered = demand_plan.loc[
        demand_plan["store_id"].isin(store_totals)
        & demand_plan["product_id"].isin(product_totals)
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
            "color": "İkmal adedi",
        },
        title="İkmal ihtiyacının mağaza ve ürünlere dağılımı",
    )


def build_revenue_overview(
    history: pd.DataFrame,
    demand_plan: pd.DataFrame,
) -> go.Figure:
    observed = float(history.get("observed_revenue", pd.Series(dtype=float)).sum())
    historical_loss = float(
        history.get("estimated_lost_revenue", pd.Series(dtype=float)).sum()
    )
    planned = float(
        demand_plan.get("planned_revenue", pd.Series(dtype=float)).sum()
    )
    future_risk = float(
        demand_plan.get(
            "expected_lost_revenue_no_action",
            pd.Series(dtype=float),
        ).sum()
    )

    chart_df = pd.DataFrame(
        {
            "Dönem": [
                "Tarihsel",
                "Tarihsel",
                "Gelecek",
                "Gelecek",
            ],
            "Bileşen": [
                "Gerçekleşen ciro",
                "Stokout kaynaklı kayıp",
                "Planlanan ciro",
                "Aksiyon alınmazsa risk",
            ],
            "Tutar": [
                observed,
                historical_loss,
                planned,
                future_risk,
            ],
        }
    )
    return px.bar(
        chart_df,
        x="Dönem",
        y="Tutar",
        color="Bileşen",
        barmode="group",
        labels={"Tutar": "Tutar (TL)"},
        title="Tarihsel ciro kaybı ve gelecek dönem ciro riski",
    )


def build_top_revenue_risk_bar(demand_plan: pd.DataFrame) -> go.Figure:
    top = (
        demand_plan.nlargest(
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
    top = top.sort_values("expected_lost_revenue_no_action")
    return px.bar(
        top,
        x="expected_lost_revenue_no_action",
        y="Konum",
        orientation="h",
        color="abc_class" if "abc_class" in top.columns else "priority",
        labels={
            "expected_lost_revenue_no_action": "Ciro riski (TL)",
            "abc_class": "ABC sınıfı",
            "priority": "Risk seviyesi",
        },
        title="Aksiyon alınmazsa en yüksek ciro riski",
    )


def build_abc_treemap(abc_product: pd.DataFrame) -> go.Figure:
    plot_df = abc_product.copy()
    plot_df["Ürün"] = plot_df["product_id"].astype(str)
    return px.treemap(
        plot_df,
        path=["abc_class", "Ürün"],
        values="total_value",
        color="abc_class",
        title="Ürün portföyünün ABC sınıflarına göre ticari değeri",
        labels={
            "abc_class": "ABC sınıfı",
            "total_value": "Ticari değer (TL)",
        },
    )


def build_abc_donut(abc_product: pd.DataFrame) -> go.Figure:
    summary = (
        abc_product.groupby("abc_class", as_index=False)
        .agg(
            total_value=("total_value", "sum"),
            product_count=("product_id", "nunique"),
        )
    )
    return px.pie(
        summary,
        names="abc_class",
        values="total_value",
        hole=0.55,
        title="ABC sınıflarının toplam ticari değer payı",
    )


def render_decision_dashboard() -> None:
    demand_plan = st.session_state.demand_plan_df
    history = st.session_state.historical_loss_summary_df
    abc_product = st.session_state.abc_product_df
    abc_summary = st.session_state.abc_summary_df

    if demand_plan is None or history is None:
        st.info("Tahmin sekmesinden tahmin ve analizleri çalıştırın.")
        return

    total_future = demand_plan["planned_demand"].sum()
    total_replenishment = demand_plan[
        "recommended_replenishment"
    ].sum()
    risky_count = (
        demand_plan["stockout_risk"]
        .astype("boolean")
        .fillna(False)
        .sum()
    )
    lost_demand = history["estimated_lost_demand"].sum()
    future_revenue_risk = (
        demand_plan.get(
            "expected_lost_revenue_no_action",
            pd.Series(dtype=float),
        ).sum()
    )
    historical_revenue_loss = (
        history.get(
            "estimated_lost_revenue",
            pd.Series(dtype=float),
        ).sum()
    )

    row1 = st.columns(6)
    row1[0].metric("Tahmin Ufku Talebi", format_number(total_future))
    row1[1].metric("Önerilen İkmal", format_number(total_replenishment))
    row1[2].metric("Riskli Mağaza–Ürün", format_number(risky_count))
    row1[3].metric("Tarihsel Kayıp Talep", format_number(lost_demand))
    row1[4].metric("Tarihsel Kayıp Ciro", format_try(historical_revenue_loss))
    row1[5].metric("Gelecek Ciro Riski", format_try(future_revenue_risk))

    tabs = st.tabs(
        [
            "Genel Bakış",
            "Stok ve İkmal",
            "Ciro ve Risk",
            "ABC Portföyü",
            "Detay ve İndirme",
        ]
    )

    with tabs[0]:
        st.markdown("#### Öncelikli aksiyon listesi")
        st.caption(
            "Amaç: Yönetimin ilk olarak hangi mağaza–ürünlere müdahale etmesi "
            "gerektiğini kısa ve anlaşılır biçimde sıralar."
        )
        action_columns = [
            "operational_priority",
            "abc_class",
            "store_id",
            "product_id",
            "priority",
            "current_stock",
            "planned_demand",
            "recommended_replenishment",
            "expected_stockout_date",
            "expected_lost_revenue_no_action",
            "recommended_action",
        ]
        st.dataframe(
            prepare_display_table(demand_plan, action_columns, rows=25),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                build_priority_donut(demand_plan),
                use_container_width=True,
            )
            st.caption(
                "Amaç: Portföyde kritik, yüksek, orta ve düşük riskli "
                "mağaza–ürünlerin oranını gösterir."
            )
        with col2:
            st.plotly_chart(
                build_risk_bubble(demand_plan),
                use_container_width=True,
            )
            st.caption(
                "Amaç: Sık stokout yaşayan ve parasal etkisi yüksek noktaları "
                "sağ üst bölgede görünür hâle getirir."
            )

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                build_top_replenishment_bar(demand_plan),
                use_container_width=True,
            )
            st.caption(
                "Amaç: En fazla ürün gönderilmesi gereken mağaza–ürünleri sıralar."
            )
        with col2:
            st.plotly_chart(
                build_replenishment_heatmap(demand_plan),
                use_container_width=True,
            )
            st.caption(
                "Amaç: İkmal yükünün belirli mağaza veya ürünlerde yoğunlaşıp "
                "yoğunlaşmadığını gösterir."
            )

        stock_columns = [
            "operational_priority",
            "store_id",
            "product_id",
            "current_stock",
            "planned_demand",
            "safety_stock",
            "target_stock_for_horizon",
            "expected_ending_stock",
            "recommended_replenishment",
            "expected_shortage_no_action",
            "expected_stockout_date",
            "stockout_risk",
        ]
        st.dataframe(
            prepare_display_table(demand_plan, stock_columns),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        if "expected_lost_revenue_no_action" not in demand_plan.columns:
            st.warning("Bu bölüm için fiyat sütunu gereklidir.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    build_revenue_overview(history, demand_plan),
                    use_container_width=True,
                )
                st.caption(
                    "Amaç: Tarihsel stokout kaybıyla gelecek dönemdeki ciro "
                    "riskini aynı yönetim görünümünde karşılaştırır."
                )
            with col2:
                st.plotly_chart(
                    build_top_revenue_risk_bar(demand_plan),
                    use_container_width=True,
                )
                st.caption(
                    "Amaç: Aksiyon alınmadığında en büyük parasal kaybı "
                    "oluşturabilecek mağaza–ürünleri sıralar."
                )

            revenue_columns = [
                "abc_class",
                "store_id",
                "product_id",
                "latest_price",
                "planned_demand",
                "planned_revenue",
                "recommended_replenishment",
                "recommended_replenishment_value",
                "estimated_lost_revenue",
                "expected_lost_revenue_no_action",
                "revenue_priority_score",
            ]
            st.dataframe(
                prepare_display_table(demand_plan, revenue_columns),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        if abc_product is None:
            st.warning("ABC analizi için fiyat bilgisi gereklidir.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    build_abc_treemap(abc_product),
                    use_container_width=True,
                )
                st.caption(
                    "Amaç: Ürünlerin ticari değerini büyüklük olarak, ABC "
                    "sınıfını ise portföy grubu olarak gösterir."
                )
            with col2:
                st.plotly_chart(
                    build_abc_donut(abc_product),
                    use_container_width=True,
                )
                st.caption(
                    "Amaç: A, B ve C ürünlerinin toplam ticari değerdeki "
                    "payını karşılaştırır."
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
        detail_choice = st.selectbox(
            "Gösterilecek detay",
            [
                "Talep ve ikmal planı",
                "Tarihsel stokout kayıpları",
                "Gelecek dönem stok seyri",
                "Model metrikleri",
            ],
        )

        detail_map = {
            "Talep ve ikmal planı": demand_plan,
            "Tarihsel stokout kayıpları": history,
            "Gelecek dönem stok seyri": st.session_state.future_demand_detail_df,
            "Model metrikleri": st.session_state.metrics_df,
        }
        selected_table = detail_map[detail_choice]
        st.dataframe(
            prepare_display_table(
                selected_table,
                list(selected_table.columns),
                rows=2000,
            ),
            use_container_width=True,
            hide_index=True,
        )

        downloads = {
            "model_metrikleri.csv": st.session_state.metrics_df,
            "gelecek_tahminleri.csv": st.session_state.future_forecast_df,
            "gelecek_stok_seyri.csv": st.session_state.future_demand_detail_df,
            "tarihsel_stokout_analizi.csv": history,
            "talep_ve_ikmal_plani.csv": demand_plan,
            "yonetici_kpi.csv": st.session_state.management_kpis_df,
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
    "Talep tahmini, stok görünürlüğü, ikmal planı, ciro riski ve ABC analizi"
)

with st.sidebar:
    st.header("Uygulama Akışı")
    st.write("1. Veri setini yükle")
    st.write("2. Tahmin ve analizleri çalıştır")
    st.write("3. Karar destek dashboardunu incele")
    st.divider()
    if st.button("Oturumu sıfırla", use_container_width=True):
        reset_everything()
        st.rerun()

data_tab, forecast_tab, dashboard_tab = st.tabs(
    [
        "1 · Veri Seti",
        "2 · Tahmin ve Stok",
        "3 · Karar Destek Dashboard",
    ]
)

with data_tab:
    st.subheader("Veri setini yükle")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "CSV, XLSX veya Parquet dosyası",
            type=["csv", "xlsx", "parquet"],
        )
    with col2:
        st.download_button(
            "Örnek veri setini indir",
            data=read_sample_bytes(),
            file_name="ornek_talep_verisi.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Örnek dosya satış, stok, fiyat, stokout ve kategori sütunları içerir."
        )

    if uploaded_file is not None:
        try:
            uploaded_df = read_uploaded_table(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
        except Exception as error:
            st.error(str(error))
            uploaded_df = None

        if uploaded_df is not None:
            st.markdown("#### Dosya önizleme")
            st.dataframe(
                uploaded_df.head(50),
                use_container_width=True,
                hide_index=True,
            )

            columns = uploaded_df.columns.astype(str).tolist()

            with st.form("data_mapping_form"):
                st.markdown("#### Sütunları eşleştir")

                col1, col2, col3, col4 = st.columns(4)
                date_column = select_column(
                    "Tarih",
                    columns,
                    "date",
                    optional=False,
                    key="map_date",
                )
                store_column = select_column(
                    "Mağaza ID",
                    columns,
                    "store",
                    optional=False,
                    key="map_store",
                )
                product_column = select_column(
                    "Ürün ID",
                    columns,
                    "product",
                    optional=False,
                    key="map_product",
                )
                sales_column = select_column(
                    "Satış adedi",
                    columns,
                    "sales",
                    optional=False,
                    key="map_sales",
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    no_stock = st.checkbox(
                        "Stok sütunum yok",
                        value=False,
                    )
                    stock_column = (
                        None
                        if no_stock
                        else select_column(
                            "Stok miktarı",
                            columns,
                            "stock",
                            optional=False,
                            key="map_stock",
                        )
                    )
                with col2:
                    price_column = select_column(
                        "Birim fiyat",
                        columns,
                        "price",
                        optional=True,
                        key="map_price",
                    )
                with col3:
                    stockout_column = select_column(
                        "Stokout bayrağı",
                        columns,
                        "stockout",
                        optional=True,
                        key="map_stockout",
                    )
                with col4:
                    category_1 = select_column(
                        "Ana kategori",
                        columns,
                        "category_1",
                        optional=True,
                        key="map_category_1",
                    )

                with st.expander("Opsiyonel alanlar ve gelişmiş ayarlar"):
                    col1, col2 = st.columns(2)
                    category_2 = select_column(
                        "Alt kategori",
                        columns,
                        "category_2",
                        optional=True,
                        key="map_category_2",
                    )
                    category_3 = select_column(
                        "Ürün grubu",
                        columns,
                        "category_3",
                        optional=True,
                        key="map_category_3",
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

                    col1, col2, col3 = st.columns(3)
                    dayfirst = col1.checkbox(
                        "Tarih biçimi gün/ay/yıl",
                        value=False,
                    )
                    stock_threshold = col2.number_input(
                        "Stokout stok eşiği",
                        value=0.0,
                        step=1.0,
                    )
                    imputation_min_history = col3.number_input(
                        "Talep düzeltme minimum geçmişi",
                        min_value=1,
                        max_value=100,
                        value=7,
                    )

                submit_data = st.form_submit_button(
                    "Veriyi hazırla",
                    type="primary",
                    use_container_width=True,
                )

            if submit_data:
                try:
                    if no_stock and stockout_column is None:
                        raise ValueError(
                            "Stok sütunu yoksa stokout bayrağı seçilmelidir."
                        )

                    working_df = uploaded_df.copy()
                    actual_stock_column = stock_column
                    if no_stock:
                        actual_stock_column = "__stock_proxy__"
                        working_df[actual_stock_column] = 1.0

                    mapping = ColumnMapping(
                        date=str(date_column),
                        store=str(store_column),
                        product=str(product_column),
                        sales=str(sales_column),
                        stock=str(actual_stock_column),
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
                        stockout_stock_threshold=float(stock_threshold),
                        combine_provided_flag_with_inferred=not no_stock,
                        stock_timing="end_of_period",
                        imputation_window=28,
                        min_history=int(imputation_min_history),
                        imputation_statistic="median",
                    )

                    prepared_df = pipeline.prepare(working_df)
                    analysis_df = pipeline.impute_stockouts(prepared_df)
                    current_stock_df = (
                        derive_current_stock(
                            prepared_df,
                            pipeline.stock_timing,
                        )
                        if not no_stock
                        else None
                    )

                    store_data_bundle(
                        raw_df=working_df,
                        pipeline=pipeline,
                        prepared_df=prepared_df,
                        analysis_df=analysis_df,
                        current_stock_df=current_stock_df,
                        stock_is_real=not no_stock,
                        has_price=price_column is not None,
                        label=uploaded_file.name,
                    )
                    st.success("Veri seti analize hazırlandı.")
                except (DataValidationError, ValueError) as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"{type(error).__name__}: {error}")
                    with st.expander("Teknik hata"):
                        st.code(traceback.format_exc())

    render_data_summary()

with forecast_tab:
    if st.session_state.prepared_df is None:
        st.info("Önce Veri Seti sekmesinden bir dosya yükleyin.")
    else:
        pipeline = st.session_state.data_pipeline
        defaults = FREQUENCY_DEFAULTS[pipeline.frequency]

        st.subheader("Tahmin ayarları")
        available_models = {
            "chronos_bolt": "Chronos Bolt Small",
            "chronos_2": "Chronos 2",
        }
        if importlib.util.find_spec("timesfm") is not None:
            available_models["timesfm_2_5"] = "TimesFM 2.5"

        selected_models = st.multiselect(
            "Zero-shot modeller",
            options=list(available_models),
            default=["chronos_bolt"],
            format_func=available_models.get,
        )

        col1, col2, col3, col4 = st.columns(4)
        horizon = int(
            col1.number_input(
                "Tahmin ufku",
                min_value=1,
                max_value=256,
                value=int(defaults["horizon"]),
            )
        )
        min_context = int(
            col2.number_input(
                "Minimum geçmiş",
                min_value=2,
                max_value=5000,
                value=int(defaults["min_context"]),
            )
        )
        maximum_series = int(
            st.session_state.prepared_df["series_id"].nunique()
        )
        max_series = int(
            col3.number_input(
                "Maksimum seri",
                min_value=1,
                max_value=maximum_series,
                value=min(100, maximum_series),
            )
        )
        safety_periods = float(
            col4.number_input(
                "Güvenlik stoğu dönemi",
                min_value=0.0,
                max_value=30.0,
                value=1.0,
                step=0.5,
            )
        )

        if not st.session_state.stock_is_real:
            st.warning(
                "Yüklenen dosyada gerçek stok bulunmadığı için stok seyri ve "
                "ikmal miktarı hesaplanamaz. Tahmin yine satış geçmişinizden üretilir."
            )

        if st.button(
            "Tahmin ve analizleri çalıştır",
            type="primary",
            disabled=not selected_models,
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Model karşılaştırması, gelecek tahmini ve stok analizleri çalışıyor..."
                ):
                    run_forecast(
                        selected_models,
                        horizon,
                        min_context,
                        max_series,
                    )
                    run_analysis(safety_periods)
                st.success("Tahmin ve analizler tamamlandı.")
            except Exception as error:
                st.error(f"{type(error).__name__}: {error}")
                with st.expander("Teknik hata"):
                    st.code(traceback.format_exc())

        render_forecast_dashboard()

with dashboard_tab:
    render_decision_dashboard()
