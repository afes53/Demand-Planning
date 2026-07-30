from __future__ import annotations

import gc
import importlib.util
import io
import traceback
import zipfile
from typing import Any, Optional

import numpy as np
import pandas as pd
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
    management_recommendations,
)

from freshretail_demo_preprocessing import (
    choose_demo_unit_multiplier,
    generate_demo_inventory,
    make_demo_current_stock,
    make_demo_prepared_df,
    prepare_freshretail_demo_commercial_data,
)


st.set_page_config(
    page_title="Demand Planning AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


FRESH_DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"
FRESH_COLUMNS = [
    "city_id",
    "store_id",
    "product_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
]

NONE_OPTION = "— Yok —"

FREQUENCY_DEFAULTS = {
    "hourly": {"horizon": 24, "min_context": 168},
    "daily": {"horizon": 7, "min_context": 60},
    "monthly": {"horizon": 3, "min_context": 24},
}


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "source_mode": "FreshRetailNet Demo",
        "raw_df": None,
        "data_pipeline": None,
        "prepared_df": None,
        "analysis_df": None,
        "current_stock_df": None,
        "stock_is_real": False,
        "has_price": False,
        "is_demo": False,
        "data_label": None,
        "metrics_df": None,
        "evaluations_df": None,
        "model_errors_df": None,
        "future_forecast_df": None,
        "forecast_settings": None,
        "historical_loss_detail_df": None,
        "historical_loss_summary_df": None,
        "future_demand_detail_df": None,
        "demand_plan_df": None,
        "management_kpis_df": None,
        "abc_product_df": None,
        "abc_summary_df": None,
        "recommendations_df": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_forecast_and_analysis() -> None:
    keys = [
        "metrics_df",
        "evaluations_df",
        "model_errors_df",
        "future_forecast_df",
        "forecast_settings",
        "historical_loss_detail_df",
        "historical_loss_summary_df",
        "future_demand_detail_df",
        "demand_plan_df",
        "management_kpis_df",
        "abc_product_df",
        "abc_summary_df",
        "recommendations_df",
    ]
    for key in keys:
        st.session_state[key] = None


def reset_everything() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialise_state()


def format_number(value: Any, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    formatted = f"{float(value):,.{decimals}f}"
    return (
        formatted.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def format_try(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{format_number(value, 2)} TL"


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def tables_zip_bytes(tables: dict[str, Optional[pd.DataFrame]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename, table in tables.items():
            if table is None or table.empty:
                continue
            archive.writestr(
                filename,
                table.to_csv(index=False).encode("utf-8-sig"),
            )
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def read_uploaded_table(
    file_bytes: bytes,
    file_name: str,
) -> pd.DataFrame:
    suffix = file_name.lower().rsplit(".", maxsplit=1)[-1]
    stream = io.BytesIO(file_bytes)

    if suffix == "csv":
        return pd.read_csv(stream, sep=None, engine="python")
    if suffix == "xlsx":
        return pd.read_excel(stream)
    if suffix == "parquet":
        return pd.read_parquet(stream)

    raise ValueError("Desteklenen dosyalar: CSV, XLSX ve Parquet.")


def _prepare_fresh_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["store_key"] = (
        result["city_id"].astype(str)
        + "||"
        + result["store_id"].astype(str)
    )
    result["temporary_series_id"] = (
        result["store_key"]
        + "||"
        + result["product_id"].astype(str)
    )
    result["dt"] = pd.to_datetime(
        result["dt"],
        errors="raise",
    )
    return result


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_freshretail_demo(
    n_series: int,
    train_periods: int,
    eval_periods: int,
) -> pd.DataFrame:
    from datasets import load_dataset

    eval_dataset = load_dataset(
        FRESH_DATASET_NAME,
        split="eval",
    ).select_columns(FRESH_COLUMNS)

    eval_df = _prepare_fresh_frame(
        eval_dataset.to_pandas()
    )

    eval_counts = (
        eval_df.groupby("temporary_series_id")
        .size()
    )
    eligible_eval_ids = set(
        eval_counts.loc[
            eval_counts >= eval_periods
        ].index.astype(str)
    )

    selected_ids: list[str] = []
    train_df: Optional[pd.DataFrame] = None

    for factor in (3, 5, 8, 12):
        candidate_rows = (
            n_series
            * train_periods
            * factor
        )

        train_dataset = load_dataset(
            FRESH_DATASET_NAME,
            split=f"train[:{candidate_rows}]",
        ).select_columns(FRESH_COLUMNS)

        candidate_train_df = _prepare_fresh_frame(
            train_dataset.to_pandas()
        )

        train_counts = (
            candidate_train_df
            .groupby("temporary_series_id")
            .size()
        )

        selected_ids = [
            series_id
            for series_id in train_counts.loc[
                train_counts >= train_periods
            ].index.astype(str)
            if series_id in eligible_eval_ids
        ][:n_series]

        if len(selected_ids) >= n_series:
            train_df = candidate_train_df
            break

    if train_df is None or len(selected_ids) < n_series:
        raise ValueError(
            f"{n_series} adet tam train/eval serisi bulunamadı. "
            "Seri sayısını azaltın veya train tarama miktarını artırın."
        )

    train_df = (
        train_df.loc[
            train_df["temporary_series_id"].isin(selected_ids)
        ]
        .sort_values(["temporary_series_id", "dt"])
        .groupby("temporary_series_id", group_keys=False)
        .head(train_periods)
        .reset_index(drop=True)
    )

    eval_df = (
        eval_df.loc[
            eval_df["temporary_series_id"].isin(selected_ids)
        ]
        .sort_values(["temporary_series_id", "dt"])
        .groupby("temporary_series_id", group_keys=False)
        .head(eval_periods)
        .reset_index(drop=True)
    )

    period_check = pd.DataFrame(
        {
            "train": (
                train_df.groupby("temporary_series_id").size()
            ),
            "eval": (
                eval_df.groupby("temporary_series_id").size()
            ),
        }
    )

    invalid = period_check.loc[
        period_check["train"].ne(train_periods)
        | period_check["eval"].ne(eval_periods)
    ]
    if not invalid.empty:
        raise ValueError(
            "Bazı FreshRetailNet serilerinin train/eval uzunluğu eksik."
        )

    for frame in (train_df, eval_df):
        frame["stockout_flag"] = (
            pd.to_numeric(
                frame["stock_hour6_22_cnt"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
        )
        frame["stock_quantity_proxy"] = 1.0

    train_df["dataset_split"] = "train"
    eval_df["dataset_split"] = "eval"

    raw_df = pd.concat(
        [train_df, eval_df],
        ignore_index=True,
    )

    return (
        raw_df.sort_values(
            ["temporary_series_id", "dt"]
        )
        .reset_index(drop=True)
    )


def build_fresh_pipeline() -> DemandDataPipeline:
    mapping = ColumnMapping(
        date="dt",
        store="store_key",
        product="product_id",
        sales="sale_amount",
        stock="stock_quantity_proxy",
        price=None,
        category_1="first_category_id",
        category_2="second_category_id",
        category_3="third_category_id",
        stockout_flag="stockout_flag",
    )

    return DemandDataPipeline(
        mapping=mapping,
        frequency="daily",
        dayfirst=False,
        duplicate_policy="error",
        date_gap_policy="warn",
        negative_value_policy="clip",
        use_sales_equals_stock_rule=False,
        stockout_tolerance=0.0,
        stockout_stock_threshold=0.0,
        combine_provided_flag_with_inferred=False,
        stock_timing="end_of_period",
        imputation_window=28,
        min_history=7,
        imputation_statistic="median",
    )


def optional_column(
    label: str,
    columns: list[str],
    *,
    key: str,
) -> Optional[str]:
    selected = st.selectbox(
        label,
        [NONE_OPTION, *columns],
        key=key,
    )
    return None if selected == NONE_OPTION else selected


def store_data_bundle(
    *,
    raw_df: pd.DataFrame,
    data_pipeline: DemandDataPipeline,
    prepared_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    current_stock_df: Optional[pd.DataFrame],
    stock_is_real: bool,
    has_price: bool,
    is_demo: bool,
    data_label: str,
) -> None:
    st.session_state.raw_df = raw_df
    st.session_state.data_pipeline = data_pipeline
    st.session_state.prepared_df = prepared_df
    st.session_state.analysis_df = analysis_df
    st.session_state.current_stock_df = current_stock_df
    st.session_state.stock_is_real = stock_is_real
    st.session_state.has_price = has_price
    st.session_state.is_demo = is_demo
    st.session_state.data_label = data_label
    reset_forecast_and_analysis()


def render_status() -> None:
    data_ready = st.session_state.prepared_df is not None
    forecast_ready = st.session_state.future_forecast_df is not None
    analysis_ready = st.session_state.demand_plan_df is not None

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Veri",
        "Hazır" if data_ready else "Bekliyor",
    )
    col2.metric(
        "Tahmin",
        "Hazır" if forecast_ready else "Bekliyor",
    )
    col3.metric(
        "Analiz",
        "Hazır" if analysis_ready else "Bekliyor",
    )


def render_prepared_summary() -> None:
    prepared_df = st.session_state.prepared_df
    pipeline = st.session_state.data_pipeline

    if prepared_df is None or pipeline is None:
        return

    st.success(
        f"Veri hazır: {st.session_state.data_label}"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Satır", format_number(len(prepared_df)))
    col2.metric(
        "Seri",
        format_number(prepared_df["series_id"].nunique()),
    )
    col3.metric(
        "Mağaza",
        format_number(prepared_df["store_id"].nunique()),
    )
    col4.metric(
        "Ürün",
        format_number(prepared_df["product_id"].nunique()),
    )

    stockout_rate = (
        prepared_df["is_stockout"].astype(bool).mean() * 100
    )
    st.caption(
        f"Stokout oranı: %{format_number(stockout_rate, 2)} · "
        f"Frekans: {pipeline.frequency_label_tr} · "
        f"Fiyat: {'Var' if st.session_state.has_price else 'Yok'} · "
        f"Stok: {'Gerçek' if st.session_state.stock_is_real else 'Demo/eksik'}"
    )

    with st.expander("Pipeline raporu"):
        stats = pipeline.report.get("stats", {})
        warnings = pipeline.report.get("warnings", [])

        if stats:
            st.json(stats)
        if warnings:
            for warning in warnings:
                st.warning(str(warning))
        else:
            st.info("Pipeline uyarısı bulunmuyor.")

    st.dataframe(
        prepared_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Hazırlanmış veriyi indir",
        data=dataframe_csv_bytes(prepared_df),
        file_name="prepared_data.csv",
        mime="text/csv",
    )


def run_forecasting(
    model_keys: list[str],
    horizon: int,
    min_context: int,
    max_series: int,
) -> None:
    prepared_df = st.session_state.prepared_df
    pipeline = st.session_state.data_pipeline

    if prepared_df is None or pipeline is None:
        raise ValueError("Önce veriyi hazırlayın.")

    if len(model_keys) == 1:
        model_key = model_keys[0]
        forecaster = None
        model_mvp = None

        try:
            forecaster = create_forecaster(
                model_key,
                freq=pipeline.pandas_freq,
            )
            model_mvp = DemandForecastMVP(
                pipeline,
                forecaster,
            )

            evaluation_df, metrics = model_mvp.backtest(
                prepared_df=prepared_df,
                horizon=horizon,
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

            future_forecast_df = model_mvp.forecast(
                prepared_df=prepared_df,
                horizon=horizon,
                min_context=min_context,
                max_series=max_series,
            )

            evaluation_df["model_key"] = model_key
            evaluation_df["model_name"] = metrics["model_name"]

            metrics_df = pd.DataFrame([metrics])
            evaluations_df = evaluation_df
            errors_df = pd.DataFrame()
        finally:
            if model_mvp is not None:
                del model_mvp
            if forecaster is not None:
                del forecaster
            gc.collect()

    else:
        metrics_df, evaluations_df, errors_df = (
            compare_zero_shot_models(
                prepared_df=prepared_df,
                data_pipeline=pipeline,
                model_keys=model_keys,
                horizon=horizon,
                min_context=min_context,
                max_series=max_series,
            )
        )

        if metrics_df.empty:
            raise RuntimeError(
                "Seçilen modellerin hiçbiri başarıyla tamamlanamadı."
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

        best_forecaster = None
        best_mvp = None

        try:
            best_forecaster = create_forecaster(
                best_model_key,
                freq=pipeline.pandas_freq,
            )
            best_mvp = DemandForecastMVP(
                pipeline,
                best_forecaster,
            )
            future_forecast_df = best_mvp.forecast(
                prepared_df=prepared_df,
                horizon=horizon,
                min_context=min_context,
                max_series=max_series,
            )
        finally:
            if best_mvp is not None:
                del best_mvp
            if best_forecaster is not None:
                del best_forecaster
            gc.collect()

    future_forecast_df["predictions"] = (
        pd.to_numeric(
            future_forecast_df["predictions"],
            errors="coerce",
        )
        .clip(lower=0)
    )

    # Demo satışları tam adet ölçeğindedir.
    if st.session_state.is_demo:
        future_forecast_df["predictions"] = (
            future_forecast_df["predictions"]
            .round()
            .astype(int)
        )

    st.session_state.metrics_df = metrics_df
    st.session_state.evaluations_df = evaluations_df
    st.session_state.model_errors_df = errors_df
    st.session_state.future_forecast_df = future_forecast_df
    st.session_state.forecast_settings = {
        "model_keys": model_keys,
        "horizon": horizon,
        "min_context": min_context,
        "max_series": max_series,
    }

    # Eski analiz, yeni tahminle uyumlu olmayabilir.
    for key in (
        "historical_loss_detail_df",
        "historical_loss_summary_df",
        "future_demand_detail_df",
        "demand_plan_df",
        "management_kpis_df",
        "abc_product_df",
        "abc_summary_df",
        "recommendations_df",
    ):
        st.session_state[key] = None


def run_business_analysis(
    safety_periods: float,
    use_revenue_priority: bool,
    use_abc: bool,
) -> None:
    analysis_df = st.session_state.analysis_df
    prepared_df = st.session_state.prepared_df
    future_forecast_df = st.session_state.future_forecast_df
    pipeline = st.session_state.data_pipeline

    if (
        analysis_df is None
        or prepared_df is None
        or future_forecast_df is None
        or pipeline is None
    ):
        raise ValueError(
            "Analiz için hazırlanmış veri ve gelecek tahmini gereklidir."
        )

    historical_detail, historical_summary = (
        build_historical_loss_analysis(
            analysis_df
        )
    )

    forecast_input = future_forecast_df[
        ["series_id", "date", "predictions"]
    ].copy()

    future_detail, demand_plan = (
        build_future_demand_plan(
            future_forecast_df=forecast_input,
            prepared_df=prepared_df,
            historical_loss_summary_df=historical_summary,
            current_stock_df=(
                st.session_state.current_stock_df
            ),
            stock_is_real=(
                st.session_state.stock_is_real
            ),
            stock_timing=pipeline.stock_timing,
            safety_periods=safety_periods,
        )
    )

    has_revenue = (
        "estimated_lost_revenue"
        in historical_summary.columns
        and (
            "latest_price" in demand_plan.columns
            or "price" in demand_plan.columns
        )
    )

    if use_revenue_priority and has_revenue:
        demand_plan = apply_revenue_weighted_priority(
            demand_plan
        )

    abc_product_df = None
    abc_summary_df = None

    if use_abc and has_revenue:
        abc_product_df = build_historical_abc_analysis(
            historical_summary
        )
        demand_plan = add_abc_to_demand_plan(
            demand_plan,
            abc_product_df,
        )
        demand_plan = add_abc_stockout_action(
            demand_plan
        )
        abc_summary_df = build_abc_management_summary(
            demand_plan
        )

    management_kpis = build_management_kpis(
        historical_summary,
        demand_plan,
    )
    recommendations = management_recommendations(
        demand_plan,
        top_n=25,
    )

    st.session_state.historical_loss_detail_df = (
        historical_detail
    )
    st.session_state.historical_loss_summary_df = (
        historical_summary
    )
    st.session_state.future_demand_detail_df = (
        future_detail
    )
    st.session_state.demand_plan_df = demand_plan
    st.session_state.management_kpis_df = management_kpis
    st.session_state.abc_product_df = abc_product_df
    st.session_state.abc_summary_df = abc_summary_df
    st.session_state.recommendations_df = recommendations


def render_forecast_results() -> None:
    metrics_df = st.session_state.metrics_df
    evaluations_df = st.session_state.evaluations_df
    forecast_df = st.session_state.future_forecast_df
    errors_df = st.session_state.model_errors_df

    if metrics_df is None or forecast_df is None:
        return

    st.subheader("Model sonuçları")

    display_metrics = metrics_df.copy()
    preferred = [
        "model_name",
        "wmape_pct",
        "mae",
        "rmse",
        "bias_pct",
        "scored_rows",
        "excluded_stockout_rows",
        "runtime_seconds",
    ]
    display_metrics = display_metrics[
        [
            column
            for column in preferred
            if column in display_metrics.columns
        ]
    ]

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True,
    )

    if errors_df is not None and not errors_df.empty:
        with st.expander("Başarısız modeller"):
            st.dataframe(
                errors_df,
                use_container_width=True,
                hide_index=True,
            )

    if evaluations_df is not None and not evaluations_df.empty:
        st.subheader("Backtest grafiği")

        model_names = (
            evaluations_df["model_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        selected_model = st.selectbox(
            "Grafikte gösterilecek model",
            model_names,
            key="evaluation_model",
        )

        model_eval = evaluations_df.loc[
            evaluations_df["model_name"].astype(str).eq(
                selected_model
            )
        ]

        series_ids = (
            model_eval["series_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        selected_series = st.selectbox(
            "Backtest serisi",
            series_ids,
            key="evaluation_series",
        )

        plot_df = (
            model_eval.loc[
                model_eval["series_id"]
                .astype(str)
                .eq(selected_series),
                ["date", "actual", "predictions"],
            ]
            .sort_values("date")
            .set_index("date")
        )
        st.line_chart(plot_df)

    st.subheader("Gelecek tahmini")

    series_ids = (
        forecast_df["series_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    selected_series = st.selectbox(
        "Tahmin serisi",
        series_ids,
        key="future_series",
    )

    selected_forecast = (
        forecast_df.loc[
            forecast_df["series_id"]
            .astype(str)
            .eq(selected_series)
        ]
        .sort_values("date")
    )

    st.line_chart(
        selected_forecast.set_index("date")[
            ["predictions"]
        ]
    )

    st.dataframe(
        forecast_df.head(500),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Gelecek tahminlerini indir",
        data=dataframe_csv_bytes(forecast_df),
        file_name="future_forecast.csv",
        mime="text/csv",
    )


def render_analysis_results() -> None:
    demand_plan = st.session_state.demand_plan_df
    history = st.session_state.historical_loss_summary_df
    kpis = st.session_state.management_kpis_df
    abc_product = st.session_state.abc_product_df
    abc_summary = st.session_state.abc_summary_df

    if demand_plan is None or history is None:
        return

    total_lost_demand = history["estimated_lost_demand"].sum()
    total_future_demand = demand_plan["planned_demand"].sum()
    risky_count = (
        demand_plan["stockout_risk"]
        .astype("boolean")
        .fillna(False)
        .sum()
        if "stockout_risk" in demand_plan.columns
        else 0
    )
    replenishment = (
        demand_plan["recommended_replenishment"].sum()
        if "recommended_replenishment"
        in demand_plan.columns
        else np.nan
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Tarihsel kayıp talep",
        format_number(total_lost_demand),
    )
    col2.metric(
        "Tahmin ufku talebi",
        format_number(total_future_demand),
    )
    col3.metric(
        "Riskli mağaza-ürün",
        format_number(risky_count),
    )
    col4.metric(
        "Önerilen ikmal",
        (
            format_number(replenishment)
            if pd.notna(replenishment)
            else "Stok gerekli"
        ),
    )

    if "estimated_lost_revenue" in history.columns:
        col1, col2 = st.columns(2)
        col1.metric(
            "Tarihsel kayıp ciro",
            format_try(
                history[
                    "estimated_lost_revenue"
                ].sum()
            ),
        )
        col2.metric(
            "Aksiyon alınmazsa ciro riski",
            format_try(
                demand_plan.get(
                    "expected_lost_revenue_no_action",
                    pd.Series(dtype=float),
                ).sum()
            ),
        )

    tabs = st.tabs(
        [
            "Yönetici özeti",
            "Tarihsel stokout",
            "İkmal planı",
            "ABC",
            "Dışa aktar",
        ]
    )

    with tabs[0]:
        if kpis is not None:
            st.dataframe(
                kpis,
                use_container_width=True,
                hide_index=True,
            )

        top_columns = [
            "operational_priority",
            "abc_class",
            "priority",
            "store_id",
            "product_id",
            "latest_price",
            "planned_demand",
            "current_stock",
            "recommended_replenishment",
            "recommended_replenishment_value",
            "expected_lost_revenue_no_action",
            "recommended_action",
            "abc_stock_action",
        ]

        st.dataframe(
            demand_plan[
                [
                    column
                    for column in top_columns
                    if column in demand_plan.columns
                ]
            ].head(30),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        quantity_top = (
            history.nlargest(
                20,
                "estimated_lost_demand",
            )
            .copy()
        )
        quantity_top["mağaza / ürün"] = (
            quantity_top["store_id"].astype(str)
            + " / "
            + quantity_top["product_id"].astype(str)
        )
        st.bar_chart(
            quantity_top.set_index("mağaza / ürün")[
                ["estimated_lost_demand"]
            ]
        )

        if "estimated_lost_revenue" in history.columns:
            revenue_top = (
                history.nlargest(
                    20,
                    "estimated_lost_revenue",
                )
                .copy()
            )
            revenue_top["mağaza / ürün"] = (
                revenue_top["store_id"].astype(str)
                + " / "
                + revenue_top["product_id"].astype(str)
            )
            st.bar_chart(
                revenue_top.set_index("mağaza / ürün")[
                    ["estimated_lost_revenue"]
                ]
            )

        st.dataframe(
            history.head(500),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        if (
            "expected_lost_revenue_no_action"
            in demand_plan.columns
        ):
            risk_top = (
                demand_plan.nlargest(
                    20,
                    "expected_lost_revenue_no_action",
                )
                .copy()
            )
            risk_top["mağaza / ürün"] = (
                risk_top["store_id"].astype(str)
                + " / "
                + risk_top["product_id"].astype(str)
            )
            st.bar_chart(
                risk_top.set_index("mağaza / ürün")[
                    ["expected_lost_revenue_no_action"]
                ]
            )
        elif (
            "recommended_replenishment"
            in demand_plan.columns
            and demand_plan[
                "recommended_replenishment"
            ].notna().any()
        ):
            repl_top = (
                demand_plan.nlargest(
                    20,
                    "recommended_replenishment",
                )
                .copy()
            )
            repl_top["mağaza / ürün"] = (
                repl_top["store_id"].astype(str)
                + " / "
                + repl_top["product_id"].astype(str)
            )
            st.bar_chart(
                repl_top.set_index("mağaza / ürün")[
                    ["recommended_replenishment"]
                ]
            )

        st.dataframe(
            demand_plan.head(1000),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        if abc_product is None:
            st.info(
                "ABC analizi için fiyat/ciro bilgisi gerekir."
            )
        else:
            st.dataframe(
                abc_product,
                use_container_width=True,
                hide_index=True,
            )

            class_value = (
                abc_product.groupby(
                    "abc_class",
                    as_index=False,
                )
                .agg(
                    total_value=("total_value", "sum"),
                    product_count=("product_id", "nunique"),
                )
                .sort_values("abc_class")
            )
            st.bar_chart(
                class_value.set_index("abc_class")[
                    ["total_value"]
                ]
            )

            if abc_summary is not None:
                st.dataframe(
                    abc_summary,
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[4]:
        result_tables = {
            "model_metrics.csv": st.session_state.metrics_df,
            "future_forecast.csv": (
                st.session_state.future_forecast_df
            ),
            "historical_loss_summary.csv": history,
            "future_demand_detail.csv": (
                st.session_state.future_demand_detail_df
            ),
            "demand_plan.csv": demand_plan,
            "management_kpis.csv": kpis,
            "abc_product_analysis.csv": abc_product,
            "abc_management_summary.csv": abc_summary,
            "recommendations.csv": (
                st.session_state.recommendations_df
            ),
        }

        st.download_button(
            "Tüm sonuçları ZIP olarak indir",
            data=tables_zip_bytes(result_tables),
            file_name="demand_planning_results.zip",
            mime="application/zip",
        )


initialise_state()

st.title("📦 Demand Planning AI")
st.caption(
    "Zero-shot talep tahmini, stokout kaybı, ikmal, ciro riski ve ABC analizi"
)

with st.sidebar:
    st.header("Proje")
    st.caption(
        "Akış: Veri → Tahmin → İş analizi"
    )

    if st.button(
        "Tüm oturumu sıfırla",
        use_container_width=True,
    ):
        reset_everything()
        st.rerun()

    st.divider()
    st.write(
        "Demo modunda satış, fiyat ve stok senaryo amaçlıdır. "
        "Yüklenen şirket verisinde gerçek sütunlar kullanılır."
    )

render_status()

data_tab, forecast_tab, analysis_tab = st.tabs(
    [
        "1 · Veri",
        "2 · Tahmin",
        "3 · İş analizi",
    ]
)

with data_tab:
    source_mode = st.radio(
        "Veri kaynağı",
        [
            "FreshRetailNet Demo",
            "Kendi dosyam",
        ],
        horizontal=True,
        key="source_mode",
    )

    if source_mode == "FreshRetailNet Demo":
        st.subheader("FreshRetailNet demo verisi")

        col1, col2, col3 = st.columns(3)
        n_series = col1.number_input(
            "Seri sayısı",
            min_value=5,
            max_value=500,
            value=100,
            step=5,
        )
        train_periods = col2.number_input(
            "Train dönemi",
            min_value=30,
            max_value=365,
            value=90,
            step=5,
        )
        eval_periods = col3.number_input(
            "Eval dönemi",
            min_value=1,
            max_value=60,
            value=7,
            step=1,
        )

        if st.button(
            "FreshRetailNet verisini yükle",
            type="primary",
        ):
            try:
                with st.spinner(
                    "FreshRetailNet indiriliyor ve seriler seçiliyor..."
                ):
                    raw_df = load_freshretail_demo(
                        int(n_series),
                        int(train_periods),
                        int(eval_periods),
                    )
                st.session_state.raw_df = raw_df
                reset_forecast_and_analysis()
                st.success("Demo veri yüklendi.")
            except Exception as error:
                st.error(str(error))
                with st.expander("Teknik hata"):
                    st.code(traceback.format_exc())

        raw_df = st.session_state.raw_df

        if raw_df is not None:
            st.dataframe(
                raw_df.head(100),
                use_container_width=True,
                hide_index=True,
            )

            positive_sales = pd.to_numeric(
                raw_df["sale_amount"],
                errors="coerce",
            )
            positive_sales = positive_sales.loc[
                positive_sales.gt(0)
            ]

            automatic_multiplier = (
                choose_demo_unit_multiplier(
                    positive_sales,
                    target_median_daily_units=20,
                )
            )

            st.subheader("Demo adet, fiyat ve stok varsayımları")

            col1, col2, col3 = st.columns(3)
            target_median = col1.number_input(
                "Hedef medyan günlük adet",
                min_value=1,
                max_value=1000,
                value=20,
                step=1,
            )

            automatic_multiplier = (
                choose_demo_unit_multiplier(
                    positive_sales,
                    target_median_daily_units=float(
                        target_median
                    ),
                )
            )

            multiplier_mode = col2.radio(
                "Global katsayı",
                ["Otomatik", "Manuel"],
                horizontal=True,
            )

            if multiplier_mode == "Otomatik":
                unit_multiplier = automatic_multiplier
                col3.metric(
                    "Seçilen katsayı",
                    format_number(unit_multiplier),
                )
            else:
                unit_multiplier = int(
                    col3.number_input(
                        "Manuel katsayı",
                        min_value=1,
                        max_value=1_000_000,
                        value=int(
                            automatic_multiplier
                        ),
                        step=1,
                    )
                )

            candidate_table = pd.DataFrame(
                {
                    "Katsayı": [
                        10,
                        20,
                        50,
                        100,
                        200,
                        500,
                        1000,
                    ]
                }
            )
            candidate_table["Medyan günlük adet"] = (
                candidate_table["Katsayı"]
                * positive_sales.median()
            )
            candidate_table["P90 günlük adet"] = (
                candidate_table["Katsayı"]
                * positive_sales.quantile(0.90)
            )

            with st.expander("Katsayı karşılaştırması"):
                st.dataframe(
                    candidate_table,
                    use_container_width=True,
                    hide_index=True,
                )

            col1, col2, col3, col4 = st.columns(4)
            price_seed = col1.number_input(
                "Fiyat seed",
                min_value=0,
                max_value=1_000_000,
                value=42,
            )
            min_price = col2.number_input(
                "Minimum fiyat (TL)",
                min_value=1.0,
                value=19.90,
                step=5.0,
            )
            max_price = col3.number_input(
                "Maksimum fiyat (TL)",
                min_value=float(min_price),
                value=max(399.90, float(min_price)),
                step=10.0,
            )
            cover_periods = col4.number_input(
                "Stok kapsama dönemi",
                min_value=0.0,
                max_value=30.0,
                value=3.0,
                step=0.5,
            )

            safety_periods_demo = st.number_input(
                "Sentetik stok güvenlik dönemi",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
            )

            if st.button(
                "Demo veriyi hazırla",
                type="primary",
            ):
                try:
                    with st.spinner(
                        "Pipeline, demo adet, fiyat ve stok oluşturuluyor..."
                    ):
                        pipeline = build_fresh_pipeline()
                        normalized_prepared = pipeline.prepare(
                            raw_df
                        )
                        normalized_adjusted = (
                            pipeline.impute_stockouts(
                                normalized_prepared
                            )
                        )

                        (
                            demo_adjusted,
                            product_catalog,
                            selected_multiplier,
                        ) = prepare_freshretail_demo_commercial_data(
                            adjusted_df=normalized_adjusted,
                            raw_df=raw_df,
                            unit_multiplier=int(
                                unit_multiplier
                            ),
                            target_median_daily_units=float(
                                target_median
                            ),
                            seed=int(price_seed),
                            min_price_try=float(min_price),
                            max_price_try=float(max_price),
                            raw_date_column="dt",
                            raw_discount_column="discount",
                            raw_series_id_column=(
                                "temporary_series_id"
                            ),
                        )

                        demo_prepared = make_demo_prepared_df(
                            normalized_prepared,
                            demo_adjusted,
                        )

                        demo_inventory = (
                            generate_demo_inventory(
                                demo_adjusted,
                                demand_column=(
                                    "demand_adjusted"
                                ),
                                rolling_window=28,
                                min_history=7,
                                cover_periods=float(
                                    cover_periods
                                ),
                                safety_periods=float(
                                    safety_periods_demo
                                ),
                            )
                        )
                        demo_current_stock = (
                            make_demo_current_stock(
                                demo_inventory
                            )
                        )

                    store_data_bundle(
                        raw_df=raw_df,
                        data_pipeline=pipeline,
                        prepared_df=demo_prepared,
                        analysis_df=demo_adjusted,
                        current_stock_df=(
                            demo_current_stock
                        ),
                        stock_is_real=False,
                        has_price=True,
                        is_demo=True,
                        data_label=(
                            "FreshRetailNet demo · "
                            f"katsayı={selected_multiplier}"
                        ),
                    )

                    st.session_state[
                        "demo_product_catalog_df"
                    ] = product_catalog
                    st.session_state[
                        "demo_inventory_df"
                    ] = demo_inventory

                    st.success(
                        "Demo satış, fiyat ve stok verisi hazır."
                    )
                except Exception as error:
                    st.error(str(error))
                    with st.expander("Teknik hata"):
                        st.code(traceback.format_exc())

    else:
        st.subheader("Şirket verisi yükle")

        uploaded_file = st.file_uploader(
            "CSV, XLSX veya Parquet",
            type=["csv", "xlsx", "parquet"],
        )

        if uploaded_file is not None:
            try:
                uploaded_df = read_uploaded_table(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                )
                st.session_state.raw_df = uploaded_df
            except Exception as error:
                st.error(str(error))
                uploaded_df = None

            if uploaded_df is not None:
                st.dataframe(
                    uploaded_df.head(100),
                    use_container_width=True,
                    hide_index=True,
                )

                columns = uploaded_df.columns.astype(
                    str
                ).tolist()

                st.subheader("Sütun eşleme")

                col1, col2 = st.columns(2)

                with col1:
                    date_column = st.selectbox(
                        "Tarih",
                        columns,
                        key="map_date",
                    )
                    store_column = st.selectbox(
                        "Mağaza ID",
                        columns,
                        key="map_store",
                    )
                    product_column = st.selectbox(
                        "Ürün ID",
                        columns,
                        key="map_product",
                    )
                    sales_column = st.selectbox(
                        "Satış adedi",
                        columns,
                        key="map_sales",
                    )

                with col2:
                    no_stock = st.checkbox(
                        "Stok sütunum yok",
                        value=False,
                    )

                    stock_column = (
                        None
                        if no_stock
                        else st.selectbox(
                            "Stok miktarı",
                            columns,
                            key="map_stock",
                        )
                    )

                    price_column = optional_column(
                        "Birim fiyat",
                        columns,
                        key="map_price",
                    )
                    stockout_column = optional_column(
                        "Stokout bayrağı",
                        columns,
                        key="map_stockout",
                    )

                col1, col2, col3 = st.columns(3)
                with col1:
                    category_1 = optional_column(
                        "Kategori 1",
                        columns,
                        key="map_category_1",
                    )
                with col2:
                    category_2 = optional_column(
                        "Kategori 2",
                        columns,
                        key="map_category_2",
                    )
                with col3:
                    category_3 = optional_column(
                        "Kategori 3",
                        columns,
                        key="map_category_3",
                    )

                st.subheader("Pipeline ayarları")

                col1, col2, col3 = st.columns(3)
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

                col1, col2, col3 = st.columns(3)
                dayfirst = col1.checkbox(
                    "Tarih gün/ay/yıl",
                    value=False,
                )
                stock_threshold = col2.number_input(
                    "Stokout stok eşiği",
                    value=0.0,
                    step=1.0,
                )
                min_history = col3.number_input(
                    "Demand-adjustment minimum geçmiş",
                    min_value=1,
                    max_value=100,
                    value=7,
                )

                if st.button(
                    "Şirket verisini hazırla",
                    type="primary",
                ):
                    try:
                        if no_stock and stockout_column is None:
                            raise ValueError(
                                "Stok sütunu yoksa stokout bayrağı "
                                "seçilmelidir."
                            )

                        working_df = uploaded_df.copy()
                        actual_stock_column = stock_column

                        if no_stock:
                            working_df[
                                "__stock_quantity_proxy__"
                            ] = 1.0
                            actual_stock_column = (
                                "__stock_quantity_proxy__"
                            )

                        mapping = ColumnMapping(
                            date=date_column,
                            store=store_column,
                            product=product_column,
                            sales=sales_column,
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
                            duplicate_policy=(
                                duplicate_policy
                            ),
                            date_gap_policy=(
                                date_gap_policy
                            ),
                            negative_value_policy="clip",
                            use_sales_equals_stock_rule=False,
                            stockout_tolerance=0.0,
                            stockout_stock_threshold=float(
                                stock_threshold
                            ),
                            combine_provided_flag_with_inferred=(
                                not no_stock
                            ),
                            stock_timing="end_of_period",
                            imputation_window=28,
                            min_history=int(min_history),
                            imputation_statistic="median",
                        )

                        prepared_df = pipeline.prepare(
                            working_df
                        )
                        adjusted_df = (
                            pipeline.impute_stockouts(
                                prepared_df
                            )
                        )

                        store_data_bundle(
                            raw_df=working_df,
                            data_pipeline=pipeline,
                            prepared_df=prepared_df,
                            analysis_df=adjusted_df,
                            current_stock_df=None,
                            stock_is_real=not no_stock,
                            has_price=(
                                price_column is not None
                            ),
                            is_demo=False,
                            data_label=uploaded_file.name,
                        )

                        st.success(
                            "Şirket verisi hazırlandı."
                        )
                    except (
                        DataValidationError,
                        ValueError,
                    ) as error:
                        st.error(str(error))
                    except Exception as error:
                        st.error(str(error))
                        with st.expander("Teknik hata"):
                            st.code(
                                traceback.format_exc()
                            )

    render_prepared_summary()

with forecast_tab:
    if st.session_state.prepared_df is None:
        st.info(
            "Önce Veri sekmesinde veri setini hazırlayın."
        )
    else:
        pipeline = st.session_state.data_pipeline
        prepared_df = st.session_state.prepared_df
        defaults = FREQUENCY_DEFAULTS[
            pipeline.frequency
        ]

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
        )

        col1, col2, col3 = st.columns(3)
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
        max_series_limit = int(
            prepared_df["series_id"].nunique()
        )
        max_series = int(
            col3.number_input(
                "Maksimum seri",
                min_value=1,
                max_value=max_series_limit,
                value=min(100, max_series_limit),
            )
        )

        st.caption(
            "İlk model çalıştırmasında model ağırlıkları "
            "Hugging Face üzerinden indirilir."
        )

        if st.button(
            "Tahmini çalıştır",
            type="primary",
            disabled=not selected_models,
        ):
            try:
                with st.spinner(
                    "Model backtest ve gelecek tahmini çalışıyor..."
                ):
                    run_forecasting(
                        model_keys=selected_models,
                        horizon=horizon,
                        min_context=min_context,
                        max_series=max_series,
                    )
                st.success("Tahmin tamamlandı.")
            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )
                with st.expander("Teknik hata"):
                    st.code(traceback.format_exc())

        render_forecast_results()

with analysis_tab:
    if st.session_state.future_forecast_df is None:
        st.info(
            "Önce Tahmin sekmesinde gelecek tahminini oluşturun."
        )
    else:
        has_price = st.session_state.has_price

        col1, col2, col3 = st.columns(3)
        safety_periods = col1.number_input(
            "Güvenlik stoğu dönemi",
            min_value=0.0,
            max_value=30.0,
            value=1.0,
            step=0.5,
        )

        use_revenue_priority = col2.checkbox(
            "Fiyat/ciro bazlı öncelik",
            value=has_price,
            disabled=not has_price,
        )

        use_abc = col3.checkbox(
            "ABC analizi",
            value=has_price,
            disabled=not has_price,
        )

        if not has_price:
            st.warning(
                "Fiyat sütunu olmadığı için ciro önceliği ve ABC "
                "analizi çalıştırılmayacak."
            )

        if st.button(
            "İş analizini çalıştır",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Stokout, ikmal, ciro ve ABC analizleri hesaplanıyor..."
                ):
                    run_business_analysis(
                        safety_periods=float(
                            safety_periods
                        ),
                        use_revenue_priority=(
                            use_revenue_priority
                        ),
                        use_abc=use_abc,
                    )
                st.success("İş analizi tamamlandı.")
            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )
                with st.expander("Teknik hata"):
                    st.code(traceback.format_exc())

        render_analysis_results()
