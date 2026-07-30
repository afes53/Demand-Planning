from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


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


def build_historical_loss_analysis(
    adjusted_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stokout kaynaklı tahmini tarihsel kaybı hesaplar.

    Kayıp talep:
        max(demand_adjusted - sales, 0)

    Fiyat varsa kayıp ciro:
        estimated_lost_demand * price

    Returns
    -------
    detail_df:
        Dönem seviyesinde kayıp detayları.
    summary_df:
        Mağaza-ürün seviyesinde yönetim özeti.
    """
    required = {
        "date",
        "series_id",
        "store_id",
        "product_id",
        "sales",
        "demand_adjusted",
        "is_stockout",
    }
    _require_columns(adjusted_df, required, "adjusted_df")

    detail = adjusted_df.copy()
    detail["date"] = pd.to_datetime(detail["date"], errors="raise")
    detail["sales"] = pd.to_numeric(detail["sales"], errors="coerce")
    detail["demand_adjusted"] = pd.to_numeric(
        detail["demand_adjusted"], errors="coerce"
    )

    if detail[["sales", "demand_adjusted"]].isna().any().any():
        raise ValueError("Satış veya demand_adjusted sütununda geçersiz değer var.")

    detail["estimated_lost_demand"] = (
        detail["demand_adjusted"] - detail["sales"]
    ).clip(lower=0)

    detail["lost_demand_period"] = detail["estimated_lost_demand"].gt(0)

    has_price = "price" in detail.columns
    if has_price:
        detail["price"] = pd.to_numeric(detail["price"], errors="coerce")
        detail["observed_revenue"] = detail["sales"] * detail["price"]
        detail["estimated_lost_revenue"] = (
            detail["estimated_lost_demand"] * detail["price"]
        )

    group_columns = ["series_id", "store_id", "product_id"]
    for category_column in ("category_1", "category_2", "category_3"):
        if category_column in detail.columns:
            group_columns.append(category_column)

    aggregations: dict[str, tuple[str, str]] = {
        "period_count": ("date", "size"),
        "stockout_period_count": ("is_stockout", "sum"),
        "observed_sales": ("sales", "sum"),
        "adjusted_demand": ("demand_adjusted", "sum"),
        "estimated_lost_demand": ("estimated_lost_demand", "sum"),
        "loss_period_count": ("lost_demand_period", "sum"),
    }

    if has_price:
        aggregations.update(
            {
                "observed_revenue": ("observed_revenue", "sum"),
                "estimated_lost_revenue": ("estimated_lost_revenue", "sum"),
                "average_price": ("price", "mean"),
            }
        )

    summary = (
        detail.groupby(group_columns, as_index=False, dropna=False)
        .agg(**aggregations)
    )

    summary["stockout_rate_pct"] = (
        summary["stockout_period_count"]
        / summary["period_count"].replace(0, np.nan)
        * 100
    )

    summary["lost_demand_share_pct"] = (
        summary["estimated_lost_demand"]
        / summary["adjusted_demand"].replace(0, np.nan)
        * 100
    )

    sort_column = (
        "estimated_lost_revenue"
        if "estimated_lost_revenue" in summary.columns
        else "estimated_lost_demand"
    )

    summary = summary.sort_values(
        [sort_column, "stockout_rate_pct"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return detail.sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True), summary


def make_current_stock_template(
    prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Güncel stok yükleme şablonu oluşturur.

    FreshRetailNet gibi gerçek sayısal stok vermeyen veri setlerinde
    kullanıcı bu şablonu gerçek stok miktarlarıyla doldurabilir.
    """
    required = {"series_id", "store_id", "product_id"}
    _require_columns(prepared_df, required, "prepared_df")

    columns = ["series_id", "store_id", "product_id"]
    for category_column in ("category_1", "category_2", "category_3"):
        if category_column in prepared_df.columns:
            columns.append(category_column)

    template = (
        prepared_df.sort_values("date")
        .groupby("series_id", as_index=False)
        .tail(1)[columns]
        .drop_duplicates("series_id")
        .sort_values(["store_id", "product_id"])
        .reset_index(drop=True)
    )

    template["current_stock"] = np.nan
    return template


def _standardise_current_stock(
    current_stock_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    stock = current_stock_df.copy()

    if "current_stock" not in stock.columns:
        raise ValueError(
            "Güncel stok dosyasında 'current_stock' sütunu bulunmalıdır."
        )

    stock["current_stock"] = pd.to_numeric(
        stock["current_stock"], errors="coerce"
    )

    if stock["current_stock"].isna().any():
        raise ValueError(
            "current_stock sütununda boş veya sayısal olmayan değerler var."
        )

    if stock["current_stock"].lt(0).any():
        raise ValueError("current_stock negatif değer içeremez.")

    if "series_id" in stock.columns:
        result = stock[["series_id", "current_stock"]].copy()
        result["series_id"] = result["series_id"].astype(str)
    elif {"store_id", "product_id"}.issubset(stock.columns):
        stock["store_id"] = stock["store_id"].astype(str)
        stock["product_id"] = stock["product_id"].astype(str)
        lookup = (
            prepared_df[["series_id", "store_id", "product_id"]]
            .drop_duplicates()
            .copy()
        )
        lookup["store_id"] = lookup["store_id"].astype(str)
        lookup["product_id"] = lookup["product_id"].astype(str)
        result = stock.merge(
            lookup,
            on=["store_id", "product_id"],
            how="left",
            validate="one_to_one",
        )[["series_id", "current_stock"]]
    else:
        raise ValueError(
            "Güncel stok dosyasında ya 'series_id' ya da "
            "'store_id' ve 'product_id' sütunları bulunmalıdır."
        )

    if result["series_id"].isna().any():
        raise ValueError(
            "Bazı güncel stok satırları hazırlanmış veriyle eşleştirilemedi."
        )

    duplicate_mask = result.duplicated("series_id", keep=False)
    if duplicate_mask.any():
        raise ValueError(
            "Güncel stok dosyasında aynı seri için birden fazla satır var."
        )

    return result


def build_future_demand_plan(
    future_forecast_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    historical_loss_summary_df: Optional[pd.DataFrame] = None,
    current_stock_df: Optional[pd.DataFrame] = None,
    *,
    stock_is_real: bool = True,
    stock_timing: str = "end_of_period",
    safety_periods: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gelecek talep, stok açığı ve önerilen sevkiyat planını oluşturur.

    current_stock_df verilirse bu tablo kullanılır.
    Aksi halde stock_is_real=True ise prepared_df içindeki son stok kullanılır.
    stock_is_real=False ise sevkiyat miktarı hesaplanmaz; yalnızca talep önceliği çıkarılır.

    safety_periods:
        Ortalama bir dönemlik talebin kaç katının güvenlik stoğu olarak
        ekleneceğini belirler.
    """
    _require_columns(
        future_forecast_df,
        {"series_id", "date", "predictions"},
        "future_forecast_df",
    )
    _require_columns(
        prepared_df,
        {"series_id", "date", "store_id", "product_id", "stock", "sales"},
        "prepared_df",
    )

    if safety_periods < 0:
        raise ValueError("safety_periods negatif olamaz.")

    forecast = future_forecast_df.copy()
    forecast["date"] = pd.to_datetime(forecast["date"], errors="raise")
    forecast["predictions"] = pd.to_numeric(
        forecast["predictions"], errors="coerce"
    ).clip(lower=0)

    if forecast["predictions"].isna().any():
        raise ValueError("Tahmin tablosunda geçersiz predictions değeri var.")

    metadata_columns = ["series_id", "store_id", "product_id"]
    for column in ("category_1", "category_2", "category_3", "price"):
        if column in prepared_df.columns:
            metadata_columns.append(column)

    # future_forecast_df, DemandForecastMVP.forecast() tarafından üretildiyse
    # store_id/product_id zaten içeride olabilir. Aynı sütunları tekrar merge etmek
    # store_id_x/store_id_y oluşturur ve aşağıdaki groupby işlemini bozar.
    missing_metadata_columns = [
        column
        for column in metadata_columns
        if column != "series_id" and column not in forecast.columns
    ]

    if missing_metadata_columns:
        metadata = (
            prepared_df.sort_values("date")
            .groupby("series_id", as_index=False)
            .tail(1)[["series_id", *missing_metadata_columns]]
            .drop_duplicates("series_id")
        )

        forecast = forecast.merge(
            metadata,
            on="series_id",
            how="left",
            validate="many_to_one",
        )

    # Daha önce yanlış bir merge yapılmış tablo verilirse suffix'leri toparla.
    for base_column in (
        "store_id",
        "product_id",
        "category_1",
        "category_2",
        "category_3",
        "price",
    ):
        if base_column not in forecast.columns:
            candidate_columns = [
                column
                for column in (f"{base_column}_x", f"{base_column}_y")
                if column in forecast.columns
            ]
            if candidate_columns:
                forecast[base_column] = forecast[candidate_columns[0]]

    _require_columns(
        forecast,
        {"store_id", "product_id"},
        "future_forecast_df + metadata",
    )

    if current_stock_df is not None:
        current_stock = _standardise_current_stock(
            current_stock_df=current_stock_df,
            prepared_df=prepared_df,
        )
        stock_source = "uploaded_current_stock"
    elif stock_is_real:
        latest = (
            prepared_df.sort_values("date")
            .groupby("series_id", as_index=False)
            .tail(1)[["series_id", "stock", "sales"]]
            .copy()
        )
        if stock_timing == "start_of_period":
            latest["current_stock"] = (
                pd.to_numeric(latest["stock"], errors="coerce")
                - pd.to_numeric(latest["sales"], errors="coerce")
            ).clip(lower=0)
        elif stock_timing == "end_of_period":
            latest["current_stock"] = pd.to_numeric(
                latest["stock"], errors="coerce"
            ).clip(lower=0)
        else:
            raise ValueError(
                "stock_timing 'start_of_period' veya 'end_of_period' olmalıdır."
            )
        current_stock = latest[["series_id", "current_stock"]]
        stock_source = "latest_dataset_stock"
    else:
        current_stock = (
            forecast[["series_id"]]
            .drop_duplicates()
            .copy()
        )
        current_stock["current_stock"] = np.nan
        stock_source = "not_available"

    forecast = forecast.merge(
        current_stock,
        on="series_id",
        how="left",
        validate="many_to_one",
    )

    forecast = forecast.sort_values(["series_id", "date"]).reset_index(drop=True)
    forecast["cumulative_demand"] = forecast.groupby("series_id")[
        "predictions"
    ].cumsum()

    forecast["projected_stock"] = (
        forecast["current_stock"] - forecast["cumulative_demand"]
    )
    forecast["stockout_risk"] = (
        forecast["current_stock"].notna()
        & forecast["projected_stock"].le(0)
    )
    forecast["period_shortage"] = (
        -forecast["projected_stock"]
    ).clip(lower=0)

    summary_group_columns = ["series_id", "store_id", "product_id"]
    for category_column in ("category_1", "category_2", "category_3"):
        if category_column in forecast.columns:
            summary_group_columns.append(category_column)

    summary = (
        forecast.groupby(
            summary_group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            forecast_start=("date", "min"),
            forecast_end=("date", "max"),
            horizon_periods=("date", "size"),
            planned_demand=("predictions", "sum"),
            average_period_demand=("predictions", "mean"),
            peak_period_demand=("predictions", "max"),
            current_stock=("current_stock", "first"),
            expected_ending_stock=("projected_stock", "last"),
            expected_shortage_no_action=("period_shortage", "max"),
            stockout_risk=("stockout_risk", "max"),
        )
    )

    summary["safety_stock"] = (
        summary["average_period_demand"] * float(safety_periods)
    )
    summary["target_stock_for_horizon"] = (
        summary["planned_demand"] + summary["safety_stock"]
    )

    summary["recommended_replenishment"] = (
        summary["target_stock_for_horizon"] - summary["current_stock"]
    ).clip(lower=0)

    missing_stock_mask = summary["current_stock"].isna()
    summary["stockout_risk"] = summary["stockout_risk"].astype("boolean")

    summary.loc[
        missing_stock_mask,
        [
            "expected_ending_stock",
            "expected_shortage_no_action",
            "recommended_replenishment",
        ],
    ] = np.nan
    summary.loc[missing_stock_mask, "stockout_risk"] = pd.NA

    first_stockout = (
        forecast.loc[forecast["stockout_risk"]]
        .groupby("series_id")["date"]
        .min()
        .rename("expected_stockout_date")
    )
    summary = summary.merge(first_stockout, on="series_id", how="left")

    if "price" in forecast.columns:
        latest_price = (
            forecast.groupby("series_id", as_index=False)["price"]
            .last()
            .rename(columns={"price": "latest_price"})
        )
        summary = summary.merge(
            latest_price,
            on="series_id",
            how="left",
            validate="one_to_one",
        )
        summary["expected_lost_revenue_no_action"] = (
            summary["expected_shortage_no_action"] * summary["latest_price"]
        )

    if historical_loss_summary_df is not None:
        history_columns = [
            "series_id",
            "stockout_rate_pct",
            "estimated_lost_demand",
            "lost_demand_share_pct",
        ]
        if "estimated_lost_revenue" in historical_loss_summary_df.columns:
            history_columns.append("estimated_lost_revenue")
        available_history_columns = [
            column
            for column in history_columns
            if column in historical_loss_summary_df.columns
        ]
        summary = summary.merge(
            historical_loss_summary_df[available_history_columns],
            on="series_id",
            how="left",
            validate="one_to_one",
        )

    # Açıklanabilir öncelik puanı:
    # stok biliniyorsa ana sinyal stok açığı; bilinmiyorsa talep ve geçmiş stokout.
    if summary["current_stock"].notna().any():
        demand_denominator = summary["planned_demand"].replace(0, np.nan)
        shortage_ratio = (
            summary["expected_shortage_no_action"] / demand_denominator
        ).fillna(0)
        replenishment_ratio = (
            summary["recommended_replenishment"] / demand_denominator
        ).fillna(0)
        historical_rate = (
            summary.get("stockout_rate_pct", pd.Series(0, index=summary.index))
            .fillna(0)
            / 100
        )
        summary["priority_score"] = (
            shortage_ratio * 60
            + replenishment_ratio * 30
            + historical_rate * 10
        )
    else:
        demand_rank = summary["planned_demand"].rank(
            pct=True, method="average"
        )
        historical_rate = (
            summary.get("stockout_rate_pct", pd.Series(0, index=summary.index))
            .fillna(0)
            / 100
        )
        lost_share = (
            summary.get("lost_demand_share_pct", pd.Series(0, index=summary.index))
            .fillna(0)
            / 100
        )
        summary["priority_score"] = (
            demand_rank * 50
            + historical_rate * 30
            + lost_share * 20
        )

    score = summary["priority_score"]
    summary["priority"] = np.select(
        [
            score >= score.quantile(0.90),
            score >= score.quantile(0.70),
            score >= score.quantile(0.40),
        ],
        ["Kritik", "Yüksek", "Orta"],
        default="Düşük",
    )

    def action_text(row: pd.Series) -> str:
        if pd.notna(row["recommended_replenishment"]):
            amount = int(np.ceil(row["recommended_replenishment"]))
            if amount > 0:
                date_text = (
                    pd.Timestamp(row["expected_stockout_date"]).strftime("%Y-%m-%d")
                    if pd.notna(row.get("expected_stockout_date"))
                    else "ufuk içinde"
                )
                return (
                    f"{amount} birim sevk et; stokout riski {date_text}."
                )
            return "Mevcut stok tahmin ufku için yeterli."
        target = int(np.ceil(row["target_stock_for_horizon"]))
        return (
            f"Güncel stok girilmeli. Önümüzdeki ufuk için hedef stok "
            f"yaklaşık {target} birim."
        )

    summary["recommended_action"] = summary.apply(action_text, axis=1)
    summary["stock_source"] = stock_source

    sort_columns = ["priority_score", "planned_demand"]
    summary = summary.sort_values(
        sort_columns,
        ascending=[False, False],
    ).reset_index(drop=True)

    return forecast, summary


def allocate_product_supply(
    demand_plan_df: pd.DataFrame,
    supply_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ürün bazındaki dağıtılabilir stoğu mağaza-ürün ihtiyaçlarına öncelik sırasıyla dağıtır.

    supply_df zorunlu sütunları:
        product_id
        available_to_distribute
    """
    _require_columns(
        demand_plan_df,
        {
            "series_id",
            "store_id",
            "product_id",
            "recommended_replenishment",
            "priority_score",
        },
        "demand_plan_df",
    )
    _require_columns(
        supply_df,
        {"product_id", "available_to_distribute"},
        "supply_df",
    )

    if demand_plan_df["recommended_replenishment"].isna().any():
        raise ValueError(
            "Dağıtım planı için gerçek güncel stok gereklidir. "
            "recommended_replenishment sütununda boş değer var."
        )

    supply = supply_df.copy()
    supply["product_id"] = supply["product_id"].astype(str)
    supply["available_to_distribute"] = pd.to_numeric(
        supply["available_to_distribute"], errors="coerce"
    )

    if supply["available_to_distribute"].isna().any():
        raise ValueError(
            "available_to_distribute sütununda geçersiz değer var."
        )
    if supply["available_to_distribute"].lt(0).any():
        raise ValueError("Dağıtılabilir stok negatif olamaz.")
    if supply["product_id"].duplicated().any():
        raise ValueError("supply_df içinde ürün başına tek satır olmalıdır.")

    plan = demand_plan_df.copy()
    plan["product_id"] = plan["product_id"].astype(str)
    plan["recommended_replenishment"] = pd.to_numeric(
        plan["recommended_replenishment"], errors="raise"
    ).clip(lower=0)

    plan = plan.merge(
        supply,
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    plan["available_to_distribute"] = (
        plan["available_to_distribute"].fillna(0)
    )

    allocation_rows: list[dict[str, object]] = []

    for product_id, group in plan.groupby("product_id", sort=False):
        remaining = float(group["available_to_distribute"].iloc[0])
        ordered = group.sort_values(
            ["priority_score", "recommended_replenishment"],
            ascending=[False, False],
        )

        for _, row in ordered.iterrows():
            need = float(row["recommended_replenishment"])
            allocated = min(need, remaining)
            remaining -= allocated

            output_row = row.to_dict()
            output_row["recommended_distribution"] = allocated
            output_row["unmet_need_after_allocation"] = max(
                need - allocated, 0
            )
            output_row["product_supply_remaining"] = remaining
            allocation_rows.append(output_row)

    allocation = pd.DataFrame(allocation_rows)
    allocation["recommended_distribution"] = np.ceil(
        allocation["recommended_distribution"]
    ).astype(int)
    allocation["unmet_need_after_allocation"] = np.ceil(
        allocation["unmet_need_after_allocation"]
    ).astype(int)

    return allocation.sort_values(
        ["priority_score", "recommended_distribution"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_management_kpis(
    historical_loss_summary_df: pd.DataFrame,
    demand_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Yönetici raporu için KPI tablosu oluşturur.
    """
    kpis: list[dict[str, object]] = []

    total_lost_demand = historical_loss_summary_df[
        "estimated_lost_demand"
    ].sum()
    affected_series = int(
        historical_loss_summary_df["estimated_lost_demand"].gt(0).sum()
    )
    average_stockout_rate = historical_loss_summary_df[
        "stockout_rate_pct"
    ].mean()

    kpis.extend(
        [
            {
                "KPI": "Tahmini tarihsel kayıp talep",
                "Değer": float(total_lost_demand),
            },
            {
                "KPI": "Kayıp yaşayan mağaza-ürün",
                "Değer": affected_series,
            },
            {
                "KPI": "Ortalama stokout oranı (%)",
                "Değer": float(average_stockout_rate),
            },
            {
                "KPI": "Tahmin ufku toplam talep",
                "Değer": float(demand_plan_df["planned_demand"].sum()),
            },
        ]
    )

    if demand_plan_df["current_stock"].notna().any():
        kpis.extend(
            [
                {
                    "KPI": "Riskli mağaza-ürün",
                    "Değer": int(
                        demand_plan_df["stockout_risk"]
                        .astype("boolean")
                        .fillna(False)
                        .sum()
                    ),
                },
                {
                    "KPI": "Önerilen toplam sevkiyat",
                    "Değer": float(
                        demand_plan_df["recommended_replenishment"].sum()
                    ),
                },
                {
                    "KPI": "Aksiyon alınmazsa beklenen açık",
                    "Değer": float(
                        demand_plan_df["expected_shortage_no_action"].sum()
                    ),
                },
            ]
        )

    if "estimated_lost_revenue" in historical_loss_summary_df.columns:
        kpis.append(
            {
                "KPI": "Tahmini tarihsel kayıp ciro",
                "Değer": float(
                    historical_loss_summary_df[
                        "estimated_lost_revenue"
                    ].sum()
                ),
            }
        )

    if "expected_lost_revenue_no_action" in demand_plan_df.columns:
        kpis.append(
            {
                "KPI": "Aksiyon alınmazsa beklenen kayıp ciro",
                "Değer": float(
                    demand_plan_df[
                        "expected_lost_revenue_no_action"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(kpis)


def management_recommendations(
    demand_plan_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Yöneticiye gösterilecek en önemli aksiyon satırlarını döndürür.
    """
    columns = [
        "priority",
        "store_id",
        "product_id",
        "planned_demand",
        "current_stock",
        "recommended_replenishment",
        "expected_shortage_no_action",
        "expected_stockout_date",
        "recommended_action",
    ]

    if "expected_lost_revenue_no_action" in demand_plan_df.columns:
        columns.insert(-1, "expected_lost_revenue_no_action")

    available_columns = [
        column for column in columns if column in demand_plan_df.columns
    ]

    return demand_plan_df[available_columns].head(top_n).reset_index(drop=True)


def plot_top_historical_losses(
    historical_loss_summary_df: pd.DataFrame,
    top_n: int = 15,
    quantity_label: str = "Birim",
) -> None:
    import matplotlib.pyplot as plt

    top = historical_loss_summary_df.nlargest(
        top_n, "estimated_lost_demand"
    ).copy()
    top["location"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values("estimated_lost_demand")

    plt.figure(figsize=(11, 7))
    plt.barh(top["location"], top["estimated_lost_demand"])
    plt.title("En Yüksek Tarihsel Kayıp Talep Noktaları")
    plt.xlabel(f"Tahmini kayıp talep ({quantity_label})")
    plt.ylabel("Mağaza / Ürün")
    plt.tight_layout()
    plt.show()


def plot_top_stockout_rates(
    historical_loss_summary_df: pd.DataFrame,
    top_n: int = 15,
) -> None:
    import matplotlib.pyplot as plt

    top = historical_loss_summary_df.nlargest(
        top_n, "stockout_rate_pct"
    ).copy()
    top["location"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values("stockout_rate_pct")

    plt.figure(figsize=(11, 7))
    plt.barh(top["location"], top["stockout_rate_pct"])
    plt.title("En Yüksek Stokout Oranına Sahip Noktalar")
    plt.xlabel("Stokout oranı (%)")
    plt.ylabel("Mağaza / Ürün")
    plt.tight_layout()
    plt.show()


def plot_top_demand_priorities(
    demand_plan_df: pd.DataFrame,
    top_n: int = 15,
    quantity_label: str = "Birim",
) -> None:
    import matplotlib.pyplot as plt

    top = demand_plan_df.nlargest(
        top_n, "priority_score"
    ).copy()
    top["location"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )

    value_column = (
        "recommended_replenishment"
        if top["recommended_replenishment"].notna().any()
        else "target_stock_for_horizon"
    )
    title = (
        "Önerilen Sevkiyat Öncelikleri"
        if value_column == "recommended_replenishment"
        else "Güncel Stok Girilmeden Önce Hedef Stok Öncelikleri"
    )

    top = top.sort_values(value_column)

    plt.figure(figsize=(11, 7))
    plt.barh(top["location"], top[value_column])
    plt.title(title)
    plt.xlabel(f"Miktar ({quantity_label})")
    plt.ylabel("Mağaza / Ürün")
    plt.tight_layout()
    plt.show()


def plot_risk_matrix(
    demand_plan_df: pd.DataFrame,
    quantity_label: str = "Birim",
) -> None:
    import matplotlib.pyplot as plt

    plot_df = demand_plan_df.copy()
    x = plot_df.get(
        "stockout_rate_pct",
        pd.Series(0, index=plot_df.index),
    ).fillna(0)
    y = plot_df["planned_demand"].fillna(0)

    if plot_df["expected_shortage_no_action"].notna().any():
        sizes = (
            plot_df["expected_shortage_no_action"].fillna(0)
            .rank(pct=True)
            .mul(300)
            .add(20)
        )
        size_label = "Balon büyüklüğü: beklenen açık"
    else:
        sizes = (
            plot_df["priority_score"].rank(pct=True)
            .mul(300)
            .add(20)
        )
        size_label = "Balon büyüklüğü: öncelik"

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, s=sizes, alpha=0.6)
    plt.title(f"Stokout–Talep Risk Matrisi\n{size_label}")
    plt.xlabel("Tarihsel stokout oranı (%)")
    plt.ylabel(f"Tahmin ufku talebi ({quantity_label})")
    plt.tight_layout()
    plt.show()


def plot_recommended_distribution(
    allocation_df: pd.DataFrame,
    top_n: int = 15,
    quantity_label: str = "Birim",
) -> None:
    import matplotlib.pyplot as plt

    _require_columns(
        allocation_df,
        {
            "store_id",
            "product_id",
            "recommended_distribution",
        },
        "allocation_df",
    )

    top = allocation_df.nlargest(
        top_n, "recommended_distribution"
    ).copy()
    top["location"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values("recommended_distribution")

    plt.figure(figsize=(11, 7))
    plt.barh(top["location"], top["recommended_distribution"])
    plt.title("Önerilen Ürün Dağıtım Planı")
    plt.xlabel(f"Dağıtılacak miktar ({quantity_label})")
    plt.ylabel("Mağaza / Ürün")
    plt.tight_layout()
    plt.show()

# ============================================================================
# Fiyat/Ciro Bazlı Öncelik ve ABC Analizi
# ============================================================================

def apply_revenue_weighted_priority(
    demand_plan_df: pd.DataFrame,
    *,
    future_revenue_risk_weight: float = 0.40,
    replenishment_value_weight: float = 0.25,
    historical_lost_revenue_weight: float = 0.20,
    stockout_rate_weight: float = 0.10,
    planned_revenue_weight: float = 0.05,
) -> pd.DataFrame:
    """
    Mağaza-ürün önceliğini yalnızca miktara göre değil, parasal etkiye göre
    yeniden hesaplar.

    Gerekli fiyat sütunu:
        latest_price veya price

    Oluşturulan başlıca sütunlar:
        planned_revenue
        recommended_replenishment_value
        expected_lost_revenue_no_action
        revenue_priority_score
        priority
        recommended_action
    """
    _require_columns(
        demand_plan_df,
        {
            "planned_demand",
            "recommended_replenishment",
            "expected_shortage_no_action",
        },
        "demand_plan_df",
    )

    raw_weights = np.array(
        [
            future_revenue_risk_weight,
            replenishment_value_weight,
            historical_lost_revenue_weight,
            stockout_rate_weight,
            planned_revenue_weight,
        ],
        dtype=float,
    )

    if np.any(raw_weights < 0) or np.isclose(raw_weights.sum(), 0):
        raise ValueError(
            "Fiyat bazlı öncelik ağırlıkları negatif olamaz ve toplamları "
            "sıfırdan büyük olmalıdır."
        )

    weights = raw_weights / raw_weights.sum()
    plan = demand_plan_df.copy()

    price_column = (
        "latest_price"
        if "latest_price" in plan.columns
        else "price"
        if "price" in plan.columns
        else None
    )

    if price_column is None:
        raise ValueError(
            "Fiyat bazlı öncelik için demand_plan_df içinde "
            "'latest_price' veya 'price' sütunu bulunmalıdır."
        )

    plan["latest_price"] = pd.to_numeric(
        plan[price_column],
        errors="coerce",
    )

    if plan["latest_price"].isna().any():
        raise ValueError("Plan tablosunda boş veya geçersiz fiyat değeri var.")
    if plan["latest_price"].lt(0).any():
        raise ValueError("Fiyat negatif olamaz.")

    plan["planned_revenue"] = (
        plan["planned_demand"].fillna(0)
        * plan["latest_price"]
    ).round(2)

    plan["recommended_replenishment_value"] = (
        plan["recommended_replenishment"].fillna(0)
        * plan["latest_price"]
    ).round(2)

    plan["expected_lost_revenue_no_action"] = (
        plan["expected_shortage_no_action"].fillna(0)
        * plan["latest_price"]
    ).round(2)

    if "estimated_lost_revenue" not in plan.columns:
        plan["estimated_lost_revenue"] = 0.0
    if "stockout_rate_pct" not in plan.columns:
        plan["stockout_rate_pct"] = 0.0

    components = pd.DataFrame(
        {
            "future_revenue_risk_rank": (
                plan["expected_lost_revenue_no_action"]
                .fillna(0)
                .rank(pct=True, method="average")
            ),
            "replenishment_value_rank": (
                plan["recommended_replenishment_value"]
                .fillna(0)
                .rank(pct=True, method="average")
            ),
            "historical_lost_revenue_rank": (
                plan["estimated_lost_revenue"]
                .fillna(0)
                .rank(pct=True, method="average")
            ),
            "stockout_rate_rank": (
                plan["stockout_rate_pct"]
                .fillna(0)
                .rank(pct=True, method="average")
            ),
            "planned_revenue_rank": (
                plan["planned_revenue"]
                .fillna(0)
                .rank(pct=True, method="average")
            ),
        },
        index=plan.index,
    )

    plan["revenue_priority_score"] = (
        components.to_numpy() @ weights * 100
    )

    score = plan["revenue_priority_score"]
    plan["priority"] = np.select(
        [
            score >= score.quantile(0.90),
            score >= score.quantile(0.70),
            score >= score.quantile(0.40),
        ],
        ["Kritik", "Yüksek", "Orta"],
        default="Düşük",
    )

    # Dağıtım fonksiyonu mevcut priority_score alanını kullandığı için
    # parasal öncelik skorunu ana skora taşıyoruz.
    plan["priority_score"] = plan["revenue_priority_score"]

    def _revenue_action(row: pd.Series) -> str:
        replenishment = row.get("recommended_replenishment")
        if pd.isna(replenishment):
            return (
                "Güncel stok girilmeli; fiyat bazlı stokout riski "
                "stok olmadan kesinleştirilemez."
            )

        amount = int(np.ceil(max(float(replenishment), 0)))
        replenishment_value = float(
            row.get("recommended_replenishment_value", 0) or 0
        )
        lost_revenue = float(
            row.get("expected_lost_revenue_no_action", 0) or 0
        )

        if amount > 0:
            return (
                f"{amount} birim sevk et "
                f"(yaklaşık {replenishment_value:,.2f} TL stok değeri); "
                f"aksiyon alınmazsa yaklaşık {lost_revenue:,.2f} TL "
                "ciro riski."
            )

        return "Mevcut stok yeterli; tahmin ufkunda parasal stokout riski düşük."

    plan["recommended_action"] = plan.apply(
        _revenue_action,
        axis=1,
    )

    return plan.sort_values(
        [
            "revenue_priority_score",
            "expected_lost_revenue_no_action",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_abc_analysis(
    df: pd.DataFrame,
    *,
    item_column: str = "product_id",
    value_column: str = "commercial_value",
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Kalemleri parasal katkıya göre A, B ve C sınıflarına ayırır.

    Varsayılan sınıflandırma:
        A: Kümülatif ticari değerin ilk %80'i
        B: Sonraki %15 (%80-%95)
        C: Kalan %5

    Sınırı aşan ürün, sınırı aşmadan önce bulunduğu sınıfta tutulur.
    Böylece ticari değeri yüksek tek bir ürün yanlışlıkla alt sınıfa düşmez.
    """
    _require_columns(
        df,
        {item_column, value_column},
        "abc_input_df",
    )

    if not 0 < a_threshold < b_threshold <= 1:
        raise ValueError(
            "ABC eşikleri 0 < a_threshold < b_threshold <= 1 "
            "koşulunu sağlamalıdır."
        )

    source = df[[item_column, value_column]].copy()
    source[value_column] = pd.to_numeric(
        source[value_column],
        errors="coerce",
    )

    if source[value_column].isna().any():
        raise ValueError(
            f"'{value_column}' sütununda boş veya sayısal olmayan değer var."
        )
    if source[value_column].lt(0).any():
        raise ValueError("ABC analizindeki ticari değer negatif olamaz.")

    abc = (
        source.groupby(item_column, as_index=False, dropna=False)
        .agg(total_value=(value_column, "sum"))
        .sort_values("total_value", ascending=False)
        .reset_index(drop=True)
    )

    total_value = float(abc["total_value"].sum())
    if total_value <= 0:
        raise ValueError("ABC analizi için toplam ticari değer sıfır.")

    abc["value_share_pct"] = (
        abc["total_value"] / total_value * 100
    )
    abc["cumulative_value_pct"] = abc[
        "value_share_pct"
    ].cumsum()

    previous_cumulative_share = (
        abc["cumulative_value_pct"]
        .div(100)
        .shift(1)
        .fillna(0)
    )

    abc["abc_class"] = np.select(
        [
            previous_cumulative_share < a_threshold,
            previous_cumulative_share < b_threshold,
        ],
        ["A", "B"],
        default="C",
    )

    abc["abc_description"] = abc["abc_class"].map(
        {
            "A": "Yüksek ticari öneme sahip ürün",
            "B": "Orta ticari öneme sahip ürün",
            "C": "Düşük ticari öneme sahip ürün",
        }
    )
    abc["value_rank"] = np.arange(1, len(abc) + 1)

    return abc


def build_historical_abc_analysis(
    historical_loss_summary_df: pd.DataFrame,
    *,
    item_column: str = "product_id",
    a_threshold: float = 0.80,
    b_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Tarihsel gerçekleşen ciro ile tahmini stokout kaybını birlikte kullanarak
    ürün ABC sınıflarını oluşturur.

    commercial_value =
        observed_revenue + estimated_lost_revenue
    """
    _require_columns(
        historical_loss_summary_df,
        {
            item_column,
            "observed_revenue",
            "estimated_lost_revenue",
        },
        "historical_loss_summary_df",
    )

    abc_input = historical_loss_summary_df.copy()
    abc_input["commercial_value"] = (
        pd.to_numeric(
            abc_input["observed_revenue"],
            errors="coerce",
        ).fillna(0)
        + pd.to_numeric(
            abc_input["estimated_lost_revenue"],
            errors="coerce",
        ).fillna(0)
    )

    return build_abc_analysis(
        abc_input,
        item_column=item_column,
        value_column="commercial_value",
        a_threshold=a_threshold,
        b_threshold=b_threshold,
    )


def add_abc_to_demand_plan(
    demand_plan_df: pd.DataFrame,
    abc_product_df: pd.DataFrame,
    *,
    item_column: str = "product_id",
) -> pd.DataFrame:
    """
    Ürün ABC sınıfını mağaza-ürün seviyesindeki demand planına ekler.
    """
    _require_columns(
        demand_plan_df,
        {item_column},
        "demand_plan_df",
    )
    _require_columns(
        abc_product_df,
        {
            item_column,
            "abc_class",
            "total_value",
            "value_share_pct",
            "cumulative_value_pct",
        },
        "abc_product_df",
    )

    plan = demand_plan_df.copy()
    abc = abc_product_df.copy()

    # Tür farklılıkları nedeniyle eşleşme kaybını önler.
    plan[item_column] = plan[item_column].astype(str)
    abc[item_column] = abc[item_column].astype(str)

    result = plan.merge(
        abc[
            [
                item_column,
                "abc_class",
                "abc_description",
                "total_value",
                "value_share_pct",
                "cumulative_value_pct",
            ]
        ].rename(
            columns={
                "total_value": "abc_product_total_value",
                "value_share_pct": "abc_value_share_pct",
                "cumulative_value_pct": "abc_cumulative_value_pct",
            }
        ),
        on=item_column,
        how="left",
        validate="many_to_one",
    )

    if result["abc_class"].isna().any():
        missing_count = int(result["abc_class"].isna().sum())
        raise ValueError(
            f"{missing_count} demand plan satırı ABC tablosuyla eşleşmedi."
        )

    return result


def add_abc_stockout_action(
    demand_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ABC sınıfı ile stokout riskini birleştirerek yönetim aksiyonu üretir.
    """
    _require_columns(
        demand_plan_df,
        {
            "abc_class",
            "stockout_risk",
            "recommended_replenishment",
        },
        "demand_plan_df",
    )

    plan = demand_plan_df.copy()
    stockout_risk = (
        plan["stockout_risk"]
        .astype("boolean")
        .fillna(False)
    )

    plan["abc_stock_action"] = np.select(
        [
            plan["abc_class"].eq("A") & stockout_risk,
            plan["abc_class"].eq("A"),
            plan["abc_class"].eq("B") & stockout_risk,
            plan["abc_class"].eq("C") & stockout_risk,
        ],
        [
            "Acil ikmal: yüksek değerli A ürünü stokout riski taşıyor.",
            "Yüksek servis seviyesiyle takip et.",
            "Planlı ikmal: B ürünü stokout riski taşıyor.",
            "Düşük ticari değer; ikmal maliyetiyle birlikte değerlendir.",
        ],
        default="Standart stok politikası uygula.",
    )

    abc_weight = plan["abc_class"].map(
        {"A": 15.0, "B": 7.5, "C": 0.0}
    ).fillna(0)

    base_score = pd.to_numeric(
        plan.get(
            "revenue_priority_score",
            plan.get("priority_score", 0),
        ),
        errors="coerce",
    ).fillna(0)

    plan["abc_adjusted_priority_score"] = (
        base_score + abc_weight
    )

    plan["operational_priority"] = np.select(
        [
            plan["abc_class"].eq("A") & stockout_risk,
            plan["abc_class"].eq("A"),
            plan["abc_class"].eq("B") & stockout_risk,
            stockout_risk,
        ],
        [
            "1-Acil",
            "2-Yüksek",
            "3-Planlı",
            "4-İzle",
        ],
        default="5-Standart",
    )

    return plan.sort_values(
        [
            "operational_priority",
            "abc_adjusted_priority_score",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_abc_management_summary(
    demand_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ABC sınıfı bazında talep, stokout, ikmal ve ciro riskini özetler.
    """
    _require_columns(
        demand_plan_df,
        {
            "abc_class",
            "product_id",
            "series_id",
            "planned_demand",
            "recommended_replenishment",
            "stockout_risk",
        },
        "demand_plan_df",
    )

    aggregations: dict[str, tuple[str, object]] = {
        "product_count": ("product_id", "nunique"),
        "store_product_count": ("series_id", "nunique"),
        "planned_demand": ("planned_demand", "sum"),
        "recommended_replenishment": (
            "recommended_replenishment",
            "sum",
        ),
        "risky_location_count": (
            "stockout_risk",
            lambda values: (
                values.astype("boolean")
                .fillna(False)
                .sum()
            ),
        ),
    }

    if "planned_revenue" in demand_plan_df.columns:
        aggregations["planned_revenue"] = (
            "planned_revenue",
            "sum",
        )
    if "recommended_replenishment_value" in demand_plan_df.columns:
        aggregations["recommended_replenishment_value"] = (
            "recommended_replenishment_value",
            "sum",
        )
    if "expected_lost_revenue_no_action" in demand_plan_df.columns:
        aggregations["expected_lost_revenue_no_action"] = (
            "expected_lost_revenue_no_action",
            "sum",
        )

    summary = (
        demand_plan_df.groupby(
            "abc_class",
            as_index=False,
            dropna=False,
        )
        .agg(**aggregations)
    )

    class_order = pd.CategoricalDtype(
        categories=["A", "B", "C"],
        ordered=True,
    )
    summary["abc_class"] = summary["abc_class"].astype(
        class_order
    )

    return summary.sort_values("abc_class").reset_index(drop=True)


def plot_abc_curve(
    abc_product_df: pd.DataFrame,
) -> None:
    """
    Ürün ticari değeri ile kümülatif değer yüzdesini gösterir.
    """
    import matplotlib.pyplot as plt

    _require_columns(
        abc_product_df,
        {
            "value_rank",
            "total_value",
            "cumulative_value_pct",
        },
        "abc_product_df",
    )

    plot_df = abc_product_df.sort_values("value_rank").copy()

    fig, axis_value = plt.subplots(figsize=(13, 6))
    axis_value.bar(
        plot_df["value_rank"],
        plot_df["total_value"],
    )
    axis_value.set_xlabel("Ticari değere göre ürün sırası")
    axis_value.set_ylabel("Ürün ticari değeri (TL)")
    axis_value.set_title("ABC Analizi — Ürün Değeri ve Kümülatif Katkı")

    axis_cumulative = axis_value.twinx()
    axis_cumulative.plot(
        plot_df["value_rank"],
        plot_df["cumulative_value_pct"],
        marker="o",
        markersize=2,
    )
    axis_cumulative.axhline(80, linestyle="--")
    axis_cumulative.axhline(95, linestyle="--")
    axis_cumulative.set_ylabel("Kümülatif ticari değer (%)")
    axis_cumulative.set_ylim(0, 105)

    fig.tight_layout()
    plt.show()


def plot_abc_class_summary(
    abc_product_df: pd.DataFrame,
) -> None:
    """
    A, B ve C sınıflarının toplam ticari değerini gösterir.
    """
    import matplotlib.pyplot as plt

    _require_columns(
        abc_product_df,
        {
            "abc_class",
            "total_value",
        },
        "abc_product_df",
    )

    summary = (
        abc_product_df.groupby(
            "abc_class",
            as_index=False,
            observed=True,
        )
        .agg(
            product_count=("abc_class", "size"),
            total_value=("total_value", "sum"),
        )
    )

    order = pd.CategoricalDtype(
        categories=["A", "B", "C"],
        ordered=True,
    )
    summary["abc_class"] = summary["abc_class"].astype(order)
    summary = summary.sort_values("abc_class")

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary["abc_class"].astype(str),
        summary["total_value"],
    )
    plt.title("ABC Sınıflarının Toplam Ticari Değeri")
    plt.xlabel("ABC sınıfı")
    plt.ylabel("Toplam ticari değer (TL)")
    plt.tight_layout()
    plt.show()


def plot_top_lost_revenue(
    historical_loss_summary_df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> None:
    """
    En yüksek tarihsel stokout ciro kaybını gösterir.
    """
    import matplotlib.pyplot as plt

    _require_columns(
        historical_loss_summary_df,
        {
            "store_id",
            "product_id",
            "estimated_lost_revenue",
        },
        "historical_loss_summary_df",
    )

    top = historical_loss_summary_df.nlargest(
        top_n,
        "estimated_lost_revenue",
    ).copy()
    top["location"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values("estimated_lost_revenue")

    plt.figure(figsize=(11, 7))
    plt.barh(
        top["location"],
        top["estimated_lost_revenue"],
    )
    plt.title("En Yüksek Tarihsel Stokout Ciro Kaybı")
    plt.xlabel("Tahmini kayıp ciro (TL)")
    plt.ylabel("Mağaza / Ürün")
    plt.tight_layout()
    plt.show()


def plot_top_revenue_risk(
    demand_plan_df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> None:
    """
    Aksiyon alınmazsa en yüksek tahmini ciro riskini gösterir.
    """
    import matplotlib.pyplot as plt

    _require_columns(
        demand_plan_df,
        {
            "store_id",
            "product_id",
            "expected_lost_revenue_no_action",
        },
        "demand_plan_df",
    )

    top = demand_plan_df.nlargest(
        top_n,
        "expected_lost_revenue_no_action",
    ).copy()
    top["location"] = (
        top["store_id"].astype(str)
        + " / "
        + top["product_id"].astype(str)
    )
    top = top.sort_values(
        "expected_lost_revenue_no_action"
    )

    plt.figure(figsize=(11, 7))
    plt.barh(
        top["location"],
        top["expected_lost_revenue_no_action"],
    )
    plt.title("Aksiyon Alınmazsa En Yüksek Beklenen Ciro Riski")
    plt.xlabel("Beklenen ciro riski (TL)")
    plt.ylabel("Mağaza / Ürün")
    plt.tight_layout()
    plt.show()
