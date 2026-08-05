from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import gc
import glob
import os
import time

import numpy as np
import pandas as pd

def _get_torch():
    # PyTorch büyük bir bağımlılıktır. Uygulama açılırken değil,
    # yalnızca kullanıcı tahmin modelini çalıştırdığında yüklenir.
    import torch
    return torch



DataFrequency = Literal["hourly", "daily", "monthly"]

_FREQUENCY_TO_PANDAS: dict[DataFrequency, str] = {
    "hourly": "h",
    "daily": "D",
    "monthly": "MS",
}

_FREQUENCY_LABELS_TR: dict[DataFrequency, str] = {
    "hourly": "Saatlik",
    "daily": "Günlük",
    "monthly": "Aylık",
}



class DataValidationError(ValueError):
    """Kullanıcı arayüzünde doğrudan gösterilebilecek veri doğrulama hatası."""


@dataclass(frozen=True)
class ColumnMapping:
    """
    Kullanıcının yüklediği dosyadaki sütun adlarını standart şemaya bağlar.

    Zorunlu alanlar:
        date, store, product, sales, stock

    Opsiyonel alanlar:
        price, category_1, category_2, category_3, stockout_flag
    """

    date: str
    store: str
    product: str
    sales: str
    stock: str
    price: Optional[str] = None
    category_1: Optional[str] = None
    category_2: Optional[str] = None
    category_3: Optional[str] = None
    stockout_flag: Optional[str] = None

    def standardised_rename_map(self) -> dict[str, str]:
        rename_map = {
            self.date: "date",
            self.store: "store_id",
            self.product: "product_id",
            self.sales: "sales",
            self.stock: "stock",
        }
        optional_mapping = {
            self.price: "price",
            self.category_1: "category_1",
            self.category_2: "category_2",
            self.category_3: "category_3",
            self.stockout_flag: "provided_stockout_flag",
        }
        rename_map.update(
            {
                source: target
                for source, target in optional_mapping.items()
                if source is not None
            }
        )
        return rename_map

    def selected_source_columns(self) -> list[str]:
        return list(self.standardised_rename_map().keys())


class DemandDataPipeline:
    """
    Farklı sütun adlarına sahip perakende veri setlerini ortak şemaya çevirir.

    Standart çıktı şeması:
        date, store_id, product_id, sales, stock, series_id,
        is_stockout, stockout_reason

    Seçilmişse ayrıca:
        price, category_1, category_2, category_3,
        provided_stockout_flag
    """

    _TRUE_FLAG_VALUES = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "evet",
        "e",
        "stockout",
        "stock_out",
        "out_of_stock",
        "oos",
    }
    _FALSE_FLAG_VALUES = {
        "0",
        "false",
        "f",
        "no",
        "n",
        "hayir",
        "hayır",
        "h",
        "in_stock",
        "available",
        "none",
    }

    def __init__(
        self,
        mapping: ColumnMapping,
        frequency: DataFrequency,
        dayfirst: bool = False,
        duplicate_policy: Literal["error", "aggregate"] = "aggregate",
        date_gap_policy: Literal["error", "warn", "ignore"] = "warn",
        negative_value_policy: Literal["error", "clip"] = "clip",
        use_sales_equals_stock_rule: bool = False,
        stockout_tolerance: float = 0.0,
        stockout_stock_threshold: float = 0.0,
        combine_provided_flag_with_inferred: bool = True,
        stock_timing: Literal["end_of_period", "start_of_period"] = "end_of_period",
        imputation_window: int = 28,
        min_history: int = 7,
        imputation_statistic: Literal["median", "mean"] = "median",
    ) -> None:
        if frequency not in _FREQUENCY_TO_PANDAS:
            raise ValueError(
                "frequency 'hourly', 'daily' veya 'monthly' olmalıdır."
            )
        if duplicate_policy not in {"error", "aggregate"}:
            raise ValueError("duplicate_policy 'error' veya 'aggregate' olmalıdır.")
        if date_gap_policy not in {"error", "warn", "ignore"}:
            raise ValueError("date_gap_policy 'error', 'warn' veya 'ignore' olmalıdır.")
        if negative_value_policy not in {"error", "clip"}:
            raise ValueError("negative_value_policy 'error' veya 'clip' olmalıdır.")
        if stock_timing not in {"end_of_period", "start_of_period"}:
            raise ValueError(
                "stock_timing 'end_of_period' veya 'start_of_period' olmalıdır."
            )
        if imputation_statistic not in {"median", "mean"}:
            raise ValueError("imputation_statistic 'median' veya 'mean' olmalıdır.")
        if imputation_window <= 0:
            raise ValueError("imputation_window pozitif olmalıdır.")
        if min_history <= 0:
            raise ValueError("min_history pozitif olmalıdır.")

        self.mapping = mapping
        self.frequency = frequency
        self.pandas_freq = _FREQUENCY_TO_PANDAS[frequency]
        # Eski kullanım noktalarıyla uyumluluk için iç alias.
        self.freq = self.pandas_freq
        self.dayfirst = dayfirst
        self.duplicate_policy = duplicate_policy
        self.date_gap_policy = date_gap_policy
        self.negative_value_policy = negative_value_policy
        self.use_sales_equals_stock_rule = use_sales_equals_stock_rule
        self.stockout_tolerance = float(stockout_tolerance)
        self.stockout_stock_threshold = float(stockout_stock_threshold)
        self.combine_provided_flag_with_inferred = combine_provided_flag_with_inferred
        self.stock_timing = stock_timing
        self.imputation_window = int(imputation_window)
        self.min_history = int(min_history)
        self.imputation_statistic = imputation_statistic
        self.report: dict[str, object] = {"warnings": [], "stats": {}}

    def _warnings(self) -> list[str]:
        warnings = self.report["warnings"]
        assert isinstance(warnings, list)
        return warnings

    def _stats(self) -> dict[str, object]:
        stats = self.report["stats"]
        assert isinstance(stats, dict)
        return stats

    @property
    def frequency_label_tr(self) -> str:
        return _FREQUENCY_LABELS_TR[self.frequency]

    def _normalise_timestamps(self, dates: pd.Series) -> pd.Series:
        """Seçilen frekansa göre tarihleri ortak zaman adımlarına hizalar."""
        if self.frequency == "hourly":
            return dates.dt.floor("h")
        if self.frequency == "daily":
            return dates.dt.normalize()
        # Aylık veri, ayın ilk gününe hizalanır.
        return dates.dt.to_period("M").dt.to_timestamp(how="start")

    def _validate_mapping(self, raw_df: pd.DataFrame) -> None:
        selected_columns = self.mapping.selected_source_columns()

        blank_columns = [
            column
            for column in selected_columns
            if not isinstance(column, str) or not column.strip()
        ]
        if blank_columns:
            raise DataValidationError("Seçilen sütun adları boş olamaz.")

        duplicate_mappings = sorted(
            {
                column
                for column in selected_columns
                if selected_columns.count(column) > 1
            }
        )
        if duplicate_mappings:
            raise DataValidationError(
                "Aynı kaynak sütunu birden fazla role bağlanamaz. "
                f"Tekrarlanan sütunlar: {duplicate_mappings}"
            )

        missing = [column for column in selected_columns if column not in raw_df.columns]
        if missing:
            raise DataValidationError(f"Dosyada bulunmayan seçilmiş sütunlar: {missing}")

    @staticmethod
    def _clean_id_column(series: pd.Series, standard_name: str) -> pd.Series:
        missing_mask = series.isna()
        if missing_mask.any():
            rows = series.index[missing_mask].tolist()[:5]
            raise DataValidationError(
                f"'{standard_name}' boş değer içeriyor. Örnek satırlar: {rows}"
            )

        cleaned = series.astype(str).str.strip()
        empty_mask = cleaned.eq("")
        if empty_mask.any():
            rows = cleaned.index[empty_mask].tolist()[:5]
            raise DataValidationError(
                f"'{standard_name}' boş metin içeriyor. Örnek satırlar: {rows}"
            )
        return cleaned

    @staticmethod
    def _to_required_numeric(
        series: pd.Series,
        original_name: str,
    ) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        invalid_mask = numeric.isna()
        if invalid_mask.any():
            examples = series.loc[invalid_mask].astype(str).head(5).tolist()
            raise DataValidationError(
                f"'{original_name}' sayısala çevrilemedi veya boş değer içeriyor. "
                f"Örnekler: {examples}"
            )
        return numeric.astype(float)

    def _handle_negative_values(
        self,
        df: pd.DataFrame,
        column: str,
        original_name: str,
    ) -> int:
        negative_mask = df[column].lt(0)
        negative_count = int(negative_mask.sum())
        if not negative_count:
            return 0

        examples = df.loc[negative_mask, column].head(5).tolist()
        if self.negative_value_policy == "error":
            raise DataValidationError(
                f"'{original_name}' negatif değer içeriyor. "
                f"Sayı: {negative_count}, örnekler: {examples}"
            )

        df[column] = df[column].clip(lower=0)
        self._warnings().append(
            f"'{original_name}' sütunundaki {negative_count} negatif değer 0'a çekildi."
        )
        return negative_count

    def _parse_optional_price(self, df: pd.DataFrame) -> None:
        if "price" not in df.columns:
            return

        original_name = self.mapping.price or "price"
        numeric = pd.to_numeric(df["price"], errors="coerce")
        invalid_count = int(numeric.isna().sum())
        if invalid_count:
            self._warnings().append(
                f"Opsiyonel fiyat sütununda {invalid_count} eksik veya geçersiz değer var. "
                "Bu değerler NaN olarak bırakıldı."
            )
        df["price"] = numeric.astype(float)

        negative_mask = df["price"].lt(0)
        negative_count = int(negative_mask.sum())
        if negative_count:
            if self.negative_value_policy == "error":
                examples = df.loc[negative_mask, "price"].head(5).tolist()
                raise DataValidationError(
                    f"'{original_name}' negatif fiyat içeriyor. Örnekler: {examples}"
                )
            df.loc[negative_mask, "price"] = 0.0
            self._warnings().append(
                f"Opsiyonel fiyat sütunundaki {negative_count} negatif değer 0'a çekildi."
            )

    def _parse_categories(self, df: pd.DataFrame) -> None:
        for column in ("category_1", "category_2", "category_3"):
            if column not in df.columns:
                continue
            cleaned = df[column].astype("string").str.strip()
            cleaned = cleaned.mask(cleaned.eq(""), pd.NA)
            missing_count = int(cleaned.isna().sum())
            if missing_count:
                self._warnings().append(
                    f"'{column}' sütununda {missing_count} boş değer var."
                )
            df[column] = cleaned

    def _parse_stockout_flag(self, series: pd.Series) -> pd.Series:
        result = pd.Series(False, index=series.index, dtype=bool)
        non_missing_mask = series.notna()
        missing_count = int((~non_missing_mask).sum())
        if missing_count:
            self._warnings().append(
                f"Opsiyonel stokout flag sütununda {missing_count} boş değer var. "
                "Bu satırlarda stok miktarından türetilen kural kullanılacak."
            )

        non_missing = series.loc[non_missing_mask]
        if non_missing.empty:
            return result

        if pd.api.types.is_bool_dtype(non_missing):
            result.loc[non_missing_mask] = non_missing.astype(bool)
            return result

        numeric = pd.to_numeric(non_missing, errors="coerce")
        if numeric.notna().all():
            result.loc[non_missing_mask] = numeric.gt(0).to_numpy()
            return result

        normalised = (
            non_missing.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        known_mask = normalised.isin(
            self._TRUE_FLAG_VALUES.union(self._FALSE_FLAG_VALUES)
        )
        if not known_mask.all():
            unknown_examples = normalised.loc[~known_mask].drop_duplicates().head(10).tolist()
            raise DataValidationError(
                "Stokout flag sütununda yorumlanamayan değerler var. "
                f"Örnekler: {unknown_examples}. 0/1, True/False, Yes/No veya OOS kullanın."
            )

        result.loc[non_missing.index] = normalised.isin(
            self._TRUE_FLAG_VALUES
        ).to_numpy()
        return result

    @staticmethod
    def _last_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[-1] if not non_null.empty else np.nan

    @staticmethod
    def _mode_or_last(series: pd.Series):
        non_null = series.dropna()
        if non_null.empty:
            return pd.NA
        modes = non_null.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else non_null.iloc[-1]

    @staticmethod
    def _weighted_price(group: pd.DataFrame) -> float:
        valid = group["price"].notna()
        if not valid.any():
            return np.nan
        prices = group.loc[valid, "price"].astype(float)
        weights = group.loc[valid, "sales"].clip(lower=0).astype(float)
        if weights.sum() > 0:
            return float(np.average(prices, weights=weights))
        return float(prices.mean())

    def _aggregate_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = ["date", "store_id", "product_id"]
        duplicate_mask = df.duplicated(keys, keep=False)
        duplicate_row_count = int(duplicate_mask.sum())
        duplicate_group_count = int(df.loc[duplicate_mask, keys].drop_duplicates().shape[0])
        self._stats()["duplicate_row_count"] = duplicate_row_count
        self._stats()["duplicate_group_count"] = duplicate_group_count

        if not duplicate_row_count:
            return df

        examples = df.loc[duplicate_mask, keys].head(5).to_dict("records")
        if self.duplicate_policy == "error":
            raise DataValidationError(
                "Aynı tarih-mağaza-ürün için birden fazla kayıt var. "
                f"Örnekler: {examples}"
            )

        rows: list[dict[str, object]] = []
        optional_columns = [
            column
            for column in (
                "price",
                "category_1",
                "category_2",
                "category_3",
                "provided_stockout_flag",
            )
            if column in df.columns
        ]

        for key_values, group in df.groupby(keys, sort=False, dropna=False):
            date, store_id, product_id = key_values
            row: dict[str, object] = {
                "date": date,
                "store_id": store_id,
                "product_id": product_id,
                "sales": float(group["sales"].sum()),
                "stock": float(self._last_non_null(group["stock"])),
            }
            if "price" in optional_columns:
                row["price"] = self._weighted_price(group)
            for category_column in ("category_1", "category_2", "category_3"):
                if category_column in optional_columns:
                    row[category_column] = self._mode_or_last(group[category_column])
            if "provided_stockout_flag" in optional_columns:
                row["provided_stockout_flag"] = bool(
                    group["provided_stockout_flag"].fillna(False).astype(bool).any()
                )
            rows.append(row)

        self._warnings().append(
            f"{duplicate_row_count} tekrar satırı {duplicate_group_count} tarih-mağaza-ürün "
            "grubunda birleştirildi. Satış toplandı, stok son değer olarak alındı."
        )
        return pd.DataFrame(rows)

    def _derive_stockout_columns(self, df: pd.DataFrame) -> None:
        stock_zero = df["stock"].le(self.stockout_stock_threshold)
        sales_equal_stock = (
            df["stock"].gt(self.stockout_stock_threshold)
            & np.isclose(
                df["sales"].astype(float),
                df["stock"].astype(float),
                atol=self.stockout_tolerance,
                rtol=0.0,
            )
        )

        inferred = stock_zero.copy()
        if self.use_sales_equals_stock_rule:
            inferred = inferred | sales_equal_stock

        provided = (
            df["provided_stockout_flag"].fillna(False).astype(bool)
            if "provided_stockout_flag" in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )

        if "provided_stockout_flag" in df.columns:
            if self.combine_provided_flag_with_inferred:
                df["is_stockout"] = provided | inferred
            else:
                df["is_stockout"] = provided
        else:
            df["is_stockout"] = inferred

        reasons = pd.Series("", index=df.index, dtype="string")
        reasons = reasons.mask(provided, reasons + "provided_flag|")
        reasons = reasons.mask(stock_zero, reasons + "stock_zero|")
        if self.use_sales_equals_stock_rule:
            reasons = reasons.mask(sales_equal_stock, reasons + "sales_equal_stock|")
        df["stockout_reason"] = reasons.str.rstrip("|").replace("", "none")

    def _check_date_gaps(self, df: pd.DataFrame) -> None:
        if self.date_gap_policy == "ignore":
            self._stats()["missing_date_count"] = np.nan
            return

        total_missing_dates = 0
        affected_series = 0
        examples: list[dict[str, object]] = []

        for series_id, group in df.groupby("series_id", sort=False):
            dates = pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
            if len(dates) < 2:
                continue
            expected = pd.date_range(dates.min(), dates.max(), freq=self.freq)
            missing = expected.difference(dates)
            if len(missing):
                affected_series += 1
                total_missing_dates += len(missing)
                if len(examples) < 5:
                    examples.append(
                        {
                            "series_id": series_id,
                            "missing_date_count": len(missing),
                            "first_missing_date": missing.min(),
                        }
                    )

        self._stats()["missing_date_count"] = total_missing_dates
        self._stats()["series_with_date_gaps"] = affected_series

        if not total_missing_dates:
            return

        message = (
            f"{affected_series} seride toplam {total_missing_dates} eksik zaman adımı var. "
            f"Örnekler: {examples}"
        )
        if self.date_gap_policy == "error":
            raise DataValidationError(
                message
                + ". Modeller düzenli zaman adımları beklediği için devam edilmedi."
            )
        self._warnings().append(message)

    def prepare(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Yüklenen ham veriyi doğrular ve standart şemaya çevirir."""
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df bir pandas DataFrame olmalıdır.")
        if raw_df.empty:
            raise DataValidationError("Yüklenen veri boş.")

        self.report = {"warnings": [], "stats": {}}
        self._validate_mapping(raw_df)

        rename_map = self.mapping.standardised_rename_map()
        df = raw_df[list(rename_map.keys())].copy().rename(columns=rename_map)

        parsed_dates = pd.to_datetime(
            df["date"], errors="coerce", dayfirst=self.dayfirst
        )
        invalid_date_mask = parsed_dates.isna()
        if invalid_date_mask.any():
            examples = df.loc[invalid_date_mask, "date"].astype(str).head(5).tolist()
            raise DataValidationError(
                "Tarih sütunu datetime'a çevrilemedi. "
                f"Örnek değerler: {examples}"
            )
        df["date"] = self._normalise_timestamps(parsed_dates)

        df["store_id"] = self._clean_id_column(df["store_id"], "store_id")
        df["product_id"] = self._clean_id_column(df["product_id"], "product_id")

        df["sales"] = self._to_required_numeric(df["sales"], self.mapping.sales)
        df["stock"] = self._to_required_numeric(df["stock"], self.mapping.stock)

        negative_sales_count = self._handle_negative_values(
            df, "sales", self.mapping.sales
        )
        negative_stock_count = self._handle_negative_values(
            df, "stock", self.mapping.stock
        )

        self._parse_optional_price(df)
        self._parse_categories(df)

        if "provided_stockout_flag" in df.columns:
            df["provided_stockout_flag"] = self._parse_stockout_flag(
                df["provided_stockout_flag"]
            )

        df = self._aggregate_duplicates(df)
        df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

        df["series_id"] = df["store_id"] + "||" + df["product_id"]
        self._derive_stockout_columns(df)

        self._stats().update(
            {
                "row_count": int(len(df)),
                "series_count": int(df["series_id"].nunique()),
                "store_count": int(df["store_id"].nunique()),
                "product_count": int(df["product_id"].nunique()),
                "date_min": df["date"].min(),
                "date_max": df["date"].max(),
                "negative_sales_count": negative_sales_count,
                "negative_stock_count": negative_stock_count,
                "stockout_count": int(df["is_stockout"].sum()),
                "stockout_ratio_pct": round(float(df["is_stockout"].mean() * 100), 2),
                "price_provided": "price" in df.columns,
                "category_count": sum(
                    column in df.columns
                    for column in ("category_1", "category_2", "category_3")
                ),
                "stockout_flag_provided": "provided_stockout_flag" in df.columns,
                "stock_timing": self.stock_timing,
                "frequency": self.frequency,
                "frequency_label": self.frequency_label_tr,
                "pandas_frequency": self.pandas_freq,
            }
        )

        self._check_date_gaps(df)
        return df

    def impute_stockouts(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """
        Stokout günlerindeki gözlenen satışı yalnızca geçmiş stokout olmayan
        satışlardan hesaplanan nedensel tahminle yükseltir.
        """
        required = {"date", "series_id", "sales", "is_stockout"}
        missing = required.difference(prepared_df.columns)
        if missing:
            raise DataValidationError(
                f"Stokout düzeltmesi için eksik sütunlar: {sorted(missing)}"
            )

        def impute_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("date").copy()
            normal_sales = group["sales"].where(~group["is_stockout"])
            shifted = normal_sales.shift(1)

            rolling = shifted.rolling(
                window=self.imputation_window,
                min_periods=self.min_history,
            )
            expanding = shifted.expanding(min_periods=1)
            if self.imputation_statistic == "median":
                rolling_estimate = rolling.median()
                fallback_estimate = expanding.median()
            else:
                rolling_estimate = rolling.mean()
                fallback_estimate = expanding.mean()

            replacement = rolling_estimate.fillna(fallback_estimate)
            group["demand_adjusted"] = group["sales"].astype(float)
            valid_replacement = (
                group["is_stockout"]
                & replacement.notna()
                & replacement.gt(group["sales"])
            )
            group.loc[valid_replacement, "demand_adjusted"] = replacement.loc[
                valid_replacement
            ]
            group["demand_was_imputed"] = valid_replacement
            group["demand_adjustment"] = (
                group["demand_adjusted"] - group["sales"]
            )
            return group

        parts = [
            impute_group(group)
            for _, group in prepared_df.groupby("series_id", sort=False)
        ]
        if not parts:
            raise DataValidationError("İşlenecek mağaza-ürün serisi bulunamadı.")
        return (
            pd.concat(parts, ignore_index=True)
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )

    def make_model_context(
        self,
        prepared_df: pd.DataFrame,
        adjust_stockouts: bool = True,
    ) -> pd.DataFrame:
        source = (
            self.impute_stockouts(prepared_df)
            if adjust_stockouts
            else prepared_df.assign(demand_adjusted=prepared_df["sales"].astype(float))
        )
        return (
            source[["series_id", "date", "demand_adjusted"]]
            .rename(columns={"demand_adjusted": "target"})
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )

    def latest_available_stock(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        required = {"series_id", "date", "stock", "sales"}
        missing = required.difference(prepared_df.columns)
        if missing:
            raise DataValidationError(
                f"Güncel stok hesabı için eksik sütunlar: {sorted(missing)}"
            )

        latest = (
            prepared_df.sort_values("date")
            .groupby("series_id", as_index=False)
            .tail(1)
            .copy()
        )
        if self.stock_timing == "end_of_period":
            latest["opening_stock"] = latest["stock"].astype(float)
        else:
            latest["opening_stock"] = (
                latest["stock"].astype(float) - latest["sales"].astype(float)
            ).clip(lower=0)

        keep = ["series_id", "opening_stock"]
        for column in (
            "store_id",
            "product_id",
            "category_1",
            "category_2",
            "category_3",
        ):
            if column in latest.columns:
                keep.append(column)
        return latest[keep].reset_index(drop=True)


class BaseZeroShotForecaster(ABC):
    model_id: str
    model_name: str

    @abstractmethod
    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        raise NotImplementedError


def validate_model_context(context_df: pd.DataFrame) -> pd.DataFrame:
    required = {"series_id", "date", "target"}
    missing = required.difference(context_df.columns)
    if missing:
        raise ValueError(f"Model girdisinde eksik sütunlar: {sorted(missing)}")

    result = context_df[["series_id", "date", "target"]].copy()
    result["series_id"] = result["series_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["target"] = pd.to_numeric(result["target"], errors="coerce").astype(float)
    if result["target"].isna().any():
        raise ValueError("Model girdisinde eksik veya geçersiz target değeri var.")
    if np.isinf(result["target"]).any():
        raise ValueError("Model girdisinde sonsuz target değeri var.")
    result["target"] = result["target"].clip(lower=0)

    duplicate_mask = result.duplicated(["series_id", "date"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Aynı seri ve tarih için birden fazla model girdisi var.")
    return result.sort_values(["series_id", "date"]).reset_index(drop=True)


def create_future_dates(
    last_date: pd.Timestamp,
    horizon: int,
    freq: str,
) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=last_date + offset, periods=horizon, freq=freq)


class ChronosForecaster(BaseZeroShotForecaster):
    """Chronos modellerini tek nokta tahmini çıktısına uyarlayan adapter."""

    def __init__(self, model_id: str, device: Optional[str] = None) -> None:
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        torch = _get_torch()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model_id == "amazon/chronos-2":
            from chronos import Chronos2Pipeline

            pipeline_class = Chronos2Pipeline
        else:
            from chronos import BaseChronosPipeline

            pipeline_class = BaseChronosPipeline

        print(f"{self.model_name} yükleniyor | device={self.device}")
        self.pipeline = pipeline_class.from_pretrained(
            model_id,
            device_map=self.device,
        )

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon <= 0:
            raise ValueError("Tahmin ufku pozitif olmalıdır.")

        context = validate_model_context(context_df)
        prediction_df = self.pipeline.predict_df(
            context,
            prediction_length=horizon,
            # Kullanıcıya tahmin aralıkları sunulmadığı için yalnızca merkez tahmin istenir.
            quantile_levels=[0.5],
            id_column="series_id",
            timestamp_column="date",
            target="target",
        )

        result = prediction_df.copy()
        if "predictions" not in result.columns:
            if "0.5" in result.columns:
                result["predictions"] = result["0.5"]
            elif 0.5 in result.columns:
                result["predictions"] = result[0.5]
            else:
                raise RuntimeError(
                    "Chronos çıktısında merkez tahmin sütunu bulunamadı."
                )

        result["predictions"] = pd.to_numeric(
            result["predictions"], errors="coerce"
        ).clip(lower=0)
        if result["predictions"].isna().any():
            raise RuntimeError("Chronos çıktısında NaN tahmin değeri var.")

        result["model"] = self.model_name
        keep_columns = ["series_id", "date", "predictions", "model"]
        return (
            result[keep_columns]
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )


class TimesFMForecaster(BaseZeroShotForecaster):
    """TimesFM 2.5 nokta tahmin adapter'ı."""

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        freq: str = "D",
        batch_size: int = 16,
        max_context: int = 1024,
        max_horizon: int = 256,
    ) -> None:
        import timesfm

        self.model_id = model_id
        self.model_name = "TimesFM-2.5"
        # TimesFM 2.5 modele bir frekans göstergesi vermez; bu değer yalnızca
        # gelecekteki timestamp'leri oluşturmak için tutulur.
        self.freq = freq
        self.batch_size = int(batch_size)
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)

        torch = _get_torch()
        torch.set_float32_matmul_precision("high")
        print(f"{self.model_name} yükleniyor | checkpoint={self.model_id}")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=False,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=False,
            )
        )
        print(f"{self.model_name} hazır.")

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon <= 0:
            raise ValueError("Tahmin ufku pozitif olmalıdır.")
        if horizon > self.max_horizon:
            raise ValueError(
                f"TimesFM en fazla {self.max_horizon} dönem tahmin üretebilir. "
                f"İstenen: {horizon}"
            )

        context = validate_model_context(context_df)
        grouped_series = list(context.groupby("series_id", sort=False))
        if not grouped_series:
            raise ValueError("Tahmin edilecek seri bulunamadı.")

        result_parts: list[pd.DataFrame] = []
        for batch_start in range(0, len(grouped_series), self.batch_size):
            batch_groups = grouped_series[batch_start : batch_start + self.batch_size]
            model_inputs: list[np.ndarray] = []

            for _, group in batch_groups:
                values = (
                    group.sort_values("date")["target"]
                    .astype(float)
                    .to_numpy(dtype=np.float32)
                )[-self.max_context :]
                model_inputs.append(values)

            point_forecast, _ = self.model.forecast(
                horizon=horizon,
                inputs=model_inputs,
            )
            point_forecast = np.asarray(point_forecast, dtype=float)

            expected_shape = (len(batch_groups), horizon)
            if point_forecast.shape != expected_shape:
                raise RuntimeError(
                    "TimesFM beklenmeyen point forecast boyutu üretti. "
                    f"Gelen: {point_forecast.shape}, beklenen: {expected_shape}"
                )

            for batch_index, (series_id, group) in enumerate(batch_groups):
                last_date = pd.to_datetime(group["date"]).max()
                future_dates = create_future_dates(last_date, horizon, self.freq)
                predictions = np.clip(point_forecast[batch_index], 0, None)
                result_parts.append(
                    pd.DataFrame(
                        {
                            "series_id": series_id,
                            "date": future_dates,
                            "predictions": predictions,
                            "model": self.model_name,
                        }
                    )
                )

        result = pd.concat(result_parts, ignore_index=True)
        result["predictions"] = pd.to_numeric(
            result["predictions"], errors="coerce"
        ).clip(lower=0)
        if result["predictions"].isna().any():
            raise RuntimeError("TimesFM çıktısında NaN tahmin değeri var.")
        return result.sort_values(["series_id", "date"]).reset_index(drop=True)


def calculate_forecast_metrics(evaluation_df: pd.DataFrame) -> dict[str, float | int]:
    required = {"actual", "predictions", "is_stockout", "series_id"}
    missing = required.difference(evaluation_df.columns)
    if missing:
        raise ValueError(f"Metrik hesabında eksik sütunlar: {sorted(missing)}")

    scoring_mask = (
        ~evaluation_df["is_stockout"].fillna(False).astype(bool)
        & evaluation_df["actual"].notna()
        & evaluation_df["predictions"].notna()
    )
    scored = evaluation_df.loc[scoring_mask].copy()
    if scored.empty:
        raise ValueError("Skorlanabilecek stokout olmayan doğrulama gözlemi yok.")

    actual = scored["actual"].astype(float)
    prediction = scored["predictions"].astype(float)
    error = prediction - actual
    denominator = actual.abs().sum()

    metrics: dict[str, float | int] = {
        "wmape_pct": float(error.abs().sum() / denominator * 100)
        if denominator > 0
        else np.nan,
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_pct": float(error.sum() / denominator * 100)
        if denominator > 0
        else np.nan,
        "scored_observation_count": int(len(scored)),
        "excluded_stockout_count": int(
            evaluation_df["is_stockout"].fillna(False).astype(bool).sum()
        ),
        "series_count": int(evaluation_df["series_id"].nunique()),
    }
    return metrics


class DemandForecastMVP:
    def __init__(
        self,
        data_pipeline: DemandDataPipeline,
        forecaster: BaseZeroShotForecaster,
    ) -> None:
        self.data_pipeline = data_pipeline
        self.forecaster = forecaster

    @staticmethod
    def _eligible_series(
        df: pd.DataFrame,
        required_length: int,
        max_series: Optional[int],
    ) -> list[str]:
        counts = df.groupby("series_id").size().sort_values(ascending=False)
        eligible = counts[counts >= required_length]
        if max_series is not None:
            eligible = eligible.head(max_series)
        if eligible.empty:
            raise ValueError(
                "Yeterli uzunlukta mağaza-ürün serisi bulunamadı. "
                f"Gerekli minimum kayıt: {required_length}"
            )
        return eligible.index.astype(str).tolist()

    def backtest(
        self,
        prepared_df: pd.DataFrame,
        horizon: int = 14,
        min_context: int = 90,
        max_series: Optional[int] = 100,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if horizon <= 0:
            raise ValueError("horizon pozitif olmalıdır.")
        eligible_ids = self._eligible_series(
            prepared_df,
            required_length=min_context + horizon,
            max_series=max_series,
        )
        selected = prepared_df.loc[
            prepared_df["series_id"].isin(eligible_ids)
        ].copy()

        context_parts: list[pd.DataFrame] = []
        validation_parts: list[pd.DataFrame] = []
        for _, group in selected.groupby("series_id", sort=False):
            group = group.sort_values("date")
            context_parts.append(group.iloc[:-horizon].copy())
            validation_parts.append(group.iloc[-horizon:].copy())

        context_raw = pd.concat(context_parts, ignore_index=True)
        validation = pd.concat(validation_parts, ignore_index=True)

        model_context = self.data_pipeline.make_model_context(
            context_raw,
            adjust_stockouts=True,
        )
        predictions = self.forecaster.predict(model_context, horizon=horizon)
        validation = validation.rename(columns={"sales": "actual"})
        evaluation = validation.merge(
            predictions,
            on=["series_id", "date"],
            how="left",
            validate="one_to_one",
        )
        if evaluation["predictions"].isna().any():
            missing_count = int(evaluation["predictions"].isna().sum())
            raise RuntimeError(
                f"{missing_count} doğrulama satırı için tahmin eşleşmedi. "
                "Tarih frekansını ve seri tarihlerini kontrol edin."
            )

        metrics: dict[str, object] = calculate_forecast_metrics(evaluation)
        metrics["model_id"] = self.forecaster.model_id
        metrics["model_name"] = self.forecaster.model_name
        return evaluation, metrics

    def forecast(
        self,
        prepared_df: pd.DataFrame,
        horizon: int = 14,
        min_context: int = 90,
        max_series: Optional[int] = 100,
    ) -> pd.DataFrame:
        eligible_ids = self._eligible_series(
            prepared_df,
            required_length=min_context,
            max_series=max_series,
        )
        context_raw = prepared_df.loc[
            prepared_df["series_id"].isin(eligible_ids)
        ].copy()
        model_context = self.data_pipeline.make_model_context(
            context_raw,
            adjust_stockouts=True,
        )
        predictions = self.forecaster.predict(model_context, horizon=horizon)

        id_columns = ["series_id", "store_id", "product_id"]
        for category in ("category_1", "category_2", "category_3"):
            if category in context_raw.columns:
                id_columns.append(category)
        id_lookup = (
            context_raw.sort_values("date")
            .groupby("series_id", as_index=False)
            .tail(1)[id_columns]
            .drop_duplicates("series_id")
        )
        predictions = predictions.merge(
            id_lookup,
            on="series_id",
            how="left",
            validate="many_to_one",
        )
        return predictions.sort_values(["series_id", "date"]).reset_index(drop=True)

    def add_stock_risk(
        self,
        forecast_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        demand_column: str = "predictions",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if demand_column not in forecast_df.columns:
            raise ValueError(
                f"Stok hesabında kullanılacak tahmin sütunu bulunamadı: {demand_column}"
            )

        latest_stock = self.data_pipeline.latest_available_stock(prepared_df)
        detail = forecast_df.merge(
            latest_stock[["series_id", "opening_stock"]],
            on="series_id",
            how="left",
            validate="many_to_one",
        ).sort_values(["series_id", "date"])

        if detail["opening_stock"].isna().any():
            raise DataValidationError("Bazı seriler için güncel stok bulunamadı.")

        detail["planning_demand"] = pd.to_numeric(
            detail[demand_column], errors="coerce"
        ).clip(lower=0)
        if detail["planning_demand"].isna().any():
            raise ValueError(f"'{demand_column}' sütununda geçersiz tahmin değerleri var.")

        detail["cumulative_forecast"] = detail.groupby("series_id")[
            "planning_demand"
        ].cumsum()
        detail["projected_stock"] = (
            detail["opening_stock"] - detail["cumulative_forecast"]
        )
        detail["stockout_risk"] = detail["projected_stock"].le(0)
        detail["cumulative_shortage"] = (-detail["projected_stock"]).clip(lower=0)

        group_columns = ["series_id", "store_id", "product_id"]
        for category in ("category_1", "category_2", "category_3"):
            if category in detail.columns:
                group_columns.append(category)

        summary = (
            detail.groupby(group_columns, as_index=False, dropna=False)
            .agg(
                opening_stock=("opening_stock", "first"),
                forecast_demand=("planning_demand", "sum"),
                expected_ending_stock=("projected_stock", "last"),
                expected_shortage=("cumulative_shortage", "max"),
                stockout_risk=("stockout_risk", "max"),
            )
        )
        first_stockout_date = (
            detail.loc[detail["stockout_risk"]]
            .groupby("series_id")["date"]
            .min()
            .rename("expected_stockout_date")
        )
        summary = summary.merge(first_stockout_date, on="series_id", how="left")
        summary["demand_source"] = demand_column
        return detail.reset_index(drop=True), summary


MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "chronos_bolt": {
        "display_name": "Chronos Bolt Small",
        "family": "chronos",
        "model_id": "amazon/chronos-bolt-small",
    },
    "chronos_2": {
        "display_name": "Chronos 2",
        "family": "chronos",
        "model_id": "amazon/chronos-2",
    },
    "timesfm_2_5": {
        "display_name": "TimesFM 2.5",
        "family": "timesfm",
        "model_id": "google/timesfm-2.5-200m-pytorch",
    },
}


def create_forecaster(model_key: str, freq: str = "D") -> BaseZeroShotForecaster:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Bilinmeyen model: {model_key}")
    config = MODEL_CONFIGS[model_key]
    if config["family"] == "chronos":
        return ChronosForecaster(model_id=config["model_id"])
    if config["family"] == "timesfm":
        return TimesFMForecaster(
            model_id=config["model_id"],
            freq=freq,
            batch_size=16,
            max_context=1024,
            max_horizon=256,
        )
    raise ValueError(f"Desteklenmeyen model ailesi: {config['family']}")


def compare_zero_shot_models(
    prepared_df: pd.DataFrame,
    data_pipeline: DemandDataPipeline,
    model_keys: Sequence[str],
    horizon: int = 14,
    min_context: int = 90,
    max_series: Optional[int] = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    evaluation_frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, str]] = []

    for model_key in model_keys:
        model_name = MODEL_CONFIGS[model_key]["display_name"]
        print("\n" + "=" * 70)
        print(f"Model çalıştırılıyor: {model_name}")
        forecaster: Optional[BaseZeroShotForecaster] = None
        model_mvp: Optional[DemandForecastMVP] = None

        try:
            start_time = time.perf_counter()
            forecaster = create_forecaster(model_key, freq=data_pipeline.pandas_freq)
            model_mvp = DemandForecastMVP(data_pipeline, forecaster)
            evaluation_df, metrics = model_mvp.backtest(
                prepared_df=prepared_df,
                horizon=horizon,
                min_context=min_context,
                max_series=max_series,
            )
            runtime_seconds = time.perf_counter() - start_time
            metrics.update(
                {
                    "model_key": model_key,
                    "model_name": model_name,
                    "runtime_seconds": round(runtime_seconds, 2),
                }
            )
            evaluation_df["model_key"] = model_key
            evaluation_df["model_name"] = model_name
            metric_rows.append(metrics)
            evaluation_frames.append(evaluation_df)
            print(
                f"Tamamlandı | WMAPE={metrics['wmape_pct']:.2f}% | "
                f"Bias={metrics['bias_pct']:.2f}% | Süre={runtime_seconds:.1f} sn"
            )
        except Exception as error:
            error_rows.append(
                {
                    "model_key": model_key,
                    "model_name": str(model_name),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"Model başarısız: {model_name}\nHata: {type(error).__name__}: {error}")
        finally:
            if model_mvp is not None:
                del model_mvp
            if forecaster is not None:
                del forecaster
            gc.collect()
            torch = _get_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(metric_rows)
    evaluations_df = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame()
    )
    errors_df = pd.DataFrame(error_rows)
    return metrics_df, evaluations_df, errors_df


def load_local_file(file_path: str) -> pd.DataFrame:
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file_path, sep=None, engine="python")
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    if lower.endswith(".parquet"):
        return pd.read_parquet(file_path)
    raise ValueError("Yalnızca CSV, Excel veya Parquet dosyası destekleniyor.")


def load_kaggle_retail_data(
    dataset_slug: str = "anirudhchauhan/retail-store-inventory-forecasting-dataset",
) -> pd.DataFrame:
    import kagglehub

    path = kagglehub.dataset_download(dataset_slug)
    supported_files = []
    for pattern in ("*.csv", "*.xlsx", "*.xls", "*.parquet"):
        supported_files.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
    if not supported_files:
        raise FileNotFoundError("İndirilen klasörde desteklenen veri dosyası bulunamadı.")
    file_path = supported_files[0]
    print(f"Okunan dosya: {file_path}")
    return load_local_file(file_path)


def plot_sales_vs_adjusted(
    adjusted_df: pd.DataFrame,
    series_id: str,
    last_n: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    plot_df = adjusted_df.loc[
        adjusted_df["series_id"].eq(series_id)
    ].sort_values("date")
    if last_n is not None:
        plot_df = plot_df.tail(last_n)
    if plot_df.empty:
        raise ValueError(f"Seri bulunamadı: {series_id}")

    plt.figure(figsize=(15, 6))
    plt.plot(plot_df["date"], plot_df["sales"], label="Gerçek satış")
    plt.plot(
        plot_df["date"],
        plot_df["demand_adjusted"],
        label="Demand adjusted",
    )
    stockout_points = plot_df.loc[plot_df["is_stockout"]]
    if not stockout_points.empty:
        plt.scatter(
            stockout_points["date"],
            stockout_points["sales"],
            marker="x",
            s=80,
            label="Stokout",
        )
    plt.title(f"Gerçek satış ve düzeltilmiş talep — {series_id}")
    plt.xlabel("Tarih")
    plt.ylabel("Miktar")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_backtest_series(
    evaluations_df: pd.DataFrame,
    series_id: str,
) -> None:
    import matplotlib.pyplot as plt

    plot_df = evaluations_df.loc[
        evaluations_df["series_id"].eq(series_id)
    ].sort_values(["model_name", "date"])
    if plot_df.empty:
        raise ValueError(f"Seri bulunamadı: {series_id}")

    actual_df = plot_df.drop_duplicates("date").sort_values("date")
    plt.figure(figsize=(14, 6))
    plt.plot(
        actual_df["date"],
        actual_df["actual"],
        marker="o",
        label="Gerçek satış",
    )
    for model_name, group in plot_df.groupby("model_name", sort=False):
        plt.plot(
            group["date"],
            group["predictions"],
            marker="o",
            label=model_name,
        )
    plt.title(f"Backtest — {series_id}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

