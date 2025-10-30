# preprocesamiento.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, Any

class Preprocesador:
    def __init__(self):
        # diccionarios que guardarán parámetros para el reporte
        self.report: Dict[str, Any] = {
            "columns": {},
            "global": {},
            "mappings": {}
        }

    def identify_xy(self, df: pd.DataFrame, y_column: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        if y_column and y_column in df.columns:
            y_col = y_column
        else:
            y_col = df.columns[-1]
        self.report["global"]["y_column"] = y_col
        self.report["global"]["x_columns"] = [c for c in df.columns if c != y_col]
        return df, y_col

    def convert_non_numeric(self, df: pd.DataFrame, threshold_for_numeric=0.6):
        for col in df.columns:
            col_info = {"original_dtype": str(df[col].dtype)}
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["converted_to"] = "numeric"
                col_info["notes"] = ""
                self.report["columns"][col] = col_info
                continue

            # intentar conversión numérica
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_after = converted.notna().sum()
            ratio = non_null_after / max(1, len(df))
            if ratio >= threshold_for_numeric:
                # convertir y reportar
                df[col] = converted
                col_info["converted_to"] = "numeric_via_to_numeric"
                col_info["notes"] = f"{ratio:.2f} values convertible to numeric"
            else:
                # tratar como categórica -> pasar a códigos
                df[col] = df[col].astype("category")
                mapping = dict(enumerate(df[col].cat.categories))
                # cat.codes produce -1 para NaN, mantendremos NaN
                codes = df[col].cat.codes.replace({-1: pd.NA})
                df[col] = codes
                col_info["converted_to"] = "category_codes"
                col_info["notes"] = f"{len(mapping)} categorías"
                # Guardar mapping inverso (code -> category)
                inv_map = {int(k): v for k, v in mapping.items()}
                self.report["mappings"][col] = inv_map

            self.report["columns"][col] = col_info

        return df

    def drop_high_missing(self, df: pd.DataFrame, col_threshold: float = 0.5) -> pd.DataFrame:
        cols_to_drop = []
        for col in df.columns:
            missing_ratio = df[col].isna().mean()
            self.report["columns"].setdefault(col, {})
            self.report["columns"][col]["missing_before"] = int(df[col].isna().sum())
            self.report["columns"][col]["missing_ratio_before"] = float(missing_ratio)
            if missing_ratio > col_threshold:
                cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.report["global"]["dropped_columns_high_missing"] = cols_to_drop
        else:
            self.report["global"]["dropped_columns_high_missing"] = []

        return df

    def fill_missing(self, df: pd.DataFrame, strategy: str = "mean", fill_categorical: str = "mode", drop_rows: bool = False) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().sum() == 0:
                    fill_val = None
                else:
                    if strategy == "mean":
                        fill_val = float(df[col].mean())
                    elif strategy == "median":
                        fill_val = float(df[col].median())
                    elif strategy == "mode":
                        m = df[col].mode()
                        fill_val = float(m.iloc[0]) if not m.empty else 0.0
                    elif strategy == "ffill":
                        df[col] = df[col].fillna(method="ffill")
                        fill_val = "ffill"
                    elif strategy == "bfill":
                        df[col] = df[col].fillna(method="bfill")
                        fill_val = "bfill"
                    else:
                        fill_val = float(df[col].mean())

                    if isinstance(fill_val, (int, float)):
                        df[col] = df[col].fillna(fill_val)
                self.report["columns"].setdefault(col, {})
                self.report["columns"][col]["fill_used"] = fill_val
                self.report["columns"][col]["missing_after"] = int(df[col].isna().sum())
            else:
                # categóricas (ya codificadas como num o con NaN)
                if df[col].isna().sum() == 0:
                    fill_val = None
                else:
                    if fill_categorical == "mode":
                        m = df[col].mode()
                        fill_val = int(m.iloc[0]) if not m.empty else None
                        df[col] = df[col].fillna(fill_val)
                    else:
                        # llenar con un valor sentinel (por ejemplo -9999)
                        fill_val = -9999
                        df[col] = df[col].fillna(fill_val)
                self.report["columns"].setdefault(col, {})
                self.report["columns"][col]["fill_used"] = fill_val
                self.report["columns"][col]["missing_after"] = int(df[col].isna().sum())

        if drop_rows:
            before = len(df)
            df = df.dropna()
            after = len(df)
            self.report["global"]["dropped_rows_with_na"] = before - after

        return df

    def scale_features(self, df: pd.DataFrame, x_columns: list, method: Optional[str] = None):
        scaling_info = {}
        if method is None:
            self.report["global"]["scaling"] = None
            return df

        for col in x_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_vals = df[col].astype(float)
            if method == "standard":
                mu = float(col_vals.mean())
                sigma = float(col_vals.std(ddof=0) if col_vals.std(ddof=0) != 0 else 1.0)
                df[col] = (col_vals - mu) / sigma
                scaling_info[col] = {"method": "standard", "mean": mu, "std": sigma}
            elif method == "minmax":
                cmin = float(col_vals.min())
                cmax = float(col_vals.max())
                denom = (cmax - cmin) if (cmax - cmin) != 0 else 1.0
                df[col] = (col_vals - cmin) / denom
                scaling_info[col] = {"method": "minmax", "min": cmin, "max": cmax}
            else:
                scaling_info[col] = {"method": "none"}
        self.report["global"]["scaling"] = {"method": method, "params": scaling_info}
        return df

    def basic_statistics(self, df: pd.DataFrame):
        try:
            desc = df.describe(include='all').to_dict()
        except Exception:
            desc = {}
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        self.report["global"]["describe"] = desc
        self.report["global"]["missing_after"] = missing
        return

    def process(
        self,
        df: pd.DataFrame,
        original_path: Optional[str] = None,
        y_column: Optional[str] = None,
        fill_strategy: str = "mean",
        fill_categorical: str = "mode",
        drop_threshold: float = 0.5,
        drop_rows: bool = False,
        scale_method: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # reset reporte
        self.report = {"columns": {}, "global": {}, "mappings": {}}

        df = df.copy()
        # identificar y
        df, y_col = self.identify_xy(df, y_column=y_column)
        x_cols = [c for c in df.columns if c != y_col]

        # intentos de conversión (non-numeric -> numeric o cat codes)
        df = self.convert_non_numeric(df)

        # eliminar columnas con missing > threshold
        df = self.drop_high_missing(df, col_threshold=drop_threshold)

        # rellenar/fill
        df = self.fill_missing(df, strategy=fill_strategy, fill_categorical=fill_categorical, drop_rows=drop_rows)

        # scaling sobre X
        x_cols = [c for c in df.columns if c != y_col]  # recompute after possible drops
        df = self.scale_features(df, x_cols, method=scale_method)

        # estadísticas básicas
        self.basic_statistics(df)

        # guardar información de identificación
        self.report["global"]["original_path"] = str(original_path) if original_path else None
        self.report["global"]["rows_after"] = len(df)
        self.report["global"]["columns_after"] = list(df.columns)

        return df, self.report

    def save_processed(
        self,
        df: pd.DataFrame,
        original_path: Optional[str] = None,
        target_path: Optional[str] = None,
        report: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> Tuple[str, str]:
        if original_path:
            orig = Path(original_path)
            ext = orig.suffix.lower()
            base = orig.stem
            folder = orig.parent
            if target_path:
                out_path = Path(target_path)
            else:
                out_path = folder / f"{base}_preprocessed{ext}"
        else:
            if target_path:
                out_path = Path(target_path)
            else:
                out_path = Path("preprocessed_dataset.csv")

        # guardar según extensión
        if out_path.suffix in [".csv"]:
            df.to_csv(out_path, index=False)
        elif out_path.suffix in [".xlsx", ".xls"]:
            df.to_excel(out_path, index=False)
        else:
            # por defecto csv
            df.to_csv(out_path.with_suffix(".csv"), index=False)
            out_path = out_path.with_suffix(".csv")

        # guardar reporte
        report_path = out_path.with_name(out_path.stem + "_preprocess_report.json")
        rep = report if report is not None else self.report
        # asegurar serializable (pandas/numpy types)
        def sanitize(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, default=sanitize, indent=2, ensure_ascii=False)

        return str(out_path), str(report_path)
