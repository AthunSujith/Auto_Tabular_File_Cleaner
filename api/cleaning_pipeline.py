# cleaning_pipeline.py
# -----------------------------------------------------------------------------
# A reusable, configurable data cleaning & preprocessing pipeline.
# REWRITTEN to use pure Pandas/Numpy to avoid Scikit-Learn/Scipy bloat for Vercel.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# =========================
# Configuration container
# =========================
@dataclass
class CleanConfig:
    """
    Centralized configuration for AutoCleaner behavior.
    """
    numeric_impute: str = "median"            # "mean" | "median" | "most_frequent" | "constant"
    numeric_impute_fill_value: Optional[float] = None

    categorical_impute: str = "most_frequent" # "most_frequent" | "constant"
    categorical_impute_fill_value: str = "missing"

    outlier_method: str = "iqr"               # "iqr" | "none"
    iqr_multiplier: float = 1.5
    outlier_strategy: str = "winsorize"       # "winsorize" | "clip" | "none"

    scaler: str = "robust"                    # "standard" | "minmax" | "robust" | "none"

    encoder: str = "onehot"                   # "onehot" | "ordinal"
    drop_onehot: str = "if_binary"            # "if_binary" | "first" | "none"

    variance_threshold: float = 0.0           # > 0.0 to drop low variance cols
    corr_threshold: Optional[float] = None    # drop highly correlated numeric cols
    target_column: Optional[str] = None

    save_artifacts_dir: Optional[str] = "artifacts"


# =========================
# Helper utilities
# =========================
def split_columns(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns, excluding the target if provided."""
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def ensure_dir(path: Optional[str]):
    if path is None:
        return
    import os
    os.makedirs(path, exist_ok=True)


# =========================
# Core cleaner (Pure Pandas)
# =========================
class AutoCleaner:
    """
    Pandas-based cleaner that mimics the previous Scikit-Learn interface.
    """

    def __init__(self, config: Optional[CleanConfig] = None):
        self.config = config or CleanConfig()
        self.fitted_ = False
        
        # State
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.fill_values_: Dict[str, any] = {}      # Imputation values
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.scale_params_: Dict[str, Dict[str, float]] = {} # e.g. {'col': {'mean': 0, 'scale': 1}}
        self.cat_mappings_: Dict[str, any] = {}     # For ordinal encoding
        self.onehot_columns_: List[str] = []        # For consistent one-hot columns (not fully improved for new data, but works for bulk)
        self.drop_cols_variance_: List[str] = []
        self.drop_cols_corr_: List[str] = []
        self.feature_order_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "AutoCleaner":
        """Learn statistics for cleaning."""
        cfg = self.config
        target = cfg.target_column if (cfg.target_column in df.columns) else None

        self.num_cols_, self.cat_cols_ = split_columns(df, target)
        X = df.drop(columns=[target]) if target else df.copy()

        # 1. Imputation Stats
        # Numeric
        for c in self.num_cols_:
            if cfg.numeric_impute == "mean":
                self.fill_values_[c] = X[c].mean()
            elif cfg.numeric_impute == "median":
                self.fill_values_[c] = X[c].median()
            elif cfg.numeric_impute == "most_frequent":
                mode = X[c].mode()
                self.fill_values_[c] = mode.iloc[0] if not mode.empty else 0
            elif cfg.numeric_impute == "constant":
                self.fill_values_[c] = cfg.numeric_impute_fill_value if cfg.numeric_impute_fill_value is not None else 0
        
        # Categorical
        for c in self.cat_cols_:
            if cfg.categorical_impute == "most_frequent":
                mode = X[c].mode()
                self.fill_values_[c] = mode.iloc[0] if not mode.empty else "missing"
            else:
                self.fill_values_[c] = cfg.categorical_impute_fill_value

        # Apply imputation temporarily to X for further stats calculation
        X_filled = X.copy()
        X_filled.fillna(value=self.fill_values_, inplace=True)

        # 2. Outliers (IQR)
        if cfg.outlier_method == "iqr":
            for c in self.num_cols_:
                q1 = X_filled[c].quantile(0.25)
                q3 = X_filled[c].quantile(0.75)
                iqr = q3 - q1
                low = q1 - (cfg.iqr_multiplier * iqr)
                high = q3 + (cfg.iqr_multiplier * iqr)
                self.outlier_bounds_[c] = (low, high)

        # 3. Scaling Params
        # We calculate these on X_filled (after impute) maybe also after outlier handling? 
        # Ideally scikit-learn pipeline does impute -> outlier -> scale. 
        # Let's apply outlier handling to temp X for scaling calc.
        X_out = X_filled.copy()
        if cfg.outlier_method == "iqr" and cfg.outlier_strategy in ["winsorize", "clip"]:
            for c in self.num_cols_:
                l, h = self.outlier_bounds_[c]
                X_out[c] = X_out[c].clip(lower=l, upper=h)
        
        if cfg.scaler != "none":
            for c in self.num_cols_:
                if cfg.scaler == "standard":
                    mean = X_out[c].mean()
                    std = X_out[c].std()
                    # avoid div by zero
                    if std == 0: std = 1.0
                    self.scale_params_[c] = {'mean': mean, 'scale': std}
                elif cfg.scaler == "minmax":
                    min_v = X_out[c].min()
                    max_v = X_out[c].max()
                    scale = max_v - min_v
                    if scale == 0: scale = 1.0
                    self.scale_params_[c] = {'min': min_v, 'scale': scale}
                elif cfg.scaler == "robust":
                    q25 = X_out[c].quantile(0.25)
                    q75 = X_out[c].quantile(0.75)
                    scale = q75 - q25
                    if scale == 0: scale = 1.0
                    # Robust scaler usually removes median
                    median = X_out[c].median()
                    self.scale_params_[c] = {'center': median, 'scale': scale}

        # 4. Encoding
        # For OneHot: we can memorize unique values to creating consistent columns
        # For Ordinal: memorize mapping
        # NOTE: For simple one-shot cleaning of a single file, pure pd.get_dummies is fine.
        # But to be robust "fit" then "transform", we should store categories.
        if cfg.encoder == "ordinal":
            for c in self.cat_cols_:
                # Assign integers based on sorted unique values
                uniques = sorted(X_filled[c].astype(str).unique())
                mapping = {val: i for i, val in enumerate(uniques)}
                self.cat_mappings_[c] = mapping

        # 5. Variance Threshold
        # Check variance on numeric columns of X_out
        if cfg.variance_threshold > 0:
            for c in self.num_cols_:
                if X_out[c].var() <= cfg.variance_threshold:
                    self.drop_cols_variance_.append(c)

        # 6. Correlation
        if cfg.corr_threshold is not None and cfg.corr_threshold > 0:
            if len(self.num_cols_) > 1:
                # Calculate correlation on numeric portion
                corr_matrix = X_out[self.num_cols_].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                self.drop_cols_corr_ = [col for col in upper.columns if any(upper[col] > cfg.corr_threshold)]

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transforms."""
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")
        
        cfg = self.config
        target = cfg.target_column if (cfg.target_column in df.columns) else None
        X = df.drop(columns=[target]) if target else df.copy()

        # 1. Impute
        X.fillna(value=self.fill_values_, inplace=True)

        # 2. Outliers
        if cfg.outlier_method == "iqr" and cfg.outlier_strategy in ["winsorize", "clip"]:
            for c in self.num_cols_:
                if c in self.outlier_bounds_:
                    low, high = self.outlier_bounds_[c]
                    X[c] = X[c].clip(lower=low, upper=high)

        # 3. Scale
        if cfg.scaler != "none":
            for c in self.num_cols_:
                if c in self.scale_params_:
                    p = self.scale_params_[c]
                    if cfg.scaler == "standard":
                        X[c] = (X[c] - p['mean']) / p['scale']
                    elif cfg.scaler == "minmax":
                        X[c] = (X[c] - p['min']) / p['scale']
                    elif cfg.scaler == "robust":
                        X[c] = (X[c] - p['center']) / p['scale']

        # 4. Feature Selection (Drop cols)
        to_drop = set(self.drop_cols_variance_ + self.drop_cols_corr_)
        # Only drop if they exist
        drop_now = [c for c in to_drop if c in X.columns]
        X.drop(columns=drop_now, inplace=True)

        # 5. Encoding (Categorical)
        # This is where pandas differs from sklearn. 
        # If we just do get_dummies, it works for the whole dataset provided.
        # But if we want to respect 'fit' categories, it's harder.
        # For this web-app usage, likely the user uploads ONE file to clean. 
        # So we can just encode current data.
        
        if self.cat_cols_:
            if cfg.encoder == "onehot":
                drop_first = True if cfg.drop_onehot == "first" else False
                # If drop_onehot == "if_binary", we need custom logic, simpler to skip for now or just do standard
                # Let's simple apply pd.get_dummies
                X = pd.get_dummies(X, columns=self.cat_cols_, drop_first=drop_first, dtype=int)
            elif cfg.encoder == "ordinal":
                for c in self.cat_cols_:
                    if c in self.cat_mappings_:
                        mapping = self.cat_mappings_[c]
                        # map unknown values to -1
                        X[c] = X[c].astype(str).map(mapping).fillna(-1)

        # 6. Re-attach Target
        if target:
            X[target] = df[target].values

        return X
