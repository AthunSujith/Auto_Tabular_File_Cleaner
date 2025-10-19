# cleaning_pipeline.py
# -----------------------------------------------------------------------------
# A reusable, configurable data cleaning & preprocessing pipeline you can
# import or use via a CLI wrapper. Handles:
# - Missing values (numeric & categorical)
# - Outliers (IQR winsorization/clip)
# - Scaling (Standard/MinMax/Robust)
# - Categorical encoding (OneHot/Ordinal)
# - Feature selection (low variance, high correlation)
# - Stable column order + optional artifact persistence
# -----------------------------------------------------------------------------

from __future__ import annotations  # allow forward type references (py<3.11)

import json                       # save schema/config artifacts as JSON
from dataclasses import dataclass # concise config container
from typing import Optional, List, Dict, Tuple  # type hints for clarity

import numpy as np               # numeric ops, percentiles for IQR
import pandas as pd              # tabular data handling

# Scikit-learn primitives we compose into a pipeline-like object
from sklearn.base import BaseEstimator, TransformerMixin  # sklearn-style interface
from sklearn.impute import SimpleImputer                  # missing value strategies
from sklearn.preprocessing import (                       # scaling & encoding
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.feature_selection import VarianceThreshold   # low-variance filter


# =========================
# Configuration container
# =========================
@dataclass
class CleanConfig:
    """
    Centralized configuration for AutoCleaner behavior.
    Every field has a safe default; override as needed.
    """
    numeric_impute: str = "median"            # numeric NaN fill: "mean" | "median" | "most_frequent" | "constant"
    numeric_impute_fill_value: Optional[float] = None  # used only if numeric_impute == "constant"

    categorical_impute: str = "most_frequent" # categorical NaN fill: "most_frequent" | "constant"
    categorical_impute_fill_value: str = "missing"     # used only if categorical_impute == "constant"

    outlier_method: str = "iqr"               # outlier detection: "iqr" | "none"
    iqr_multiplier: float = 1.5               # whisker length: 1.5 (mild) or 3.0 (extreme)
    outlier_strategy: str = "winsorize"       # action: "winsorize" (cap) | "clip" | "none"

    scaler: str = "robust"                    # scaling: "standard" | "minmax" | "robust" | "none"

    encoder: str = "onehot"                   # categorical encoding: "onehot" | "ordinal"
    drop_onehot: str = "if_binary"            # OneHotEncoder drop: "if_binary" | "first" | "none"

    variance_threshold: float = 0.0           # remove near-constant features if > 0.0 (e.g., 0.0, 1e-5, 0.01)
    corr_threshold: Optional[float] = None    # drop highly correlated numeric cols; e.g., 0.95
    target_column: Optional[str] = None       # if present, preserved and appended back after transform

    save_artifacts_dir: Optional[str] = "artifacts"  # where to save schema.json etc.; None = disable


# =========================
# Helper utilities
# =========================
def split_columns(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns, excluding the target if provided."""
    if target_col and target_col in df.columns:              # if a target is provided and exists
        X = df.drop(columns=[target_col])                    # work only on features
    else:
        X = df                                               # otherwise use df as-is
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()           # numeric dtypes
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()  # categorical dtypes
    return num_cols, cat_cols                                # return lists


def iqr_bounds(s: pd.Series, k: float) -> Tuple[float, float]:
    """Compute IQR-based lower/upper bounds for winsorization/clipping."""
    q1, q3 = np.nanpercentile(s, 25), np.nanpercentile(s, 75)  # 25th & 75th percentiles
    iqr = q3 - q1                                              # interquartile range
    return (q1 - k * iqr, q3 + k * iqr)                        # Tukey fences


def ensure_dir(path: Optional[str]):
    """Create directory if not exists (no-op if None)."""
    if path is None:                    # artifact saving disabled
        return
    import os                           # local import to keep top clean
    os.makedirs(path, exist_ok=True)    # safe mkdir -p


# =========================
# Core cleaner
# =========================
class AutoCleaner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible cleaner with .fit() and .transform().
    Not a pure Pipeline so we can manage cross-cutting logic (order, selection).
    """

    def __init__(self, config: Optional[CleanConfig] = None):
        self.config = config or CleanConfig()    # use defaults if not provided
        self.fitted_ = False                     # flag to ensure fit before transform

        # Learned schema & components initialized here for clarity
        self.num_cols_: List[str] = []           # numeric columns discovered on fit
        self.cat_cols_: List[str] = []           # categorical columns discovered on fit

        self.num_imputer_: Optional[SimpleImputer] = None  # numeric imputer instance
        self.cat_imputer_: Optional[SimpleImputer] = None  # categorical imputer instance

        self.scaler_: Optional[BaseEstimator] = None        # scaler object (Standard/MinMax/Robust/None)
        self.encoder_: Optional[BaseEstimator] = None       # encoder object (OneHot/Ordinal)

        self.variance_filter_: Optional[VarianceThreshold] = None  # low-variance filter
        self.high_corr_drop_: List[str] = []                # numeric columns dropped for high correlation

        self.feature_order_: List[str] = []                 # final column order after encode/selection

    # ---------------------- fit ----------------------
    def fit(self, df: pd.DataFrame) -> "AutoCleaner":
        """Learn imputers, outlier bounds, encoders, scaler, and selection rules."""
        cfg = self.config                                                 # shorthand
        target = cfg.target_column if (cfg.target_column in df.columns) else None  # only if present

        self.num_cols_, self.cat_cols_ = split_columns(df, target)        # detect col types
        X = df.drop(columns=[target]) if target else df.copy()            # remove target if needed

        # ---- 1) Imputers: learn fill statistics ----
        self.num_imputer_ = SimpleImputer(                                # numeric imputer
            strategy=cfg.numeric_impute,
            fill_value=cfg.numeric_impute_fill_value
        )
        if self.num_cols_:                                                # only fit when columns exist
            self.num_imputer_.fit(X[self.num_cols_])

        self.cat_imputer_ = SimpleImputer(                                # categorical imputer
            strategy=cfg.categorical_impute,
            fill_value=cfg.categorical_impute_fill_value
        )
        if self.cat_cols_:
            self.cat_imputer_.fit(X[self.cat_cols_])

        # ---- 2) Outlier bounds (IQR) ----
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}         # per-numeric-column bounds
        if cfg.outlier_method == "iqr":                                   # if enabled
            for c in self.num_cols_:                                      # compute bounds per column
                low, high = iqr_bounds(X[c], cfg.iqr_multiplier)
                self.outlier_bounds_[c] = (low, high)

        # ---- 3) Choose scaler ----
        if cfg.scaler == "standard":
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        elif cfg.scaler == "minmax":
            self.scaler_ = MinMaxScaler()
        elif cfg.scaler == "robust":
            self.scaler_ = RobustScaler()
        elif cfg.scaler == "none":
            self.scaler_ = None
        else:
            raise ValueError(f"Unknown scaler: {cfg.scaler}")             # guardrail

        # ---- 4) Choose & fit encoder ----
        if cfg.encoder == "onehot":
            # Map config -> sklearn's 'drop' argument
            drop = "if_binary" if cfg.drop_onehot == "if_binary" else \
                   ("first" if cfg.drop_onehot == "first" else None)
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore",  # unseen categories won't break inference
                drop=drop,                # reduce multicollinearity if desired
                sparse_output=False       # return dense matrix for simple concatenation
            )
            if self.cat_cols_:
                # Fit on imputed categories to lock category levels
                self.encoder_.fit(self.cat_imputer_.transform(X[self.cat_cols_]))
        elif cfg.encoder == "ordinal":
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",  # avoid errors on unseen cats
                unknown_value=-1                     # mark unknowns as -1
            )
            if self.cat_cols_:
                self.encoder_.fit(self.cat_imputer_.transform(X[self.cat_cols_]))
        else:
            raise ValueError(f"Unknown encoder: {cfg.encoder}")

        # ---- 5) Learn variance filter (optional) ----
        if cfg.variance_threshold and cfg.variance_threshold > 0:
            # Build a temporary full numeric matrix (after impute/encode/scale) to measure variance
            X_temp = self._transform_matrix(X, apply_variance=False, apply_corr=False, for_variance=True)
            self.variance_filter_ = VarianceThreshold(threshold=cfg.variance_threshold)
            self.variance_filter_.fit(X_temp)

        # ---- 6) Learn high-correlation drops (numeric only) ----
        self.high_corr_drop_ = []                                         # reset
        if cfg.corr_threshold is not None and cfg.corr_threshold > 0:
            X_num = self._numeric_block(X)                                # current numeric block
            if not X_num.empty:
                corr = X_num.corr(numeric_only=True).abs()                # absolute correlations
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))  # upper triangle
                to_drop = [column for column in upper.columns if any(upper[column] > cfg.corr_threshold)]
                self.high_corr_drop_ = to_drop

        # ---- 7) Freeze final feature order ----
        X_final = self._transform_matrix(X, apply_variance=True, apply_corr=True, for_variance=False)
        self.feature_order_ = list(X_final.columns)                       # stable downstream order

        # ---- 8) Persist schema if requested ----
        if cfg.save_artifacts_dir:
            ensure_dir(cfg.save_artifacts_dir)                            # ensure dir exists
            schema = {
                "num_cols": self.num_cols_,                               # discovered numeric cols
                "cat_cols": self.cat_cols_,                               # discovered categorical cols
                "feature_order": self.feature_order_,                     # final columns after encoding/selection
                "high_corr_drop": self.high_corr_drop_,                   # dropped for correlation
                "config": cfg.__dict__,                                   # config snapshot
            }
            with open(f"{cfg.save_artifacts_dir}/schema.json", "w") as f: # write schema.json
                json.dump(schema, f, indent=2)

        self.fitted_ = True                                               # mark as fit
        return self                                                       # sklearn-style chaining

    # ------------------- transform -------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transforms to any compatible DataFrame (train/test/inference)."""
        if not self.fitted_:                                              # guard: must call fit first
            raise RuntimeError("Call fit() before transform().")
        cfg = self.config                                                 # shorthand
        target = cfg.target_column if (cfg.target_column in df.columns) else None  # detect target presence
        X = df.drop(columns=[target]) if target else df.copy()            # isolate features

        X_final = self._transform_matrix(X, apply_variance=True, apply_corr=True, for_variance=False)

        # Reindex columns to trained feature order to guarantee column parity across splits
        X_final = X_final.reindex(columns=self.feature_order_, fill_value=0)

        if target:                                                        # add target back if available
            X_final[target] = df[target].values
        return X_final                                                    # ready-to-save frame

    # -------------- internal: numeric block --------------
    def _numeric_block(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build numeric matrix: impute -> (winsorize/clip) -> scale -> (drop high corr)."""
        num = pd.DataFrame(index=X.index)                                 # default empty dataframe
        if self.num_cols_:
            num_vals = self.num_imputer_.transform(X[self.num_cols_])     # apply numeric imputer
            num = pd.DataFrame(num_vals, columns=self.num_cols_, index=X.index)

            # Outlier handling if enabled
            if self.config.outlier_method == "iqr" and self.config.outlier_strategy in ["winsorize", "clip"]:
                for c in self.num_cols_:
                    low, high = self.outlier_bounds_[c]                   # learned per-column bounds
                    if self.config.outlier_strategy == "winsorize":
                        # cap values outside bounds to the boundary values
                        num[c] = np.where(num[c] < low, low, np.where(num[c] > high, high, num[c]))
                    else:
                        # clip has the same effect numerically; explicit branch for readability
                        num[c] = num[c].clip(lower=low, upper=high)

            # Scaling if chosen
            if self.scaler_ is not None:
                # Some scalers need fit first; if already fit, just transform
                num[:] = self.scaler_.fit_transform(num) if not hasattr(self.scaler_, "n_features_in_") else self.scaler_.transform(num)

            # Drop highly correlated columns (learned at fit-time)
            if self.high_corr_drop_:
                keep = [c for c in num.columns if c not in self.high_corr_drop_]
                num = num[keep]
        return num

    # -------------- internal: categorical block --------------
    def _categorical_block(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build categorical matrix: impute -> encode (onehot/ordinal)."""
        cat = pd.DataFrame(index=X.index)                                  # default empty dataframe
        if self.cat_cols_:
            cat_imp = self.cat_imputer_.transform(X[self.cat_cols_])       # impute missing categories

            if isinstance(self.encoder_, OneHotEncoder):
                encoded = self.encoder_.transform(cat_imp)                 # one-hot encoding
                cat_cols = self.encoder_.get_feature_names_out(self.cat_cols_)  # readable col names
                cat = pd.DataFrame(encoded, columns=cat_cols, index=X.index)
            else:  # OrdinalEncoder
                encoded = self.encoder_.transform(cat_imp)                 # ordinal mapping (cats -> ints)
                cat = pd.DataFrame(encoded, columns=self.cat_cols_, index=X.index)
        return cat

    # -------------- internal: compose final matrix --------------
    def _transform_matrix(self, X: pd.DataFrame, apply_variance: bool, apply_corr: bool, for_variance: bool) -> pd.DataFrame:
        """Combine numeric + categorical blocks and optionally apply variance filter."""
        num = self._numeric_block(X)                                      # numeric part
        cat = self._categorical_block(X)                                  # categorical part
        M = pd.concat([num, cat], axis=1)                                 # horizontal concat

        # Apply learned variance filter if requested
        if apply_variance and self.variance_filter_ is not None:
            mask = self.variance_filter_.get_support()                    # Boolean mask of kept columns
            M = M.loc[:, M.columns[mask]]                                 # select columns with variance > threshold
        elif for_variance:
            # Special path used only during fit() to compute proper variances
            return M
        return M                                                          # final feature matrix


# =========================
# Convenience one-shot API
# =========================
def clean_dataframe(df: pd.DataFrame, config: Optional[CleanConfig] = None) -> pd.DataFrame:
    """
    Fit on df and return transformed df. Use when you don't need train/test separation.
    """
    cleaner = AutoCleaner(config)    # create cleaner with given config
    cleaner.fit(df)                  # learn statistics and encoders
    return cleaner.transform(df)     # return cleaned frame with stable columns
