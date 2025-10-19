# Auto_cleaner_tex_data

Auto_cleaner_tex_data is a small, configurable Python utility for cleaning
and preprocessing tabular datasets. It provides a scikit-learn style
transformer (`AutoCleaner`) that you can fit on training data and apply to
validation/test data or to new production data. A YAML-driven runner
(`run_pipeline.py`) is included to make it easy to run the pipeline from the
command line.

This repository is intentionally lightweight and focuses on clear, auditable
preprocessing steps suitable for modeling or exploratory analysis.

## Features

- Impute missing numeric and categorical values (median/mean/most_frequent/constant)
- IQR-based outlier detection and handling (winsorize or clip)
- Numeric scaling (Standard, MinMax or Robust)
- Categorical encoding (One-hot or Ordinal)
- Optional variance-based feature removal and high-correlation column dropping
- Stable column order and optional schema persistence to `artifacts/schema.json`

## Quickstart

1. Install dependencies (recommended to use a virtual environment):

```powershell
python -m venv clean; .\clean\Scripts\Activate.ps1
pip install -r Requriment.txt
```

2. Run the pipeline with a YAML config (see Sample Config below):

```powershell
python run_pipeline.py --config config.yaml
```

The runner will read the input table, fit the `AutoCleaner`, transform the
data, save the cleaned CSV, and optionally save schema artifacts.

## API usage

You can import and use `AutoCleaner` directly in Python:

```python
from cleaning_pipeline import AutoCleaner, CleanConfig
clean_cfg = CleanConfig(scaler="robust", encoder="onehot")
cleaner = AutoCleaner(clean_cfg)
cleaner.fit(train_df)
train_clean = cleaner.transform(train_df)
test_clean = cleaner.transform(test_df)
```

`AutoCleaner` follows scikit-learn conventions: call `fit()` to learn
imputation/encoding/scaling parameters, then call `transform()` to apply the
same preprocessing to other data.

## Sample YAML config

Minimal example `config.yaml` used by `run_pipeline.py`:

```yaml
io:
	input_path: data/input.csv
	output_path: data/processed.csv
	artifacts_dir: artifacts
	target_column: target  # optional
	save_columns_list: true

processing:
	numeric_impute: median
	categorical_impute: most_frequent
	outlier_method: iqr
	iqr_multiplier: 1.5
	outlier_strategy: winsorize
	scaler: robust
	encoder: onehot
	drop_onehot: if_binary
	variance_threshold: 0.0
	corr_threshold: 0.95

run:
	log_level: info
```

Notes on key processing options (also available as `CleanConfig` defaults):

- numeric_impute: "mean" | "median" | "most_frequent" | "constant"
- categorical_impute: "most_frequent" | "constant"
- outlier_method: "iqr" | "none" (IQR is robust and recommended)
- outlier_strategy: "winsorize" | "clip" | "none"
- scaler: "standard" | "minmax" | "robust" | "none"
- encoder: "onehot" | "ordinal"
- drop_onehot: "if_binary" | "first" | "none"
- variance_threshold: float (0.0 disables)
- corr_threshold: float between 0 and 1 (None disables)

## Artifacts

When enabled, the pipeline will write `artifacts/schema.json` (default
`artifacts/`) containing discovered numeric/categorical columns, final
feature order, list of dropped correlated columns, and the config snapshot.
This is intentionally a human-readable JSON for transparency.


## Streamlit Dashboard (optional)

The repository includes a Streamlit-based interactive EDA dashboard in
`streamlit_dashboard.py`. It is designed to work well with the AutoCleaner
output CSVs but can load any CSV or Parquet file. The dashboard provides
interactive filtering, distribution views, correlation heatmaps, pairwise
scatter plots, missingness overview, and quick target-aware visual checks.

Key features
- Upload CSV or Parquet files or use the built-in demo dataset (synthetic
	house-prices-like data) for quick exploration.
- Numeric filters: choose numeric columns and filter with range sliders.
- Categorical filters: multiselect-based filters for categories.
- Distributions: histograms, KDE-like density, and marginal box plots.
- Box/Violin plots grouped by category.
- Correlation heatmap with selectable method (Pearson/Spearman/Kendall).
- Pairwise scatter with optional color/size encodings.
- Missingness overview bar chart and a simple target vs feature quick view.
- Download filtered subset as CSV.

How to run the dashboard
1. Install Streamlit and Plotly (if not installed):

```powershell
pip install streamlit plotly
```

2. Start the dashboard from the repository root:

```powershell
streamlit run streamlit_dashboard.py
```

3. Use the left sidebar to upload your dataset or flip the "Use demo dataset"
	 toggle to load the synthetic demo. The sidebar also contains filters,
	 correlation method selection, and an export button for the filtered CSV.

Tips
- If you plan to explore AutoCleaner output, run the pipeline first and then
	load the processed CSV in the dashboard for a stable feature set.
- For large datasets, use the filtering controls to reduce the dataset size
	before plotting to keep the dashboard responsive.
- The dashboard is intentionally simple and easy to extend; add custom
	visualizations or analyses as separate Streamlit components.


## Notes and recommendations

- Use `robust` scaling when your numeric features contain outliers. If you
	train tree-based models (random forests / gradient boosting) you can skip
	scaling entirely.
- One-hot encoding is safe for linear models; for very high-cardinality
	categorical variables consider alternate approaches (target encoding,
	hashing) which are not implemented here.
- The pipeline is intentionally simple and readable. For production use you
	may want to pickle fitted scikit-learn objects (imputers/encoders) or add
	unit tests and more robust logging/CI.


or share.

## Contact

If you need help using or extending this pipeline, open an issue or contact

Athun Sujith
athundeveloperid59@gmail.com





