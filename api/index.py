from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
import io
import json
import traceback
from typing import Optional

# Import local modules
try:
    from api.cleaning_pipeline import AutoCleaner, CleanConfig
except ImportError:
    try:
        from cleaning_pipeline import AutoCleaner, CleanConfig
    except ImportError:
        import cleaning_pipeline
        AutoCleaner = cleaning_pipeline.AutoCleaner
        CleanConfig = cleaning_pipeline.CleanConfig

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "AutoCleaner API (Polars) is running"}

@app.post("/api/process")
async def process_file(
    file: UploadFile = File(...),
    config: str = Form(...)
):
    try:
        # 1. Parse Config
        try:
            config_dict = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON config")

        clean_config = CleanConfig(
            numeric_impute=config_dict.get("numeric_impute", "median"),
            numeric_impute_fill_value=config_dict.get("numeric_impute_fill_value"),
            categorical_impute=config_dict.get("categorical_impute", "most_frequent"),
            categorical_impute_fill_value=config_dict.get("categorical_impute_fill_value", "missing"),
            outlier_method=config_dict.get("outlier_method", "iqr"),
            iqr_multiplier=float(config_dict.get("iqr_multiplier", 1.5)),
            outlier_strategy=config_dict.get("outlier_strategy", "winsorize"),
            scaler=config_dict.get("scaler", "robust"),
            encoder=config_dict.get("encoder", "onehot"),
            drop_onehot=config_dict.get("drop_onehot", "if_binary"),
            variance_threshold=float(config_dict.get("variance_threshold", 0.0)),
            corr_threshold=float(config_dict.get("corr_threshold", 0.95)) if config_dict.get("corr_threshold") is not None else None,
            target_column=config_dict.get("target_column")
        )

        # 2. Read File (Polars)
        contents = await file.read()
        filename = file.filename.lower()
        
        try:
            if filename.endswith(".csv"):
                df = pl.read_csv(io.BytesIO(contents), null_values=["NA", "null", "None", ""])
            elif filename.endswith(".parquet") or filename.endswith(".pq"):
                df = pl.read_parquet(io.BytesIO(contents))
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                # Polars supports read_excel via engine
                df = pl.read_excel(io.BytesIO(contents))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Parquet.")
        except Exception as read_err:
             raise HTTPException(status_code=400, detail=f"Error reading file: {str(read_err)}")

        # 3. Clean Data
        cleaner = AutoCleaner(clean_config)
        cleaner.fit(df)
        df_cleaned = cleaner.transform(df)

        # 4. Return as CSV
        stream = io.BytesIO()
        df_cleaned.write_csv(stream)
        stream.seek(0)
        
        response = StreamingResponse(stream, media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
