from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
import traceback
from typing import Optional

# Import local modules
# In Vercel, the working directory for the function might differ, but generally relative imports work
# if files are bundled. simple "import cleaning_pipeline" should work if it's in the same dir.
try:
    from api.cleaning_pipeline import AutoCleaner, CleanConfig
except ImportError:
    # Fallback for local testing if running from root
    try:
        from cleaning_pipeline import AutoCleaner, CleanConfig
    except ImportError:
        # Fallback for when running directly inside api/ folder
        import cleaning_pipeline
        AutoCleaner = cleaning_pipeline.AutoCleaner
        CleanConfig = cleaning_pipeline.CleanConfig

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "AutoCleaner API is running"}

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

        # Map JSON config to CleanConfig object
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

        # 2. Read File
        contents = await file.read()
        
        # Determine file type
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".parquet") or filename.endswith(".pq"):
            df = pd.read_parquet(io.BytesIO(contents))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            # Requires openpyxl or xlrd
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Parquet.")

        # 3. Clean Data
        cleaner = AutoCleaner(clean_config)
        cleaner.fit(df)
        df_cleaned = cleaner.transform(df)

        # 4. Return as CSV
        stream = io.StringIO()
        df_cleaned.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
