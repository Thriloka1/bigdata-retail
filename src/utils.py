# src/utils.py
import os
import json
from pyspark.ml.pipeline import PipelineModel
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_model(model, path):
    """Save a PipelineModel (or pipeline) to disk (overwrites)."""
    ensure_dir(os.path.dirname(path))
    # PipelineModel has write().save(), Pipeline (unfitted) doesn't; we assume fitted PipelineModel
    model.write().overwrite().save(path)

def load_model(path):
    """Load a saved PipelineModel."""
    return PipelineModel.load(path)

def save_metadata(meta: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(meta, f)

def load_metadata(path: str):
    with open(path, "r") as f:
        return json.load(f)

def excel_to_parquet(excel_path: str, parquet_out: str, sheet_name=None):
    """Read Excel via pandas and write parquet for Spark to read fast."""
    ensure_dir(os.path.dirname(parquet_out))
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # Try infer datetime col if exists, otherwise let ETL handle parsing
    df.to_parquet(parquet_out, index=False)
    return parquet_out
