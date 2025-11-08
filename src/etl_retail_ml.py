# src/etl_retail_ml.py
"""
Full ETL + Feature Engineering + ML pipeline using PySpark.
Reads Excel dataset (via pandas -> parquet), builds daily series,
fills missing dates, adds lag & rolling features, trains:
 - Linear Regression (forecast daily_revenue)
 - Logistic Regression (growth / no-growth)
Saves outputs and models to output_dir.
"""
import sys
import os
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from utils import excel_to_parquet, save_model, save_metadata, ensure_dir

def create_spark(app_name="RetailETL_ML"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def read_input_as_spark(parquet_path, spark):
    df = spark.read.parquet(parquet_path)
    return df

def convert_excel_to_parquet_if_needed(input_path, temp_parquet_path):
    if input_path.lower().endswith((".xls", ".xlsx")):
        print("Converting Excel to parquet (pandas) for faster Spark reading...")
        excel_to_parquet(input_path, temp_parquet_path)
        return temp_parquet_path
    else:
        return input_path

def load_and_prepare_raw(spark, path):
    # We assume some common columns; adapt names if different.
    # Read parquet (created from excel) or CSV/parquet path.
    df = None
    if path.lower().endswith(".parquet"):
        df = spark.read.parquet(path)
    elif path.lower().endswith(".csv"):
        df = spark.read.csv(path, header=True, inferSchema=True)
    else:
        # Fallback: try parquet
        df = spark.read.format("parquet").load(path)
    return df

def preprocess(df):
    # Try multiple likely date columns: 'InvoiceDate', 'OrderDate', 'Date'
    possible_date_cols = [c for c in df.columns if c.lower() in ("invoicedate", "orderdate", "date", "transactiondate")]
    date_col = possible_date_cols[0] if possible_date_cols else None

    if date_col is None:
        raise ValueError("No date-like column found in dataset. Expected InvoiceDate/OrderDate/Date")

    # Coerce to string then timestamp parsing with common formats
    df2 = df.withColumn("raw_date_str", F.col(date_col).cast("string"))

    df2 = df2.withColumn("ts1", F.to_timestamp("raw_date_str", "MM/dd/yyyy HH:mm:ss")) \
             .withColumn("ts2", F.to_timestamp("raw_date_str", "MM/dd/yyyy HH:mm")) \
             .withColumn("ts3", F.to_timestamp("raw_date_str", "yyyy-MM-dd HH:mm:ss")) \
             .withColumn("ts4", F.to_timestamp("raw_date_str", "yyyy-MM-dd'T'HH:mm:ss")) \
             .withColumn("InvoiceDateParsed", F.coalesce(F.col("ts1"), F.col("ts2"), F.col("ts3"), F.col("ts4")))

    # Identify numeric columns for price/qty
    qty_col = next((c for c in df.columns if c.lower() in ("quantity", "qty", "units")), None)
    price_col = next((c for c in df.columns if c.lower() in ("unitprice", "price", "amount")), None)

    if qty_col is None or price_col is None:
        # Try to infer - if single numeric columns exist
        numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float", "long")]
        if len(numeric_cols) >= 2:
            qty_col = qty_col or numeric_cols[0]
            price_col = price_col or numeric_cols[1]
        else:
            raise ValueError("Could not find Quantity and UnitPrice columns. Expected Quantity and UnitPrice (or similar).")

    # Add TotalPrice and canonical columns
    df3 = df2.withColumn("Quantity", F.coalesce(F.col(qty_col).cast("long"), F.lit(0))) \
             .withColumn("UnitPrice", F.coalesce(F.col(price_col).cast("double"), F.lit(0.0))) \
             .withColumn("TotalPrice", F.col("Quantity") * F.col("UnitPrice")) \
             .filter(F.col("InvoiceDateParsed").isNotNull())

    # Optionally filter negative prices/quantities (treat as returns if you want)
    df3 = df3.filter((F.col("Quantity") != 0) & (F.col("UnitPrice") >= 0))

    return df3.select("*")  # keep all for joins if needed

def aggregate_daily(df):
    daily = df.withColumn("date", F.to_date("InvoiceDateParsed")) \
        .groupBy("date") \
        .agg(
            F.round(F.sum("TotalPrice"), 2).alias("daily_revenue"),
            F.sum("Quantity").alias("daily_quantity"),
            F.countDistinct(F.coalesce(F.col("InvoiceNo"), F.lit("no_invoice"))).alias("num_transactions")
        ) \
        .orderBy("date")
    return daily

def fill_missing_dates(spark, daily_df):
    min_date = daily_df.agg(F.min("date")).first()[0]
    max_date = daily_df.agg(F.max("date")).first()[0]
    min_date_str = min_date.strftime("%Y-%m-%d")
    max_date_str = max_date.strftime("%Y-%m-%d")

    # Use sequence to build full date range
    seq_df = spark.sql(f"select sequence(to_date('{min_date_str}'), to_date('{max_date_str}'), interval 1 day) as date_seq") \
                  .withColumn("date", F.explode("date_seq")).select("date")
    filled = seq_df.join(daily_df, on="date", how="left") \
                   .fillna(0, subset=["daily_revenue", "daily_quantity", "num_transactions"])
    return filled, min_date_str, max_date_str

def add_time_features(filled_df, min_date_str):
    df = filled_df.withColumn("day_index", F.datediff(F.col("date"), F.lit(min_date_str)).cast("double")) \
                  .withColumn("month", F.month("date")) \
                  .withColumn("day_of_week", F.date_format("date", "u").cast("int"))
    return df

def add_lag_features(df, lag_list=[1,2,3,7]):
    w = Window.orderBy("date")
    out = df
    for lag in lag_list:
        out = out.withColumn(f"lag_{lag}", F.lag("daily_revenue", lag).over(w))
    # rolling averages using rowsBetween (include current row)
    out = out.withColumn("rolling_avg_3", F.avg("daily_revenue").over(w.rowsBetween(-2, 0)))
    out = out.withColumn("rolling_avg_7", F.avg("daily_revenue").over(w.rowsBetween(-6, 0)))
    out = out.fillna(0)
    return out

def train_linear_regression(df_features, feature_cols, label_col="daily_revenue"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    lr = LinearRegression(featuresCol="features", labelCol=label_col, maxIter=100, regParam=0.01)
    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # Time-based split
    total = df_features.count()
    train_count = int(total * 0.8)
    w = Window.orderBy("date")
    df_ordered = df_features.withColumn("rownum", F.row_number().over(w))
    train_df = df_ordered.filter(F.col("rownum") <= train_count).cache()
    test_df = df_ordered.filter(F.col("rownum") > train_count).cache()

    model = pipeline.fit(train_df)
    preds = model.transform(test_df)

    rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse").evaluate(preds)
    r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2").evaluate(preds)

    return model, (rmse, r2), train_df, test_df, preds

def prepare_growth_label(df):
    w = Window.orderBy("date")
    df2 = df.withColumn("prev_revenue", F.lag("daily_revenue", 1).over(w))
    df2 = df2.withColumn("growth", F.when(F.col("daily_revenue") > F.col("prev_revenue"), 1.0).otherwise(0.0))
    df2 = df2.fillna({'prev_revenue': 0.0})
    return df2

def train_logistic_regression(df_features, feature_cols, label_col="growth"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    logr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=50, regParam=0.01)
    pipeline = Pipeline(stages=[assembler, scaler, logr])

    total = df_features.count()
    train_count = int(total * 0.8)
    w = Window.orderBy("date")
    df_ordered = df_features.withColumn("rownum", F.row_number().over(w))
    train_df = df_ordered.filter(F.col("rownum") <= train_count).cache()
    test_df = df_ordered.filter(F.col("rownum") > train_count).cache()

    model = pipeline.fit(train_df)
    preds = model.transform(test_df)

    auc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds)
    acc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy").evaluate(preds)

    return model, (auc, acc), train_df, test_df, preds

def predict_future_dates(pipeline_model, min_date_str, last_date_str, days_ahead=30):
    # Build Spark session inside function
    spark = SparkSession.builder.getOrCreate()
    # build date sequence from last_date + 1 to last_date + days_ahead
    seq_df = spark.sql(f"select sequence(to_date('{last_date_str}') + interval 1 day, to_date('{last_date_str}') + interval {days_ahead} day, interval 1 day) as ds") \
                  .withColumn("date", F.explode("ds")).select("date")
    # compute day_index relative to min_date_str (must match training)
    seq_df = seq_df.withColumn("day_index", F.datediff(F.col("date"), F.lit(min_date_str)).cast("double"))
    # Add placeholder prev_revenue/lag features as 0 or carry-forward if you'd like (simple approach: 0)
    seq_df = seq_df.withColumn("prev_revenue", F.lit(0.0)).withColumn("lag_1", F.lit(0.0)).withColumn("lag_2", F.lit(0.0)) \
                   .withColumn("lag_3", F.lit(0.0)).withColumn("lag_7", F.lit(0.0)).withColumn("rolling_avg_3", F.lit(0.0)) \
                   .withColumn("rolling_avg_7", F.lit(0.0)).withColumn("month", F.month(F.col("date"))).withColumn("day_of_week", F.date_format("date","u").cast("int"))

    preds = pipeline_model.transform(seq_df)
    return preds.select("date", "prediction")

def main(input_path, output_dir, predict_days=30):
    tmp_parquet = "/tmp/retail_input.parquet"
    ensure_dir(output_dir)
    # convert excel to parquet if needed
    source = convert_excel_to_parquet_if_needed(input_path, tmp_parquet)

    spark = create_spark()
    print("Loading source into Spark...")
    raw = read_input_as_spark(source, spark)
    print("Preprocessing...")
    df_clean = preprocess(raw)
    print("Aggregating daily...")
    daily = aggregate_daily(df_clean)
    print("Filling missing dates...")
    filled, min_date_str, max_date_str = fill_missing_dates(spark, daily)
    filled = add_time_features(filled, min_date_str)
    filled = add_lag_features(filled)
    filled = filled.orderBy("date").cache()

    # Save daily series CSV
    daily_out = os.path.join(output_dir, "daily_sales")
    filled.coalesce(1).write.mode("overwrite").csv(daily_out, header=True)
    print("Saved daily series to:", daily_out)

    # Persist metadata (min_date) for later use
    meta = {"min_date": min_date_str, "max_date": max_date_str}
    save_metadata(meta, os.path.join(output_dir, "models", "metadata.json"))

    # --- Train Linear Regression ---
    feature_cols_lr = ["day_index", "lag_1", "lag_2", "lag_3", "lag_7", "rolling_avg_3", "rolling_avg_7", "month", "day_of_week"]
    print("Training Linear Regression using features:", feature_cols_lr)
    lr_model, (rmse, r2), lr_train, lr_test, lr_preds = train_linear_regression(filled, feature_cols_lr, label_col="daily_revenue")
    print(f"LR RMSE={rmse:.4f}, R2={r2:.4f}")

    # Save LR predictions & model
    lr_pred_out = os.path.join(output_dir, "lr_test_predictions")
    lr_preds.select("date", "daily_revenue", "prediction").orderBy("date").coalesce(1).write.mode("overwrite").csv(lr_pred_out, header=True)
    save_model(lr_model, os.path.join(output_dir, "models", "lr_model"))
    print("Saved LR model & predictions.")

    # --- Train Logistic Regression ---
    df_growth = prepare_growth_label(filled)
    feature_cols_logr = ["day_index", "prev_revenue", "lag_1", "lag_7", "rolling_avg_3", "month", "day_of_week"]
    print("Training Logistic Regression using features:", feature_cols_logr)
    logr_model, (auc, acc), logr_train, logr_test, logr_preds = train_logistic_regression(df_growth, feature_cols_logr, label_col="growth")
    print(f"Logistic AUC={auc:.4f}, ACC={acc:.4f}")

    logr_pred_out = os.path.join(output_dir, "logr_test_predictions")
    logr_preds.select("date", "daily_revenue", "prev_revenue", "prediction", "probability").orderBy("date").coalesce(1).write.mode("overwrite").csv(logr_pred_out, header=True)
    save_model(logr_model, os.path.join(output_dir, "models", "logr_model"))
    print("Saved Logistic model & predictions.")

    # --- Future predictions (linear) ---
    print(f"Predicting next {predict_days} days (simple generation)...")
    future = predict_future_dates(lr_model, min_date_str, max_date_str, days_ahead=predict_days)
    future_out = os.path.join(output_dir, "future_lr_predictions")
    future.orderBy("date").coalesce(1).write.mode("overwrite").csv(future_out, header=True)
    print("Saved future predictions to:", future_out)

    spark.stop()
    print("Pipeline complete. Outputs in:", output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: etl_retail_ml.py <input_excel_or_parquet_or_csv> <output_dir> [predict_days]")
        sys.exit(1)
    input_path = sys.argv[1]
    out_dir = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) >= 4 else 30
    main(input_path, out_dir, predict_days=days)
