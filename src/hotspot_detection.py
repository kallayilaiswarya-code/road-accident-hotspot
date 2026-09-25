"""
K-Means Hotspot Detection Module for Road Accident Analysis.

Purpose:
--------
This script implements spatial and severity-based accident hotspot clustering
using PySpark MLlib KMeans on the engineered dataset (data/processed/accidents_engineered.csv).

Methodology:
------------
1. Features:
   - Start_Lat
   - Start_Lng
   - Severity
2. Scaling:
   - Standardized using PySpark MLlib StandardScaler (withMean=True, withStd=True)
     to place coordinates and severity on comparable scales.
3. K Evaluation:
   - Evaluates K in range [3, 8] with seed=42 using PySpark ClusteringEvaluator (silhouette).
   - Selects K with the highest silhouette score.
4. Hotspot Summary:
   - Generates summary statistics per cluster (count, average severity/coordinates, min/max coordinates).
   - NO synthetic Risk_Category (High/Medium/Low) is generated at this stage.
5. Windows-Safe Export:
   - Collects results in Spark and exports to CSV using Python's standard csv module.
"""

import csv
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

INPUT_PATH = "data/processed/accidents_engineered.csv"
CLUSTERED_OUTPUT_PATH = "data/processed/accidents_clustered.csv"
HOTSPOT_SUMMARY_PATH = "output/hotspot_summary.csv"
EVALUATION_OUTPUT_PATH = "output/kmeans_evaluation.csv"
RANDOM_SEED = 42


def main():
    # Fallback for Windows environment if JAVA_HOME is unset
    if "JAVA_HOME" not in os.environ:
        default_java = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
        if os.path.exists(default_java):
            os.environ["JAVA_HOME"] = default_java

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("RoadAccidentHotspotDetection")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"Reading engineered accident data from {INPUT_PATH}...")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(INPUT_PATH)
    )

    total_records = df.count()
    print(f"Total input records: {total_records}")

    # 1. Feature Assembly: Start_Lat, Start_Lng, Severity
    clustering_features = ["Start_Lat", "Start_Lng", "Severity"]
    print(f"Clustering features: {clustering_features}")

    assembler = VectorAssembler(
        inputCols=clustering_features,
        outputCol="raw_features"
    )
    assembled_df = assembler.transform(df)

    # 2. Feature Scaling using StandardScaler
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df).cache()

    # 3. K Evaluation (K=3 to K=8)
    print("\n--- Evaluating K-Means with K in [3, 8] ---")
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster_id",
        metricName="silhouette"
    )

    k_results = []
    models = {}

    for k in range(3, 9):
        print(f"Training K-Means for K={k}...")
        kmeans = KMeans(
            k=k,
            seed=RANDOM_SEED,
            featuresCol="features",
            predictionCol="cluster_id",
            maxIter=20
        )
        model = kmeans.fit(scaled_df)
        predictions = model.transform(scaled_df)
        silhouette = evaluator.evaluate(predictions)
        k_results.append((k, silhouette))
        models[k] = (model, predictions)
        print(f"  -> K={k}: Silhouette Score = {silhouette:.6f}")

    # 4. Select Best K based on highest Silhouette Score
    best_k, best_score = max(k_results, key=lambda x: x[1])
    print(f"\nSelected K={best_k} based on highest Silhouette Score ({best_score:.6f})")

    best_model, best_predictions = models[best_k]

    # 5. Generate Cluster Hotspot Summary
    print("\nGenerating cluster hotspot summary...")
    summary_df = (
        best_predictions.groupBy("cluster_id")
        .agg(
            F.count("*").alias("accident_count"),
            F.round(F.avg("Severity"), 4).alias("average_severity"),
            F.round(F.avg("Start_Lat"), 4).alias("average_latitude"),
            F.round(F.avg("Start_Lng"), 4).alias("average_longitude"),
            F.round(F.min("Start_Lat"), 4).alias("minimum_latitude"),
            F.round(F.max("Start_Lat"), 4).alias("maximum_latitude"),
            F.round(F.min("Start_Lng"), 4).alias("minimum_longitude"),
            F.round(F.max("Start_Lng"), 4).alias("maximum_longitude"),
        )
        .orderBy("cluster_id")
    )

    print("\nHotspot Summary:")
    summary_df.show(truncate=False)

    # 6. Save Evaluation Results (output/kmeans_evaluation.csv)
    os.makedirs(os.path.dirname(EVALUATION_OUTPUT_PATH), exist_ok=True)
    with open(EVALUATION_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "silhouette_score"])
        for k, score in k_results:
            writer.writerow([k, score])
    print(f"Saved K evaluation results to {EVALUATION_OUTPUT_PATH}")

    # 7. Save Hotspot Summary (output/hotspot_summary.csv)
    summary_rows = summary_df.collect()
    summary_columns = summary_df.columns
    os.makedirs(os.path.dirname(HOTSPOT_SUMMARY_PATH), exist_ok=True)
    with open(HOTSPOT_SUMMARY_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(summary_columns)
        for row in summary_rows:
            writer.writerow([row[c] for c in summary_columns])
    print(f"Saved hotspot summary to {HOTSPOT_SUMMARY_PATH}")

    # 8. Save Clustered Accident Records (data/processed/accidents_clustered.csv)
    # Contains all existing engineered fields + cluster_id
    export_df = best_predictions.drop("raw_features", "features").orderBy("ID")
    export_rows = export_df.collect()
    export_columns = export_df.columns

    os.makedirs(os.path.dirname(CLUSTERED_OUTPUT_PATH), exist_ok=True)
    print(f"Writing {len(export_rows)} clustered records to {CLUSTERED_OUTPUT_PATH}...")
    with open(CLUSTERED_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(export_columns)
        for row in export_rows:
            writer.writerow([row[c] for c in export_columns])

    print(f"Success: Clustered accident dataset successfully written to {CLUSTERED_OUTPUT_PATH}")

    scaled_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
