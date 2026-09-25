"""
Feature Engineering Module for Road Accident Hotspot Detection and Risk Analysis.

Purpose:
--------
This script performs the feature engineering stage on the cleaned accident dataset
(data/processed/accidents_cleaned.csv) using Apache PySpark.

Why these features are being created:
-------------------------------------
- Temporal Analysis:
  Extracting Accident_Year, Accident_Month, Accident_DayOfWeek, Accident_Hour,
  Is_Weekend, and Is_Rush_Hour enables discovery of peak accident periods,
  seasonal variations, weekday commuting patterns, and rush-hour vulnerability.
- Accident Impact & Clearance:
  Accident_Duration_Minutes measures the duration between Start_Time and End_Time
  to quantify temporal disruption and scene clearance times.
- Environmental & Road Conditions:
  Retaining weather attributes (Temperature, Humidity, Pressure, Visibility,
  Wind Speed, Wind Direction, Weather Condition) and road infrastructure flags
  (Traffic Signal, Junction, Crossing, Stop, Railway, etc.) enables multi-factor
  environmental risk assessment.
- Spatial Analysis:
  Retaining Start_Lat, Start_Lng, State, City, Street, and County preserves
  fine-grained spatial context essential for subsequent hotspot clustering (K-Means).

Target Formulation Policy:
--------------------------
- No synthetic final risk label (such as High / Medium / Low) is created at this stage.
- Severity is retained strictly as an existing raw attribute and is NOT used as a
  substitute for the future risk target.
- Categorical values remain in readable format without one-hot encoding at this stage.
"""

import csv
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    year,
    month,
    dayofweek,
    hour,
    when,
    round as spark_round,
)

INPUT_PATH = "data/processed/accidents_cleaned.csv"
OUTPUT_PATH = "data/processed/accidents_engineered.csv"


def main():
    # Fallback for Windows environment if JAVA_HOME is unset
    if "JAVA_HOME" not in os.environ:
        default_java = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
        if os.path.exists(default_java):
            os.environ["JAVA_HOME"] = default_java

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("RoadAccidentFeatureEngineering")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading cleaned accident data from {INPUT_PATH}...")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(INPUT_PATH)
    )

    orig_rows = df.count()
    orig_cols = len(df.columns)
    print("Input rows:", orig_rows)
    print("Input columns:", orig_cols)

    # Ensure Start_Time and End_Time are timestamp types
    df = df.withColumn("Start_Time_TS", to_timestamp(col("Start_Time")))
    df = df.withColumn("End_Time_TS", to_timestamp(col("End_Time")))

    # 1. Accident_Year: extract year from Start_Time
    df = df.withColumn("Accident_Year", year(col("Start_Time_TS")))

    # 2. Accident_Month: extract month number (1-12) from Start_Time
    df = df.withColumn("Accident_Month", month(col("Start_Time_TS")))

    # 3. Accident_DayOfWeek: extract day of week from Start_Time (1 = Sunday, 2 = Monday, ..., 7 = Saturday)
    df = df.withColumn("Accident_DayOfWeek", dayofweek(col("Start_Time_TS")))

    # 4. Accident_Hour: extract hour of day (0-23) from Start_Time
    df = df.withColumn("Accident_Hour", hour(col("Start_Time_TS")))

    # 5. Is_Weekend: 1 for Saturday (7) or Sunday (1), otherwise 0
    df = df.withColumn(
        "Is_Weekend",
        when(col("Accident_DayOfWeek").isin(1, 7), 1).otherwise(0)
    )

    # 6. Is_Rush_Hour: Weekday AND (07:00-10:00 OR 16:00-19:00)
    # Morning rush: hours 7, 8, 9 (07:00:00 to 09:59:59)
    # Evening rush: hours 16, 17, 18 (16:00:00 to 18:59:59)
    is_rush_hour_condition = (
        (col("Is_Weekend") == 0) & (
            ((col("Accident_Hour") >= 7) & (col("Accident_Hour") < 10)) |
            ((col("Accident_Hour") >= 16) & (col("Accident_Hour") < 19))
        )
    )
    df = df.withColumn(
        "Is_Rush_Hour",
        when(is_rush_hour_condition, 1).otherwise(0)
    )

    # 7. Accident_Duration_Minutes: difference between End_Time and Start_Time in minutes
    # If invalid (e.g. negative duration) or missing, leave null
    duration_seconds = col("End_Time_TS").cast("long") - col("Start_Time_TS").cast("long")
    duration_minutes = spark_round(duration_seconds / 60.0, 2)
    df = df.withColumn(
        "Accident_Duration_Minutes",
        when((duration_minutes.isNotNull()) & (duration_minutes >= 0), duration_minutes).otherwise(None)
    )

    # 8. Keep important existing fields and newly created time/date features
    selected_columns = [
        "ID",
        "Severity",
        "Start_Time",
        "End_Time",
        "Start_Lat",
        "Start_Lng",
        "Distance(mi)",
        "Temperature(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Weather_Condition",
        "Wind_Direction",
        "State",
        "City",
        "Street",
        "County",
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "Railway",
        "Roundabout",
        "Stop",
        "Traffic_Signal",
        "Station",
        "Weather_Timestamp",
        "Accident_Year",
        "Accident_Month",
        "Accident_DayOfWeek",
        "Accident_Hour",
        "Is_Weekend",
        "Is_Rush_Hour",
        "Accident_Duration_Minutes",
    ]

    df = df.select(*selected_columns)

    # Deterministic export: sort by ID
    df = df.orderBy("ID")

    final_rows = df.count()
    final_cols = len(df.columns)
    print("Engineered rows:", final_rows)
    print("Engineered columns:", final_cols)

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Collecting {final_rows} records for local CSV export...")
    rows = df.collect()
    columns = df.columns

    print(f"Writing records to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])

    print(f"Success: Engineered dataset successfully written to {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
