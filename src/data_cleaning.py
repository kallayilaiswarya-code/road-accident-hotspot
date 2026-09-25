import csv
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

INPUT_PATH = "data/sample/US_Accidents_sample.csv"
OUTPUT_PATH = "data/processed/accidents_cleaned.csv"


def main():
    if "JAVA_HOME" not in os.environ:
        default_java = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
        if os.path.exists(default_java):
            os.environ["JAVA_HOME"] = default_java

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("RoadAccidentDataCleaning")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("Reading accident data...")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(INPUT_PATH)
    )

    orig_rows = df.count()
    orig_cols = len(df.columns)
    print("Original rows:", orig_rows)
    print("Original columns:", orig_cols)

    # Remove duplicate accident records using the unique accident ID.
    df = df.dropDuplicates(["ID"])

    # These fields were completely missing in the development sample,
    # while Wind Chill and Precipitation had very high missingness.
    columns_to_drop = [
        "End_Lat",
        "End_Lng",
        "Wind_Chill(F)",
        "Precipitation(in)",
    ]

    df = df.drop(*columns_to_drop)

    # Fill missing categorical values.
    categorical_columns = [
        "Timezone",
        "Wind_Direction",
        "Weather_Condition",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]

    for column in categorical_columns:
        df = df.fillna({column: "Unknown"})

    # Fill missing numerical values using approximate medians.
    numerical_columns = [
        "Temperature(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
    ]

    medians = {}

    for column in numerical_columns:
        median_value = df.approxQuantile(column, [0.5], 0.01)[0]
        medians[column] = median_value

    df = df.fillna(medians)

    cleaned_rows = df.count()
    cleaned_cols = len(df.columns)
    print("Cleaned rows:", cleaned_rows)
    print("Cleaned columns:", cleaned_cols)

    print("\nCleaned schema:")
    df.printSchema()

    print("\nRemaining missing values:")

    for column in df.columns:
        missing = df.filter(col(column).isNull()).count()

        if missing > 0:
            print(f"{column}: {missing}")

    # Make export deterministic by sorting by ID
    df = df.orderBy("ID")

    print("\nOutput path:", OUTPUT_PATH)

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Collecting cleaned records for local export...")
    columns = df.columns
    rows = df.collect()

    print(f"Writing {len(rows)} records to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])

    print(f"Success: Cleaned data successfully written to {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()



