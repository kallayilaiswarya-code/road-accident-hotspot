import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

INPUT_PATH = "data/processed/accidents_clustered.csv"
PREDICTION_OUTPUT_PATH = "output/risk_predictions.csv"
METRICS_OUTPUT_PATH = "output/random_forest_metrics.txt"
FEATURE_IMPORTANCE_OUTPUT_PATH = "output/random_forest_feature_importance.csv"
RANDOM_SEED = 42

NUMERICAL_FEATURES = ["Start_Lat", "Start_Lng", "Distance(mi)", "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Accident_Month", "Accident_DayOfWeek", "Accident_Hour", "Is_Weekend", "Is_Rush_Hour", "Accident_Duration_Minutes"]
CATEGORICAL_FEATURES = ["Weather_Condition", "Wind_Direction", "State"]
BINARY_FEATURES = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "Railway", "Roundabout", "Stop", "Traffic_Signal", "Station"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

def main():
    print(f"Reading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df["Risk_Category"] = df["Severity"].map({1: "Low", 2: "Medium", 3: "High", 4: "High"})
    print("\nRisk-category distribution:")
    print(df["Risk_Category"].value_counts())
    X = df[FEATURES].copy()
    for col in BINARY_FEATURES:
        X[col] = X[col].astype(int)
    y = df["Risk_Category"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numerical", numerical_pipeline, NUMERICAL_FEATURES), ("categorical", categorical_pipeline, CATEGORICAL_FEATURES), ("binary", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES)])
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    print("\nTraining Random Forest...")
    pipeline.fit(X_train, y_train)
    print("\nEvaluating model...")
    y_pred = pipeline.predict(X_test)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=["Low", "Medium", "High"], zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix [Low, Medium, High]:")
    print(matrix)
    os.makedirs("output", exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("Random Forest Risk Prediction Evaluation\n========================================\n\n")
        f.write(f"Training records: {len(X_train)}\nTesting records: {len(X_test)}\nBalanced Accuracy: {balanced_acc:.4f}\n\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix [Low, Medium, High]:\n")
        f.write(str(matrix))
        f.write("\n")
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_OUTPUT_PATH, index=False)
    prediction_df = df.loc[X_test.index, ["ID", "Start_Time", "Start_Lat", "Start_Lng", "Severity", "Risk_Category"]].copy()
    prediction_df["Predicted_Risk"] = y_pred
    probabilities = pipeline.predict_proba(X_test)
    class_names = pipeline.named_steps["model"].classes_
    for i, class_name in enumerate(class_names):
        prediction_df[f"Probability_{class_name}"] = probabilities[:, i]
    prediction_df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    print(f"\nSaved metrics to {METRICS_OUTPUT_PATH}")
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_OUTPUT_PATH}")
    print(f"Saved predictions to {PREDICTION_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
