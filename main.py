import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from code.load_dataset import load_dataset
from code.get_predictions import get_predictions
from code.error_analysis import analyze_errors
from code.metrics import MyMeanAveragePrecision
from code.error_impact import calculate_error_impact
from code.visualise import show_image_with_boxes

print("Loading dataset...")

images_df, targets_df = load_dataset()

print("Dataset Loaded")

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# ---------------------------
# Predictions
# ---------------------------
preds_df = get_predictions(images_df)

print("\nSample Predictions:")
print(preds_df.head())

# ---------------------------
# VISUALIZATION FIRST ✅
# ---------------------------
image_id = images_df.iloc[0]["id"]
file_name = images_df.iloc[0]["file_name"]

image_path = Path("dataset/val2017") / file_name

print("\nImage Path:", image_path)

show_image_with_boxes(str(image_path), preds_df, targets_df, image_id)

# ---------------------------
# Error Analysis
# ---------------------------
valid_image_ids = preds_df["image_id"].unique()
targets_df = targets_df[targets_df["image_id"].isin(valid_image_ids)]

errors_df = analyze_errors(targets_df, preds_df)

print("\nError Analysis Results:")
print(errors_df.head())

errors_df.to_csv(results_dir / "error_analysis.csv", index=False)
print("Saved: error_analysis.csv")

# ---------------------------
# CLEAN ERROR COUNTS
# ---------------------------
counts = errors_df["error_type"].value_counts()

print("\nError Counts:")
for err, cnt in counts.items():
    print(f"{err:<5} : {cnt}")

# ---------------------------
# ERROR DISTRIBUTION PLOT
# ---------------------------
plt.figure()
counts.plot(kind="bar")
plt.title("Error Distribution")
plt.xlabel("Error Type")
plt.ylabel("Count")
plt.show()

# ---------------------------
# Error Impact
# ---------------------------
metric = MyMeanAveragePrecision(0.5)

impact = calculate_error_impact(
    "mAP",
    metric,
    errors_df,
    targets_df,
    preds_df
)

print("\nError Impact:")
for k, v in impact.items():
    print(f"{k:<5} : {v:.6f}")

with open(results_dir / "error_impact.json", "w") as f:
    json.dump(impact, f, indent=4)

print("Saved: error_impact.json")

# ---------------------------
# ERROR IMPACT PLOT
# ---------------------------
impact_plot = {k: v for k, v in impact.items() if k != "mAP"}

plt.figure()
plt.bar(impact_plot.keys(), impact_plot.values())
plt.title("Error Impact on mAP")
plt.xlabel("Error Type")
plt.ylabel("mAP Change")
plt.show()