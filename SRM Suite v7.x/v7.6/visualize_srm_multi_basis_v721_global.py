# visualize_srm_multi_basis_v721_global.py
# v7.2.1 Analysis Tool — Global Multi-Folder CSV Picker for SRM Basis Comparison

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# === CONFIG ===
ROOT_DIR = Path("experiments")
SAVE_DIR = ROOT_DIR / "visualizations_v721"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# === SCAN ALL CSV FILES ===
print("Scanning all SRM CSVs under experiments/ ...")
csv_files = list(ROOT_DIR.rglob("srm_data_*group_all.csv"))

if not csv_files:
    print("No SRM CSV files found.")
    exit(1)

print(f"Found {len(csv_files)} SRM CSV file(s):")
for i, file in enumerate(csv_files):
    basis_label = file.stem.replace("srm_data_", "").replace("group_all", "").strip("_")
    print(f"  {i+1}: {basis_label}  [{file.parent.parent.name}]")

# === PROMPT USER TO SELECT MULTIPLE ===
print("\nEnter the numbers of the files you want to compare (e.g. 1,2,5):")
selection = input("Your selection: ").strip()
selected_indices = [int(s)-1 for s in selection.split(",") if s.strip().isdigit()]
selected_files = [csv_files[i] for i in selected_indices if 0 <= i < len(csv_files)]

if not selected_files:
    print("No valid selections made. Exiting.")
    exit(1)

# === LOAD DATA ===
srm_data = []
basis_labels = []
for file in selected_files:
    try:
        df = pd.read_csv(file)
        label = file.stem.replace("srm_data_", "").replace("group_all", "").strip("_")
        df["basis"] = label
        srm_data.append(df)
        basis_labels.append(label)
        print(f"✓ Loaded: {file.name} ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ Error loading {file.name}: {e}")

if not srm_data:
    print("Failed to load any data.")
    exit(1)

# === PLOT 1: POLAR ===
print("\nGenerating Polar Resonance Plot...")
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
colors = cm.tab10(np.linspace(0, 1, len(srm_data)))

for df, color, label in zip(srm_data, colors, basis_labels):
    angles = np.deg2rad(df["angle_deg"])
    values = df["mean_similarity"]
    ax.plot(angles, values, label=label, color=color)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("🦇 Full Bat Country Compass Overlay (Multi-Basis)")
ax.legend(loc='lower left', fontsize='small')
polar_path = SAVE_DIR / "srm_polar_overlay_multibasis.png"
plt.savefig(polar_path, bbox_inches='tight')
plt.close()
print(f"Saved polar plot to: {polar_path}")

# === PLOT 2: PEAK SUMMARY ===
print("Generating Peak Similarity Summary Bar Chart...")
summary = []
for df, label in zip(srm_data, basis_labels):
    if df.empty:
        continue
    peak_row = df.loc[df["mean_similarity"].idxmax()]
    summary.append({
        "basis": label,
        "peak_angle": int(peak_row["angle_deg"]),
        "peak_sim": peak_row["mean_similarity"]
    })

if summary:
    summary_df = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary_df["basis"], summary_df["peak_sim"], color="cornflowerblue")
    for i, row in summary_df.iterrows():
        ax.text(i, row["peak_sim"] + 0.002, f"{row['peak_angle']}°", ha='center', fontsize=9)
    ax.set_ylim(0, max(summary_df["peak_sim"]) + 0.05)
    ax.set_ylabel("Peak Cosine Similarity")
    ax.set_title("Peak SRM Similarity and Direction by Basis")
    bar_path = SAVE_DIR / "srm_peak_summary_multibasis.png"
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved peak summary bar chart to: {bar_path}")
else:
    print("No summary data to plot.")

print("\n✅ Done. Multi-basis SRM visualizations complete.")
