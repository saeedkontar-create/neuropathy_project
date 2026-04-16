import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# LOAD CSV
# =========================
df = pd.read_csv("labels.csv")

# folder لحفظ الصور
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 0) TOTAL NUMBER OF STEPS PER GROUP
# =========================
steps_per_group = df.groupby("label").size().reset_index(name="num_steps")

plt.figure(figsize=(8,5))
bars = plt.bar(steps_per_group["label"], steps_per_group["num_steps"])

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.title("Number of Steps per Group")
plt.xlabel("Group")
plt.ylabel("Number of Steps")
plt.tight_layout()
plt.savefig(f"{output_dir}/steps_per_group.png")
plt.close()

# =========================
# 1) NUMBER OF STEPS PER PATIENT
# =========================
steps_per_patient = df.groupby(["label", "patient_id"]).size().reset_index(name="num_steps")

for group in steps_per_patient["label"].unique():
    group_df = steps_per_patient[steps_per_patient["label"] == group]

    plt.figure(figsize=(10,5))
    bars = plt.bar(group_df["patient_id"], group_df["num_steps"])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 str(int(height)), ha='center', va='bottom')

    plt.title(f"Number of Steps per Patient - {group}")
    plt.xlabel("Patient ID")
    plt.ylabel("Number of Steps")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # ✅ save
    plt.savefig(f"{output_dir}/steps_per_patient_{group}.png")
    plt.close()


# =========================
# 2) TOTAL FRAMES PER PATIENT
# =========================
frames_per_patient = df.groupby(["label", "patient_id"])["num_frames"].sum().reset_index()

for group in frames_per_patient["label"].unique():
    group_df = frames_per_patient[frames_per_patient["label"] == group]

    plt.figure(figsize=(10,5))
    bars = plt.bar(group_df["patient_id"], group_df["num_frames"])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 str(int(height)), ha='center', va='bottom')

    plt.title(f"Total Frames per Patient - {group}")
    plt.xlabel("Patient ID")
    plt.ylabel("Number of Frames")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # ✅ save
    plt.savefig(f"{output_dir}/frames_per_patient_{group}.png")
    plt.close()


# =========================
# 3) FRAMES PER STEP
# =========================
for group in df["label"].unique():
    group_df = df[df["label"] == group]

    plt.figure(figsize=(12,5))
    bars = plt.bar(group_df["step_id"], group_df["num_frames"])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 str(int(height)), ha='center', va='bottom', fontsize=7)

    plt.title(f"Frames per Step - {group}")
    plt.xlabel("Step ID")
    plt.ylabel("Number of Frames")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # ✅ save
    plt.savefig(f"{output_dir}/frames_per_step_{group}.png")
    plt.close()

print("All plots saved in 'plots' folder ✅")