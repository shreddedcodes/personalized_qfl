"""
This file splits the dataset into 2 non-IID clients and saves each to a CSV.
The dataset is split on the basis of PC1:
  - client1.csv : majority lower half of PC1, minority upper half
  - client2.csv : majority upper half of PC1, minority lower half

Overlap sampling is STRATIFIED — each client's overlap preserves the original
class ratio, preventing class imbalance from skewing training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load and shuffle dataset
# -------------------------
df = pd.read_csv("breast_cancer_dataset_pca.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------
# Split by PC1 median
# -------------------------
df_sorted = df.sort_values("PC1").reset_index(drop=True)
n_total = len(df_sorted)
half_n = n_total // 2
pc1_split = df_sorted.loc[half_n, "PC1"]

client1_df = df_sorted[df_sorted["PC1"] <= pc1_split].copy()
client2_df = df_sorted[df_sorted["PC1"] > pc1_split].copy()

print("Before overlap:")
print(f"  Client 1 size: {len(client1_df)} | Class distribution:\n{client1_df['binary targets'].value_counts(normalize=True).round(3)}")
print(f"  Client 2 size: {len(client2_df)} | Class distribution:\n{client2_df['binary targets'].value_counts(normalize=True).round(3)}")

# -------------------------
# Stratified overlap sampling
# -------------------------
# Each overlap sample is drawn class-proportionally so neither client ends up
# with a skewed class ratio after the overlap is added.

overlap_size = int(0.5 * min(len(client1_df), len(client2_df)))

def stratified_sample(df, n, random_state=42):
    """
    Sample n rows from df while preserving the class ratio of 'binary targets'.
    """
    return (
        df.groupby("binary targets", group_keys=False)
        .apply(lambda x: x.sample(
            max(1, round(n * len(x) / len(df))),  # proportional count per class
            random_state=random_state
        ))
        .sample(frac=1, random_state=random_state)  # shuffle after groupby
        .reset_index(drop=True)
    )

overlap_from_client2 = stratified_sample(client2_df, overlap_size)  # added to client 1
overlap_from_client1 = stratified_sample(client1_df, overlap_size)  # added to client 2

# -------------------------
# Build final client datasets
# -------------------------
client1_final = pd.concat([client1_df, overlap_from_client2]).sample(frac=1, random_state=42).reset_index(drop=True)
client2_final = pd.concat([client2_df, overlap_from_client1]).sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------
# Verify class balance
# -------------------------
print("\nAfter overlap:")
print(f"  Client 1 size: {len(client1_final)} | Class distribution:\n{client1_final['binary targets'].value_counts(normalize=True).round(3)}")
print(f"  Client 2 size: {len(client2_final)} | Class distribution:\n{client2_final['binary targets'].value_counts(normalize=True).round(3)}")
print(f"\nOverlap size per client: {overlap_size}")

# -------------------------
# Save to CSV
# -------------------------
client1_final.to_csv("client_1.csv", index=False)
client2_final.to_csv("client_2.csv", index=False)
print("\nClient CSVs created!")

# -------------------------
# Visualisation
# -------------------------
overlap = pd.merge(client1_final, client2_final, how="inner", on=["PC1", "PC2", "binary targets"])

client1_unique = (
    client1_final.merge(overlap, how="outer", indicator=True)
    .query('_merge == "left_only"')
    .drop("_merge", axis=1)
)
client2_unique = (
    client2_final.merge(overlap, how="outer", indicator=True)
    .query('_merge == "left_only"')
    .drop("_merge", axis=1)
)

# Combined plot
plt.figure(figsize=(8, 6))
plt.scatter(client1_unique["PC1"], client1_unique["PC2"], color="blue", label="Client 1 Unique", alpha=0.6)
plt.scatter(client2_unique["PC1"], client2_unique["PC2"], color="red", label="Client 2 Unique", alpha=0.6)
plt.scatter(overlap["PC1"], overlap["PC2"], color="green", label="Overlap", alpha=0.8, marker="x")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Combined Client Data")
plt.legend(); plt.grid(True); plt.show()

# Client 1 plot
plt.figure(figsize=(8, 6))
plt.scatter(client1_unique["PC1"], client1_unique["PC2"], color="blue", label="Client 1 Unique", alpha=0.6)
plt.scatter(overlap["PC1"], overlap["PC2"], color="green", label="Overlap", alpha=0.8, marker="x")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Client 1 Data")
plt.legend(); plt.grid(True); plt.show()

# Client 2 plot
plt.figure(figsize=(8, 6))
plt.scatter(client2_unique["PC1"], client2_unique["PC2"], color="red", label="Client 2 Unique", alpha=0.6)
plt.scatter(overlap["PC1"], overlap["PC2"], color="green", label="Overlap", alpha=0.8, marker="x")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Client 2 Data")
plt.legend(); plt.grid(True); plt.show()