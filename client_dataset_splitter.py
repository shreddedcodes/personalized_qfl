"""
This file splits the dataset into 2 non iid data and puts each in a csv
The dataset is split on the basis of PC1. 
client1.csv : 2/3 of the dataset is the lower half of PC1 and 1/3 of the dataset is the upper half of PC1
client2.csv: 2/3 of the dataset is the upper half of PC1 and 1/3 of the dataset is the lower half of PC1
"""

import pandas as pd
import numpy as np

# Load your dataset (columns: 'PC1', 'PC2', 'label')
df = pd.read_csv("breast_cancer_dataset_pca.csv")

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sort by PC1
df_sorted = df.sort_values("PC1").reset_index(drop=True)

# Determine split index for equal halves
n_total = len(df_sorted)
half_n = n_total // 2

# Find the PC1 value at the split point
pc1_split = df_sorted.loc[half_n, "PC1"]

# Initial split
client1_df = df_sorted[df_sorted['PC1'] <= pc1_split].copy()
client2_df = df_sorted[df_sorted['PC1'] > pc1_split].copy()

# Determine overlap size (1/2 of each client)
overlap_size = int(0.5 * len(client1_df))  # both clients are equal size, so either works

# Select overlap samples randomly
overlap_client1 = client1_df.sample(overlap_size, random_state=42)
overlap_client2 = client2_df.sample(overlap_size, random_state=42)

# Add overlap to each client
client1_final = pd.concat([client1_df, overlap_client2]).reset_index(drop=True)
client2_final = pd.concat([client2_df, overlap_client1]).reset_index(drop=True)

# Shuffle final clients
client1_final = client1_final.sample(frac=1, random_state=42).reset_index(drop=True)
client2_final = client2_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
client1_final.to_csv("client1.csv", index=False)
client2_final.to_csv("client2.csv", index=False)

print("Client CSVs created!")
print("Client 1 size:", len(client1_final))
print("Client 2 size:", len(client2_final))
print("Overlap size:", overlap_size)


import matplotlib.pyplot as plt
import pandas as pd

# Assume client1_final and client2_final exist

# Identify overlaps
overlap = pd.merge(client1_final, client2_final, how='inner', on=['PC1','PC2','binary targets'])

# Unique points
client1_unique = client1_final.merge(overlap, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)
client2_unique = client2_final.merge(overlap, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)

# -----------------------
# Combined plot
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(client1_unique['PC1'], client1_unique['PC2'], color='blue', label='Client 1 Unique', alpha=0.6)
plt.scatter(client2_unique['PC1'], client2_unique['PC2'], color='red', label='Client 2 Unique', alpha=0.6)
plt.scatter(overlap['PC1'], overlap['PC2'], color='green', label='Overlap', alpha=0.8, marker='x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Combined Client Data')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# Client 1 plot
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(client1_unique['PC1'], client1_unique['PC2'], color='blue', label='Client 1 Unique', alpha=0.6)
plt.scatter(overlap['PC1'], overlap['PC2'], color='green', label='Overlap', alpha=0.8, marker='x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Client 1 Data')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# Client 2 plot
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(client2_unique['PC1'], client2_unique['PC2'], color='red', label='Client 2 Unique', alpha=0.6)
plt.scatter(overlap['PC1'], overlap['PC2'], color='green', label='Overlap', alpha=0.8, marker='x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Client 2 Data')
plt.legend()
plt.grid(True)
plt.show()











