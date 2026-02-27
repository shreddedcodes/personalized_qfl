"""
find_borderline.py
Run this locally (needs your breast_cancer_dataset_pca.csv or original dataset + scaler + pca).
Finds the samples whose model predictions are closest to 0.5 across all three models.
"""

import numpy as np
import pickle
import pandas as pd
from qiskit import qpy
from QuantumNeuralNetwork import QuantumNeuralNetwork

# ── Load preprocessing ────────────────────────────────────────────────────────
scaler = pickle.load(open("scaler.pkl", "rb"))
pca    = pickle.load(open("pca_model.pkl", "rb"))

with open("parameterized_qc_2.qpy", "rb") as f:
    qc = qpy.load(f)[0]

# ── Load weights ──────────────────────────────────────────────────────────────
def extract_w(path):
    arr = np.load(path, allow_pickle=True)
    while hasattr(arr, "dtype") and arr.dtype == object:
        arr = arr.flat[0]
    return np.array(arr, dtype=float).ravel()

w_global = extract_w("saved_weights/global_weights.npy")
w_c00    = extract_w("saved_weights/client00_weights.npy")
w_c01    = extract_w("saved_weights/client01_weights.npy")

dummy_x = np.zeros((2, 2))
dummy_y = np.array([0, 1])

def make_model(w):
    m = QuantumNeuralNetwork(
        qc=qc, x_train=dummy_x, binary_targets_train=dummy_y,
        x_test=dummy_x, binary_targets_test=dummy_y, epoch=1
    )
    m.w_vec = w.copy()
    return m

model_global = make_model(w_global)
model_c00    = make_model(w_c00)
model_c01    = make_model(w_c01)

# ── Load original dataset ─────────────────────────────────────────────────────
# Load whichever CSV you have — needs the 9 raw feature columns
df = pd.read_csv("breast_cancer_original.csv")  # adjust filename

feature_cols = [
    "Clump_thickness", "Uniformity_of_cell_size", "Uniformity_of_cell_shape",
    "Marginal_adhesion", "Single_epithelial_cell_size", "Bare_nuclei",
    "Bland_chromatin", "Normal_nucleoli", "Mitoses"
]

X_raw = df[feature_cols].values
y     = df["Class"].values  # 2=benign, 4=malignant (or 0/1 depending on your version)

# ── Run inference on every sample ─────────────────────────────────────────────
results = []
print("Running inference on all samples (this will take a while)...")

for i, row in enumerate(X_raw):
    scaled = scaler.transform(row.reshape(1, -1))
    x_pca  = pca.transform(scaled)[0]

    # Prepare full parameter vector = [weights..., inputs...]
    params_g   = np.concatenate([model_global.w_vec, x_pca])
    params_c00 = np.concatenate([model_c00.w_vec, x_pca])
    params_c01 = np.concatenate([model_c01.w_vec, x_pca])

    p_g   = float(model_global._QNN_output(None, params_g))
    p_c00 = float(model_c00._QNN_output(None, params_c00))
    p_c01 = float(model_c01._QNN_output(None, params_c01))

    # "borderline" = scores close to 0.5, especially where models disagree
    avg_dist_from_half = (abs(p_g - 0.5) + abs(p_c00 - 0.5) + abs(p_c01 - 0.5)) / 3
    models_disagree    = (round(p_g) != round(p_c00)) or (round(p_g) != round(p_c01))

    results.append({
        "index":           i,
        "true_label":      int(y[i]),
        "p_global":        round(p_g, 4),
        "p_c00":           round(p_c00, 4),
        "p_c01":           round(p_c01, 4),
        "avg_dist_half":   round(avg_dist_from_half, 4),
        "models_disagree": models_disagree,
        **{col: int(row[j]) for j, col in enumerate(feature_cols)}
    })

    if i % 20 == 0:
        print(f"  {i}/{len(X_raw)} done...")

results_df = pd.DataFrame(results)

# ── Print the most interesting samples ───────────────────────────────────────
print("\n=== TOP 10 SAMPLES WHERE MODELS DISAGREE ===")
disagree_df = results_df[results_df["models_disagree"]].sort_values("avg_dist_half")
print(disagree_df[["index", "true_label", "p_global", "p_c00", "p_c01"] + feature_cols].head(10).to_string())

print("\n=== TOP 10 MOST BORDERLINE (closest to 0.5, models may agree) ===")
borderline_df = results_df.sort_values("avg_dist_half")
print(borderline_df[["index", "true_label", "p_global", "p_c00", "p_c01"] + feature_cols].head(10).to_string())

results_df.to_csv("inference_results.csv", index=False)
print("\nFull results saved to inference_results.csv")