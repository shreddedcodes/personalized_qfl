"""
app.py — PQFL Live Prediction Demo
Personalized Quantum Federated Learning on the Breast Cancer Wisconsin (Original) Dataset

Required files in the same directory as app.py:
  scaler.pkl
  pca_model.pkl
  parameterized_qc_2.qpy
  QuantumNeuralNetwork.py
  Qfuncs5.py
  saved_weights/global_weights.npy
  saved_weights/client00_weights.npy
  saved_weights/client01_weights.npy
"""

import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PQFL Demo", page_icon="⚛️", layout="wide")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚛️ Personalized Quantum Federated Learning")
st.markdown("**Breast Cancer Wisconsin (Original) Dataset** — Live Prediction Demo")
st.markdown(
    "Enter tumor biopsy features using the sliders in the sidebar. "
    "The app runs inference through the quantum circuit using three separately trained "
    "weight vectors: the **global** federated model and the two **personalized** client models."
)
st.divider()

# ── Load all assets once ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    from qiskit import qpy
    from QuantumNeuralNetwork import QuantumNeuralNetwork

    scaler = pickle.load(open("scaler.pkl", "rb"))
    pca    = pickle.load(open("pca_model.pkl", "rb"))

    with open("parameterized_qc_2.qpy", "rb") as f:
        qc = qpy.load(f)[0]

    def load_weights(path):
        """Load a weight vector saved as np.array([w_vec], dtype=object)."""
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            return np.array(arr[0], dtype=float).ravel()
        return np.array(arr, dtype=float).ravel()

    w_global = load_weights("saved_weights/global_weights.npy")
    w_c00    = load_weights("saved_weights/client00_weights.npy")
    w_c01    = load_weights("saved_weights/client01_weights.npy")

    # Dummy QNN used only for _QNN_output() — training data is irrelevant
    dummy_x = np.zeros((2, 2))
    dummy_y = np.array([0, 1])
    model = QuantumNeuralNetwork(
        qc=qc,
        x_train=dummy_x,
        binary_targets_train=dummy_y,
        x_test=dummy_x,
        binary_targets_test=dummy_y,
        epoch=1
    )

    return scaler, pca, model, w_global, w_c00, w_c01


def run_inference(model, w_vec, x2):
    """
    Run a single forward pass through the QNN.
    x2: numpy array of shape (2,) — the PCA-compressed input.
    Returns (label_str, prob_malignant_float).
    """
    prob = float(model._QNN_output(x2, w_vec.copy()))
    label = "🔴 MALIGNANT" if prob >= 0.5 else "🟢 BENIGN"
    return label, prob


# ── Sidebar: feature sliders ──────────────────────────────────────────────────
st.sidebar.header("🔬 Tumor Biopsy Features")
st.sidebar.markdown(
    "All features are rated **1 – 10** by a pathologist. "
    "1 = most normal, 10 = most abnormal."
)
st.sidebar.divider()

# (display_name, col_name, default, tooltip)
FEATURES = [
    ("Clump Thickness",             "Clump_thickness",             4,
     "Single-layer cells → 1. Multi-layer → high."),
    ("Uniformity of Cell Size",     "Uniformity_of_cell_size",     1,
     "Benign cells are uniform. Malignant cells vary in size."),
    ("Uniformity of Cell Shape",    "Uniformity_of_cell_shape",    1,
     "Benign cells have regular shapes."),
    ("Marginal Adhesion",           "Marginal_adhesion",           1,
     "Normal cells adhere. Malignant may separate."),
    ("Single Epithelial Cell Size", "Single_epithelial_cell_size", 2,
     "Enlarged epithelial cells indicate malignancy."),
    ("Bare Nuclei",                 "Bare_nuclei",                 1,
     "Nuclei without cytoplasm — common in malignant cells."),
    ("Bland Chromatin",             "Bland_chromatin",             3,
     "Uniform texture = benign. Coarse = malignant."),
    ("Normal Nucleoli",             "Normal_nucleoli",             1,
     "Small in benign; prominent/multiple in malignant."),
    ("Mitoses",                     "Mitoses",                     1,
     "Cell division rate. High → malignant."),
]

# Session state to allow preset buttons to override sliders
if "preset" not in st.session_state:
    st.session_state.preset = None

PRESETS = {
    "benign":     [2, 1, 1, 1, 2, 1, 1, 1, 1],
    "malignant":  [8, 8, 8, 5, 5, 10, 7, 8, 1],
    "borderline": [4, 3, 3, 2, 2, 4, 3, 2, 1],
}

feature_values = {}
for i, (display, col, default, tip) in enumerate(FEATURES):
    # If a preset was selected, use preset value; otherwise use default
    if st.session_state.preset and col:
        preset_key = st.session_state.preset
        val = PRESETS[preset_key][i]
    else:
        val = default
    feature_values[col] = st.sidebar.slider(display, 1, 10, val, help=tip)

st.sidebar.divider()
st.sidebar.markdown("**Quick presets:**")
col_a, col_b, col_c = st.sidebar.columns(3)
if col_a.button("🟢 Benign"):
    st.session_state.preset = "benign"
    st.rerun()
if col_b.button("🔴 Malignant"):
    st.session_state.preset = "malignant"
    st.rerun()
if col_c.button("⚪ Border"):
    st.session_state.preset = "borderline"
    st.rerun()
if st.sidebar.button("↺ Clear preset"):
    st.session_state.preset = None
    st.rerun()

# ── Main panel ────────────────────────────────────────────────────────────────
predict_btn = st.button("⚡  Run Prediction", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Running quantum circuit inference — this may take a few seconds..."):
        try:
            scaler, pca, model, w_global, w_c00, w_c01 = load_assets()

            # Build raw feature vector in the exact order the scaler expects
            raw = np.array([[feature_values[col] for _, col, _, _ in FEATURES]], dtype=float)

            # Preprocess: standardise → PCA → 2 features
            scaled     = scaler.transform(raw)
            compressed = pca.transform(scaled)[0]   # shape (2,)

            # Run inference with all three weight vectors
            label_g,  prob_g  = run_inference(model, w_global, compressed)
            label_c00, prob_c00 = run_inference(model, w_c00,   compressed)
            label_c01, prob_c01 = run_inference(model, w_c01,   compressed)

            # ── Results ───────────────────────────────────────────────────────
            st.markdown("### 📊 Prediction Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🌐 Global Model")
                st.metric("Prediction", label_g,
                          delta=f"P(malignant) = {prob_g:.3f}")
                st.progress(prob_g, text=f"{prob_g*100:.1f}% malignant")

            with col2:
                st.markdown("#### 👤 Client 00 (Personalized)")
                st.metric("Prediction", label_c00,
                          delta=f"P(malignant) = {prob_c00:.3f}")
                st.progress(prob_c00, text=f"{prob_c00*100:.1f}% malignant")

            with col3:
                st.markdown("#### 👤 Client 01 (Personalized)")
                st.metric("Prediction", label_c01,
                          delta=f"P(malignant) = {prob_c01:.3f}")
                st.progress(prob_c01, text=f"{prob_c01*100:.1f}% malignant")

            st.divider()

            # ── PCA compression ───────────────────────────────────────────────
            st.markdown("### 🔭 PCA Compression")
            st.markdown(
                "Your 9 biopsy features were standardised and compressed to "
                "**2 principal components** before entering the quantum circuit:"
            )
            pc1, pc2 = st.columns(2)
            pc1.metric("PC1", f"{compressed[0]:.4f}")
            pc2.metric("PC2", f"{compressed[1]:.4f}")

            st.divider()

            # ── Personalization explanation ────────────────────────────────────
            st.markdown("### 💡 Personalization Analysis")

            labels = [label_g, label_c00, label_c01]
            preds  = [prob_g >= 0.5, prob_c00 >= 0.5, prob_c01 >= 0.5]
            all_agree = len(set(preds)) == 1

            if all_agree:
                st.success(
                    "✅ All three models agree on this sample. "
                    "Clear-cut cases produce consensus across the global and personalized models."
                )
            else:
                st.warning(
                    "⚠️ The models **disagree** on this sample. "
                    "This is personalization in action — each client was fine-tuned on a "
                    "different data distribution, so they produce different decisions on "
                    "ambiguous inputs. Try the **⚪ Borderline** preset to see this effect most clearly."
                )

            with st.expander("📖 How personalization works in PQFL", expanded=not all_agree):
                st.markdown("""
**Why do the clients have different weights?**

During federated training, the dataset was split by PC1 median into two non-IID partitions.
Client 00 saw the lower-PC1 half (different malignant-to-benign ratio than Client 01).
Each client's circuit was fine-tuned on its own local data using the **Moreau envelope**:

$$\\mathcal{L}_i(\\omega_i) + \\frac{\\lambda}{2}\\|\\omega_i - \\omega^t\\|^2$$

The proximal term keeps each client close to the global model while allowing local adaptation.

**How does FedTPR aggregation work?**

After each round, the server computes:

$$\\alpha_i = \\frac{\\text{TPR}_i}{\\sum_j \\text{TPR}_j}$$

Clients achieving higher True Positive Rate (better malignancy detection) contribute
more to the next global model. This steers the global model toward clinical sensitivity.

**The result:** Personalized models diverge from the global model on borderline samples,
reflecting their calibration to different local data distributions.
                """)

        except FileNotFoundError as e:
            st.error(
                f"Missing file: `{e.filename}`\n\n"
                "Make sure the following are present alongside `app.py`:\n"
                "- `scaler.pkl`\n"
                "- `pca_model.pkl`\n"
                "- `parameterized_qc_2.qpy`\n"
                "- `QuantumNeuralNetwork.py`\n"
                "- `Qfuncs5.py`\n"
                "- `saved_weights/global_weights.npy`\n"
                "- `saved_weights/client00_weights.npy`\n"
                "- `saved_weights/client01_weights.npy`"
            )
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.exception(e)

else:
    # ── Default state: show current feature values ────────────────────────────
    import pandas as pd
    st.markdown("### Current Feature Values")
    display_dict = {
        name: feature_values[col]
        for name, col, _, _ in FEATURES
    }
    st.dataframe(
        pd.DataFrame(display_dict, index=["Score (1–10)"]),
        use_container_width=True
    )
    st.info(
        "👈 Adjust the sliders in the sidebar, then press **⚡ Run Prediction**.\n\n"
        "Try the **⚪ Borderline** preset to see a case where the personalized "
        "models disagree with each other — demonstrating the effect of "
        "Moreau envelope personalization on ambiguous inputs."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PQFL Framework · SSN College of Engineering, Chennai · "
    "Quantum circuit: Qiskit · Federated training: Flower · "
    "Personalization: Moreau Envelope (λ=0.001) · Aggregation: FedTPR"
)
