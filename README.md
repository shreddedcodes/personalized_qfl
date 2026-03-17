# ⚛️ PQFL — Personalized Quantum Federated Learning
### Breast Cancer Wisconsin (Original) Dataset

A federated learning framework that trains a **Quantum Neural Network (QNN)** across multiple clients using the **Flower** framework, with **Moreau Envelope personalization** and **FedTPR aggregation**. Includes a live **Streamlit** demo app for interactive inference.

---

## 📁 Project Structure

```
pqfl/
├── app.py                          # Streamlit live prediction demo
├── server.py                       # Flower federated server (FedTPR strategy)
├── client_1.py                     # Federated client 1 (Moreau personalization)
├── client_2.py                     # Federated client 2 (Moreau personalization)
├── QuantumNeuralNetwork.py         # QNN implementation (parameter-shift rule)
├── Qfuncs5.py                      # Qiskit utility helpers
├── parameterized_qc_2.qpy          # Serialized parameterized quantum circuit
├── scaler.pkl                      # Fitted StandardScaler (9 features)
├── pca_model.pkl                   # Fitted PCA model (9 → 2 components)
├── breast_cancer_dataset_pca.csv   # Full PCA-compressed dataset
├── client_1.csv                    # Client 1 local data partition (low PC1 half)
├── client_2.csv                    # Client 2 local data partition (high PC1 half)
├── requirements.txt                # Python dependencies
└── saved_weights/                  # Created automatically after federated training
    ├── global_weights.npy
    ├── client00_weights.npy
    └── client01_weights.npy
```

---

## 🗄️ Dataset

**Breast Cancer Wisconsin (Original) Dataset**

| Field        | Details |
|-------------|---------|
| **Source**  | UCI Machine Learning Repository |
| **URL**     | https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original |
| **Instances** | 699 samples (after removing missing values: 683) |
| **Features** | 9 integer-valued biopsy features, rated 1–10 |
| **Target**  | Binary — Benign (2 → 0) / Malignant (4 → 1) |

### Features Used

| # | Feature Name | Description |
|---|-------------|-------------|
| 1 | Clump Thickness | Single vs. multi-layer cell groupings |
| 2 | Uniformity of Cell Size | Variation in size across cells |
| 3 | Uniformity of Cell Shape | Regularity of cell shape |
| 4 | Marginal Adhesion | Cell adhesion to surrounding tissue |
| 5 | Single Epithelial Cell Size | Size of individual epithelial cells |
| 6 | Bare Nuclei | Nuclei not surrounded by cytoplasm |
| 7 | Bland Chromatin | Texture/uniformity of nuclear chromatin |
| 8 | Normal Nucleoli | Size and prominence of nucleoli |
| 9 | Mitoses | Rate of cell division |

### Preprocessing

The 9 raw features are:
1. **Standardized** using `StandardScaler` (saved as `scaler.pkl`)
2. **Compressed** to 2 principal components using PCA (saved as `pca_model.pkl`)
3. The 2 PCA features (PC1, PC2) are fed directly into the quantum circuit

The dataset is split into two non-IID client partitions by **PC1 median** — Client 1 receives the lower-PC1 half, Client 2 receives the upper-PC1 half.

---

## 💻 Software Requirements

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| `flwr` | 1.26.1 | Federated learning framework |
| `qiskit` | 2.1.2 | Quantum circuit simulation |
| `numpy` | 2.4.2 | Numerical computation |
| `pandas` | 3.0.1 | Data handling |
| `scikit-learn` | 1.8.0 | StandardScaler, PCA, train/test split |
| `matplotlib` | 3.10.8 | Plotting (confusion matrix, learning curve) |
| `ipython` | 8.12.3 | Display utilities |
| `streamlit` | any recent | Demo app UI (not in requirements.txt — install separately) |

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4-core, 2 GHz | 8-core, 3+ GHz |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB free | 5 GB free |
| OS | Windows 10 / macOS 11 / Ubuntu 20.04 | Ubuntu 22.04 / macOS 13+ |
| GPU | Not required | Not required |

> **Note:** The QNN uses **Qiskit's statevector simulator** (exact, shot-based). Training is CPU-bound. Each federated round (30 epochs × mini-batch) takes approximately 2–10 minutes depending on hardware. The full 10-round training run may take **20–90 minutes** total.

---

## ⚙️ Installation

### 1. Clone / Download the Project

Place all project files in a single directory, e.g. `pqfl/`.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install streamlit        # for the demo app
```

> **Verify Qiskit:**
> ```bash
> python -c "import qiskit; print(qiskit.__version__)"
> ```
> Expected output: `2.1.2`

---

## 🚀 Execution Instructions

There are **two ways** to use this project:

### Option A — Run the Full Federated Training Pipeline

This trains the QNN across two federated clients and saves the resulting weight files needed by the demo app.

You need **three separate terminal windows** open in the project directory.

#### Step 1 — Start the Federated Server

```bash
python server.py
```

The server listens on `0.0.0.0:8080` and waits for 2 clients to connect before starting Round 1.

Expected output:
```
[ROUND 1] Server-side aggregation begun...
ALPHA (TPR-weighted) has been calculated
  Client 00 | True Positive Rate = 0.xxxx | Alpha = 0.xxxx
  Client 01 | True Positive Rate = 0.xxxx | Alpha = 0.xxxx
...
[FINAL ROUND] Saving weights to disk...
  saved_weights/global_weights.npy
  saved_weights/client00_weights.npy
  saved_weights/client01_weights.npy
```

#### Step 2 — Start Client 1 (in a new terminal)

```bash
python client_1.py
```

Expected output:
```
Train size total:  XXX
Train size per round:  XX.X
Test size total and per round:  XX

[CLIENT 1] Round 1 | Training started
============================================================
Epoch 1
Cost Function :  0.XXXX
Model Error   :  XX.XX %
...
[CLIENT 1] Training finished | Loss: 0.XXXX | TPR (test): 0.XXXX
```

#### Step 3 — Start Client 2 (in a new terminal)

```bash
python client_2.py
```

Output mirrors Client 1 but labelled `[CLIENT 2]`.

> **Important:** Start the **server first**, then both clients within a few seconds of each other. The server waits for both clients before beginning Round 1.
>
> After all 10 rounds complete, the `saved_weights/` directory will be created automatically.

---

### Option B — Run the Streamlit Demo App (Inference Only)

This requires the `saved_weights/` directory to already exist (generated by Option A, or provided separately).

#### Prerequisites

Ensure the following files are present in the project directory:

```
scaler.pkl
pca_model.pkl
parameterized_qc_2.qpy
QuantumNeuralNetwork.py
Qfuncs5.py
saved_weights/global_weights.npy
saved_weights/client00_weights.npy
saved_weights/client01_weights.npy
```

#### Launch the App

```bash
streamlit run app.py
```

Streamlit will open a browser tab at `http://localhost:8501`.

#### Using the App

1. **Adjust sliders** in the left sidebar — each slider maps to one of the 9 biopsy features (rated 1–10).
2. Use the **Quick Presets** buttons (🟢 Benign / 🔴 Malignant / ⚪ Borderline) to load representative sample values instantly.
3. Press **⚡ Run Prediction** to execute inference.
4. The app displays:
   - Prediction label and P(malignant) for the **Global model** and both **Personalized client models**
   - The 2 PCA-compressed values passed to the quantum circuit
   - A personalization analysis explaining whether the models agree or diverge

> **Tip:** Use the **⚪ Borderline** preset to observe disagreement between personalized models — this is the core demonstration of Moreau Envelope personalization on ambiguous inputs.

---

## 🧪 Running the Notebooks

Two Jupyter notebooks are included for exploration and circuit generation:

| Notebook | Purpose |
|---------|---------|
| `1__generate_pqc_copy.ipynb` | Generates and serializes the parameterized quantum circuit (`parameterized_qc_2.qpy`) |
| `3_2_QNN_breast_cancer.ipynb` | End-to-end QNN training and evaluation on the full dataset (non-federated baseline) |

Launch Jupyter and run cells top-to-bottom:

```bash
pip install notebook
jupyter notebook
```

---

## 🧠 Architecture Overview

```
Raw Input (9 features)
       ↓
  StandardScaler
       ↓
    PCA (→ 2 components)
       ↓
Parameterized Quantum Circuit (parameterized_qc_2.qpy)
       ↓
  P(|1⟩) → threshold 0.5 → BENIGN / MALIGNANT
```

**Federated Training:**
- Each round, clients initialize from the global weights, train locally for 30 epochs with Moreau proximal regularization, then send updated weights to the server.
- The server aggregates using **FedTPR**: each client's contribution is weighted by its True Positive Rate on the local test set.
- After 10 rounds, the global and both personalized weight vectors are saved to disk.

---

## 📌 Troubleshooting

| Problem | Fix |
|--------|-----|
| `ModuleNotFoundError: flwr` | Run `pip install flwr==1.26.1` |
| `ModuleNotFoundError: qiskit` | Run `pip install qiskit==2.1.2` |
| `Connection refused` on client startup | Start `server.py` before the clients |
| `FileNotFoundError: saved_weights/...` | Run the full federated training pipeline first (Option A) |
| App loads but inference is very slow | Expected — statevector simulation is CPU-intensive. Wait 10–30 s per prediction. |
| `MismatchError` on circuit parameters | Ensure `parameterized_qc_2.qpy` was not regenerated with different qubit/parameter counts |

---

## 🏫 Credits

**PQFL Framework** — SSN College of Engineering, Chennai

- Quantum Circuit: [Qiskit](https://qiskit.org/)
- Federated Learning: [Flower (flwr)](https://flower.dev/)
- Personalization: Moreau Envelope (λ = 0.001)
- Aggregation: FedTPR (True Positive Rate weighted)
- Dataset: UCI ML Repository — Breast Cancer Wisconsin (Original)
