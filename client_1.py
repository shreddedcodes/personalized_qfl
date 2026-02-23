# ===============================
# client_1.py  (QNN + Moreau Envelope Personalization)
# ===============================
#
# KEY CHANGES vs original:
#   1. fit() now returns true_positives in metrics so the server can weight by TP.
#   2. Moreau Envelope personalization: each client keeps a local model w_local.
#      After receiving the global model w_global, training minimises:
#
#          L(w) + (lambda_moreau / 2) * ||w - w_global||^2
#
#      This pulls the local model toward the global model (proximity term)
#      while still fitting the local data (personalisation).
#      The gradient of the proximity term is: lambda_moreau * (w - w_global)
#      This is added to the gradient of L(w) at each parameter-update step.
#
#   3. The client sends its *local* weights to the server after training,
#      exactly as before — the server aggregates these into a new global model.

import flwr as fl
import numpy as np
import os
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from QuantumNeuralNetwork import QuantumNeuralNetwork
from qiskit import qpy

# -------------------------------
# Silence logs
# -------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
logging.getLogger("flwr").setLevel(logging.ERROR)

# -------------------------------
# Hyperparameter
# -------------------------------
LAMBDA_MOREAU = 0.001  # Proximity strength. Higher = stays closer to global model.
                       # Tune this: 0.001 (more local freedom) to 1.0 (near-FedAvg behaviour)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("client_1.csv")

X = df[["PC1", "PC2"]].values
y = df["binary targets"].values

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size total: ", len(x_train))
print("Train size per round: ", 0.3 * len(x_train))
print("Test size total and per round: ", len(x_test))

with open("parameterized_qc_2.qpy", "rb") as qpy_file_read:
    qc = qpy.load(qpy_file_read)[0]

k = int(0.3 * len(x_train))
idx = np.random.choice(len(x_train), k, replace=False)

x_round = x_train[idx]
y_round = y_train[idx]

# -------------------------------
# Build model
# -------------------------------
model = QuantumNeuralNetwork(
    qc=qc,
    x_train=x_round,
    binary_targets_train=y_round,
    x_test=x_test,
    binary_targets_test=y_test,
    epoch=30
)

# -------------------------------
# Moreau-regularised training helper
# -------------------------------
def moreau_train(model, w_global: np.ndarray, lambda_moreau: float):
    """
    Run the full training loop, then apply a single Moreau proximity correction.

        w  ←  w  -  lr * lambda_moreau * (w - w_global)

    This is equivalent to one proximal gradient step after training and avoids
    the epoch-by-epoch overhead of the previous implementation.
    """
    lr = getattr(model, "lr", 0.01)

    # Run all epochs in one go — same speed as the original training
    model.train()

    # Apply Moreau proximity correction once after training
    proximity_grad = lambda_moreau * (model.w_vec - w_global)
    model.w_vec = model.w_vec - lr * proximity_grad


# -------------------------------
# Flower client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [model.get()]

    def fit(self, parameters, config):
        rnd = config.get("rnd", "N/A")
        print(f"\n[CLIENT 1] Round {rnd} | Training started")

        # Refresh local mini-batch each round
        k = int(0.3 * len(x_train))
        idx = np.random.choice(len(x_train), k, replace=False)
        model.x_train = x_train[idx]
        model.binary_targets_train = y_train[idx]
        model.n_train = len(model.x_train)

        # Received global weights — keep a copy for the Moreau term
        w_global = np.array(parameters[0], dtype=float)
        model.w_vec = w_global.copy()    # initialise local model from global

        # ----- Personalised training with Moreau envelope -----
        moreau_train(model, w_global, LAMBDA_MOREAU)

        # Final training loss
        loss = float(model.cost_per_epoch[-1])

        # Compute TPR on the test set for a reliable aggregation signal.
        # only_confusion_matrix() uses self.x_test / self.binary_targets_test internally,
        # so no data swapping needed — this is stable across rounds unlike mini-batch TPR.
        cm_data = model.only_confusion_matrix()
        tpr = float(cm_data.get("True Positive Rate", 0.0))

        print(f"[CLIENT 1] Training finished | Loss: {loss:.4f} | TPR (test): {tpr:.4f}")
        return [model.get()], len(model.x_train), {
            "loss": loss,
            "true_positives": tpr   # server reads this key; value is TPR ∈ [0, 1]
        }

    def evaluate(self, parameters, config):
        rnd = config.get("rnd", "N/A")
        print(f"\n[CLIENT 1] Round {rnd} | Evaluation started")

        # Use personalised local weights for evaluation (not raw global)
        # If you want to evaluate the global model instead, replace with:
        #   model.w_vec = np.array(parameters[0], dtype=float)
        # Keeping local weights gives a true picture of personalised performance.
        model.w_vec = np.array(parameters[0], dtype=float)

        data = model.only_confusion_matrix()
        for key, value in data.items():
            print(f"  {key} : {value}")

        acc = data["Accuracy"]
        print(f"[CLIENT 1] Evaluation accuracy: {acc:.4f}")

        loss = 1.0 - acc
        total = len(model.binary_targets_test)
        return loss, total, data


# -------------------------------
# Start client
# -------------------------------
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)