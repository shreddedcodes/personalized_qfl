import flwr as fl
import numpy as np
import os
import logging
import pickle

# minimise chattiness
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

logging.getLogger("flwr").setLevel(logging.ERROR)

NUM_ROUNDS = 10

class FedTPR(fl.server.strategy.FedAvg):

    def __init__(self):
        super().__init__(
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.eval_config
        )
        self.final_global_weights = None
        self.client_final_weights = {}  # stores last weights per client

    def fit_config(self, rnd: int):
        return {"rnd": rnd}

    def eval_config(self, rnd: int):
        return {"rnd": rnd}

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n[ROUND {rnd}] Server-side aggregation begun...")

        if not results:
            return None, {}

        weights_list = []
        tp_list = []

        for i, (_, fit_res) in enumerate(results):
            w = fit_res.parameters
            metrics = fit_res.metrics
            tp = metrics.get("true_positives", 1.0)

            weights_list.append(w)
            tp_list.append(tp)

            # Save each client's weights every round (overwrites, so final round sticks)
            if rnd == NUM_ROUNDS:
                w_nd = fl.common.parameters_to_ndarrays(w)
                self.client_final_weights[i] = w_nd

        tp_array = np.array(tp_list, dtype=np.float64)

        if np.allclose(tp_array, 0.0):
            alpha = np.ones(len(tp_array)) / len(tp_array)
        else:
            total_tp = tp_array.sum()
            alpha = tp_array / total_tp

        print("\nALPHA (TPR-weighted) has been calculated")
        for i in range(len(tp_array)):
            print(f"  Client {i:02d} | True Positive Rate = {tp_array[i]:.4f} | Alpha = {alpha[i]:.4f}")

        agg_weights = [
            np.zeros_like(w)
            for w in fl.common.parameters_to_ndarrays(weights_list[0])
        ]

        for i, w in enumerate(weights_list):
            w_nd = fl.common.parameters_to_ndarrays(w)
            for j in range(len(w_nd)):
                agg_weights[j] += alpha[i] * w_nd[j]

        aggregated_parameters = fl.common.ndarrays_to_parameters(agg_weights)

        # Save global weights — final round overwrites with the last aggregation
        self.final_global_weights = agg_weights

        # On the last round, save everything to disk
        if rnd == NUM_ROUNDS:
            print("\n[FINAL ROUND] Saving weights to disk...")
            os.makedirs("saved_weights", exist_ok=True)

            np.save("saved_weights/global_weights.npy",
                    np.array(agg_weights, dtype=object))

            for client_id, w_nd in self.client_final_weights.items():
                np.save(f"saved_weights/client{client_id:02d}_weights.npy",
                        np.array(w_nd, dtype=object))

            print("  saved_weights/global_weights.npy")
            for client_id in self.client_final_weights:
                print(f"  saved_weights/client{client_id:02d}_weights.npy")

        print("\nGlobal weights have been calculated. Sending global weights to clients...")
        return aggregated_parameters, {"alpha": alpha.tolist()}

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}

        print(f"\n[ROUND {rnd}] Server-side evaluation aggregation begun...")

        all_losses = []
        all_metrics = []

        for i, (client, eval_res) in enumerate(results):
            loss = eval_res.loss
            metrics = eval_res.metrics

            all_losses.append(loss)
            all_metrics.append(metrics)

            print(f"\nClient {i:02d} | Loss: {loss:.4f}")
            print("Metrics:")
            for key, value in metrics.items():
                print(f"  {key} : {value}")

        avg_loss = np.mean(all_losses)
        print(f"\n[ROUND {rnd}] Average evaluation loss across clients: {avg_loss:.4f}")

        avg_metrics = {}
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)

        print(f"[ROUND {rnd}] Average metrics across clients:")
        for key, value in avg_metrics.items():
            print(f"  {key} : {value:.4f}")

        return avg_loss, {"average_metrics": avg_metrics, "client_metrics": all_metrics}


strategy = FedTPR()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)