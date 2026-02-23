"""
QuantumNeuralNetwork.py  (GENERALIZED)
James Saslow (modified)
Requires Qfuncs5.py

Generalized to support variable number of input features / qubits and variable number
of trainable parameters. Keeps the original public interface unchanged.
"""

#===================================================================================================
# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Qiskit imports (kept as in original)
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, qpy

import Qfuncs5 as qf  # user's helper module (unchanged)

#===================================================================================================

class QuantumNeuralNetwork:
    '''
    Generalized QNN class supporting variable number of inputs/qubits.
    Public methods preserved: train(), get(), confusion_matrix(), learning_graph()
    '''

    def __init__(self, qc, x_train, binary_targets_train, x_test, binary_targets_test, **kwargs):
        '''
        Constructor signature unchanged.

        The code now:
         - Detects number of input features from x_train (n_inputs).
         - Detects number of total parameters in qc (total_params).
         - Assumes the qc parameters are ordered as [w_0, w_1, ..., w_{m-1}, x_0, x_1, ..., x_{n-1}]
           (this preserves the earlier behaviour). So number of trainable weights = total_params - n_inputs.

        **kwargs:
            - epoch (default 30)
            - lr (default 1)
            - w_vec (optional) : length must equal inferred number of weights (otherwise trimmed/expanded)
            - shots (default 500_000)
        '''
        self.qc = qc

        # Convert inputs to numpy arrays and detect shapes
        Xtr = np.array(x_train)
        Xte = np.array(x_test)
        if Xtr.ndim == 1:
            Xtr = Xtr.reshape(-1, 1)
        if Xte.ndim == 1:
            Xte = Xte.reshape(-1, 1)

        self.x_train = Xtr
        self.x_test = Xte

        self.binary_targets_train = np.array(binary_targets_train)
        self.binary_targets_test  = np.array(binary_targets_test)

        # number of training/test samples and input dims
        self.n_train, self.n_inputs = self.x_train.shape
        self.n_test,  _             = self.x_test.shape

        # Determine number of parameters in the circuit
        total_params = len(list(qc.parameters))
        # Expect inputs to be provided as last n_inputs parameters (same behavior as original code).
        n_weights = total_params - self.n_inputs
        if n_weights <= 0:
            raise ValueError("Inferred number of weights <= 0. Check qc.parameters ordering and x_train shape.")

        self.n_weights = n_weights

        # kwargs handling (epoch, lr, w_vec, shots)
        self.epoch = int(kwargs.get('epoch', 30))
        self.lr = float(kwargs.get('lr', 1.0))
        # w_vec: if provided, check length; otherwise random
        if 'w_vec' in kwargs:
            w_vec_in = np.array(kwargs['w_vec'], dtype=float)
            if w_vec_in.size == self.n_weights:
                self.w_vec = w_vec_in.copy()
            elif w_vec_in.size < self.n_weights:
                # pad randomly for missing params
                pad = 2*np.pi*np.random.random(self.n_weights - w_vec_in.size)
                self.w_vec = np.hstack((w_vec_in, pad))
            else:
                # trim if user passed longer vector
                self.w_vec = w_vec_in[:self.n_weights].copy()
        else:
            self.w_vec = 2*np.pi*np.random.random(self.n_weights)

        self.shots = int(kwargs.get('shots', 500_000))

        # cost per epoch storage
        self.cost_per_epoch = np.zeros(self.epoch)

    def get(self):
        '''
        Returns current weight vector
        '''
        return self.w_vec

    def _QNN_output(self, x_vec, omega_vec):
        '''
        Binds parameters and returns probability of measuring |1> on the classifier qubit.
        Works for any number of inputs / weights as long as qc.parameters ordering matches:
        [w0,w1,...,w_{m-1}, x0,x1,...,x_{n-1}]
        '''
        qc = self.qc
        shots = self.shots

        # Classical register used for measurement - same as original assumption
        try:
            c = qc.cregs[0]
        except Exception as e:
            # if no classical registers exist (unlikely given original code), raise meaningful error
            raise RuntimeError("QuantumCircuit must have a classical register for measurement. Original code expects qc.cregs[0].") from e

        # Make sure x_vec is numpy array and has correct length
        x_arr = np.array(x_vec, dtype=float).ravel()
        if x_arr.size != self.n_inputs:
            raise ValueError(f"Length of x_vec ({x_arr.size}) does not match expected number of inputs ({self.n_inputs}).")

        # Prepare ordered parameter values (same assumption as original)
        param_values = np.hstack((omega_vec, x_arr))
        parameters = list(qc.parameters)

        if len(param_values) != len(parameters):
            raise ValueError("Mismatch between prepared parameter vector length and qc.parameters length. "
                             f"Prepared: {len(param_values)}, circuit expects: {len(parameters)}. "
                             "Ensure qc.parameters ordering matches [weights..., inputs...].")

        parameter_values = {param: val for param, val in zip(parameters, param_values)}

        bound_qc = qc.assign_parameters(parameter_values)

        # Use Qfuncs5.Measure to get probabilities. It returns bases, ans (probabilities in big-endian)
        bases, probs = qf.Measure(bound_qc, c, shots=self.shots)
        # We expect a single-qubit readout (or classifier qubit mapped to a single bit).
        # The original code assumed probs has two entries [p0, p1]
        if len(probs) < 2:
            raise RuntimeError("Unexpected measurement result length. Expected at least 2 basis probs (p0,p1).")

        # Return probability of measuring |1>
        # If multiple measured bits exist, original code relied on basis ordering such that index 1 corresponds to '1'
        # We'll return the probability associated with basis '1' if present, else the last element.
        # But original code simply used p0,p1 = probs; p1 returned. We'll preserve that semantics for single-bit classifier.
        if len(probs) == 2:
            p0, p1 = probs
            return p1
        else:
            # If more than 2 bases returned, attempt to find the single-bit '1' basis (i.e., '...1').
            # We'll default to probability of basis with binary string of all zeros except last bit = 1, if bases available.
            # Since Qfuncs5 returns bases as strings like '00', '01', etc., find basis with last char '1' and others '0'.
            for idx, b in enumerate(bases):
                if set(b) <= set('01') and b.endswith('1') and b.count('1') == 1:
                    return probs[idx]
            # fallback: return sum of probabilities of all bases with last bit 1
            p1_total = 0.0
            for idx, b in enumerate(bases):
                if b.endswith('1'):
                    p1_total += probs[idx]
            return p1_total

    def _cost_function(self, p1, d_class):
        '''
        0.5*(d - p)^2 as before
        '''
        return 0.5 * (d_class - p1)**2

    def _live_misclassification_detection(self, p1, d_class):
        '''
        Round p1 to nearest integer and compare to d_class
        '''
        prediction = np.round(p1)
        return not (prediction == d_class)

    def _cost_function_gradient(self, x_vec, omega_vec, d_class):
        '''
        Computes cost, misclassification, and gradient dC/dw using the parameter-shift rule.

        Parameter-shift rule:
            dP/dw_k = 0.5 * ( P(w_k + pi/2) - P(w_k - pi/2) )

        Then dC/dw = dC/dP * dP/dw.

        Returns:
            cost (scalar),
            misclassification (bool),
            dC_dw (np.array of length n_weights)
        '''
        # Compute p1 at current parameters
        p1 = self._QNN_output(x_vec, omega_vec)
        cost = self._cost_function(p1, d_class)
        misclassification = self._live_misclassification_detection(p1, d_class)

        # derivative of cost wrt p: dC/dP = -(d - p)
        dC_dP = -(d_class - p1)

        # parameter-shift for each weight
        dP_dw = np.zeros(self.n_weights, dtype=float)
        shift = np.pi / 2.0

        # Save original omega to avoid side effects
        for k in range(self.n_weights):
            omega_plus = omega_vec.copy()
            omega_minus = omega_vec.copy()
            omega_plus[k] += shift
            omega_minus[k] -= shift

            p_plus = self._QNN_output(x_vec, omega_plus)
            p_minus = self._QNN_output(x_vec, omega_minus)

            # parameter-shift derivative
            dP_dw[k] = 0.5 * (p_plus - p_minus)

        dC_dw = dC_dP * dP_dw

        return cost, misclassification, dC_dw

    def train(self):
        '''
        Gradient descent training using the generalized gradient computation.
        Preserves previous printouts/behaviour.
        '''
        w_vec = self.w_vec
        lr = self.lr
        epoch = self.epoch
        cost_per_epoch = self.cost_per_epoch

        s = self.n_train

        for k in range(epoch):
            total_cost = 0.0
            num_misclassifications = 0
            for i in range(s):
                d_class = self.binary_targets_train[i]
                x_vec = self.x_train[i, :]

                cost, misclassification, dC_dw = self._cost_function_gradient(x_vec, w_vec, d_class)

                # update weights (note original used lr/s * dC_dw)
                w_vec -= (lr / s) * dC_dw
                total_cost += cost / s
                if misclassification:
                    num_misclassifications += 1

            cost_per_epoch[k] = total_cost

            # Keep print statements similar to original
            print('============================================================')
            print('Epoch ' + str(k + 1))
            print('Cost Function : ', total_cost)
            print('Model Error   : ', 100 * num_misclassifications / s, '%')
            print('omega_vec = ', w_vec)
            print('============================================================')
            print(' ')

        # update stored weights
        self.w_vec = w_vec
        self.cost_per_epoch = cost_per_epoch

    def confusion_matrix(self, **kwargs):
        '''
        Evaluates model on test data and prints confusion stats plus two plots:
         - A bar chart of TP, FP, FN, TN
         - A scatter plot of the test data colored by classification result
           If input dimension > 2, we do a PCA projection to 2D for plotting.
        '''
        xlabel = kwargs.get('xlabel', 'x')
        ylabel = kwargs.get('ylabel', 'y')
        vertical_bool = kwargs.get('vertical', False)

        w_vec = self.w_vec
        X_test = self.x_test
        y_test = self.binary_targets_test
        num_data = self.n_test

        TP = TN = FP = FN = 0
        prediction_record = []

        # record projected points for plotting
        # If inputs == 2, use original coords; else compute PCA to 2D
        if self.n_inputs == 2:
            plot_points = X_test.copy()
            plot_xlabel = xlabel
            plot_ylabel = ylabel
        else:
            # PCA via SVD for projection to 2 components
            Xc = X_test - np.mean(X_test, axis=0)
            # compute V (loadings)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            V = Vt.T
            proj = Xc.dot(V[:, :2])
            plot_points = proj
            plot_xlabel = 'PC1'
            plot_ylabel = 'PC2'

        # Evaluate model on test set
        for i in range(num_data):
            x_vec = X_test[i, :]
            prob1 = self._QNN_output(x_vec, w_vec)
            prediction = int(np.round(prob1))
            d_class = int(y_test[i])

            if prediction == d_class:
                if prediction == 0:
                    TN += 1
                    prediction_record.append('red')
                else:
                    TP += 1
                    prediction_record.append('lime')
            else:
                if prediction == 0:
                    FN += 1
                    prediction_record.append('orange')
                else:
                    FP += 1
                    prediction_record.append('green')

        confusion_labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
        confusion_outputs = [TP, FP, FN, TN]

        # Plotting
        if vertical_bool:
            fig, axs = plt.subplots(2, 1, figsize=(6, 10))
            ax_bar = axs[0]
            ax_scatter = axs[1]
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            ax_bar = axs[0]
            ax_scatter = axs[1]

        ax_bar.set_title('Confusion Matrix on Test Data')
        ax_bar.tick_params(axis='x', rotation=35)
        ax_bar.set_ylabel('Occurrences')
        bars = ax_bar.bar(confusion_labels, confusion_outputs, color=['lime', 'green', 'orange', 'red'])
        for bar, count in zip(bars, confusion_outputs):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

        ax_scatter.set_title('Test Data Assignment')
        ax_scatter.scatter(plot_points[:, 0], plot_points[:, 1], marker='o', edgecolor='black', c=prediction_record)
        ax_scatter.set_xlabel(plot_xlabel)
        ax_scatter.set_ylabel(plot_ylabel)
        plt.tight_layout()
        plt.show()

        # Calculate metrics (with safe guards against division by zero)
        accuracy = (TP + TN) / num_data if num_data > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        FPR = 1 - specificity

        data = {
            "Accuracy": accuracy,
            "Precision": precision,
            "True Positive Rate": TPR,
            "Specificity": specificity,
            "False Positive Rate": FPR
        }

        df = pd.DataFrame(data, index=["metrics"])
        display(df)

        
    def only_confusion_matrix(self, **kwargs):

        w_vec = self.w_vec
        X_test = self.x_test
        y_test = self.binary_targets_test
        num_data = self.n_test

        TP = TN = FP = FN = 0
        prediction_record = []


        # Evaluate model on test set
        for i in range(num_data):
            x_vec = X_test[i, :]
            prob1 = self._QNN_output(x_vec, w_vec)
            prediction = int(np.round(prob1))
            d_class = int(y_test[i])

            if prediction == d_class:
                if prediction == 0:
                    TN += 1
                    prediction_record.append('red')
                else:
                    TP += 1
                    prediction_record.append('lime')
            else:
                if prediction == 0:
                    FN += 1
                    prediction_record.append('orange')
                else:
                    FP += 1
                    prediction_record.append('green')

        # Calculate metrics (with safe guards against division by zero)
        accuracy = (TP + TN) / num_data if num_data > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        FPR = 1 - specificity

        data = {
            "Accuracy": accuracy,
            "Precision": precision,
            "True Positive Rate": TPR,
            "Specificity": specificity,
            "False Positive Rate": FPR
        }

        return data

    def learning_graph(self):
        '''
        Plot cost per epoch (same behaviour as original)
        '''
        epoch = self.epoch
        cost_per_epoch = self.cost_per_epoch
        lr = self.lr

        epoch_array = np.arange(1, epoch + 1, 1)

        plt.figure(figsize=(5, 4))
        plt.title('Cost Function vs Epoch for lr = ' + str(lr))
        plt.xlabel('Epoch')
        plt.ylabel('Cost Function')
        plt.plot(epoch_array, cost_per_epoch)
        plt.show()
