# Qfuncs5.py
# Utility helpers for Qiskit: statevector extraction, measurement helpers, plotting.
# Robust and self-contained: uses Statevector for exact probabilities and multinomial sampling for shots.

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import qiskit

# -------------------------
# Private helper functions
# -------------------------
def _bin_gen(number, num_qubits):
    """Generate binary string of length num_qubits (big-endian)."""
    s = format(number, 'b').zfill(num_qubits)
    return s

def _rev_bin_gen(number, num_qubits):
    """Return reversed (little-endian style) binary string of given width."""
    return _bin_gen(number, num_qubits)[::-1]

def _binto10(binary_str):
    """Convert binary string (big-endian) to integer."""
    return int(binary_str, 2)

def _dual_sort(keys, values):
    """
    Sort 'values' according to 'keys' (keys is an array-like of integers),
    returning the values reordered to match sorted(keys).
    """
    keys = np.array(keys)
    values = np.array(values)
    sorted_indices = np.argsort(keys)
    return list(values[sorted_indices])

def _get_statevec(qc):
    """Return numpy statevector (complex) from circuit `qc` (ignores measurements)."""
    # strip measurements then get Statevector
    qc_unitary = _strip_measurements(qc)
    sv = Statevector.from_instruction(qc_unitary)
    return np.array(sv.data, dtype=complex)

def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Return a new QuantumCircuit with measurement instructions removed,
    preserving the same number of qubits.
    """
    new_qc = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        if instr.name == 'measure':
            continue
        # append instruction using indices (safe because qubit ordering is same)
        # recreate the qargs as references into new_qc.qubits
        new_qargs = [ new_qc.qubits[ qc.qubits.index(q) ] for q in qargs ]
        # unitary instructions won't use classical args; pass empty list
        new_qc.append(instr, new_qargs, [])
    return new_qc

# -------------------------
# Public API
# -------------------------
def QiskitVersion():
    """Print current Qiskit version."""
    print(qiskit.__version__)

def ReturnPsi(qc, **kwargs):
    """
    Return or print the statevector (wavefunction) in big-endian ordering.
    kwargs:
      - braket=True  : print in |...> notation
      - zeros=True   : print zero amplitudes as well
      - polar=True   : show polar form amplitude (r exp(1j*theta))
      - precision=N  : rounding precision (default 5)
    """
    dec = int(kwargs.get('precision', 5))
    sv = _get_statevec(qc)
    sv_rounded = np.round(sv, dec)
    num_bases = len(sv_rounded)
    nqubits = int(np.log2(num_bases))

    # Build big-endian ordering (index 0 => |00...0>)
    if kwargs.get('braket', False):
        polar_bool = kwargs.get('polar', False)
        zeros_bool = kwargs.get('zeros', False)
        for i in range(num_bases):
            amp = sv_rounded[i]
            if not zeros_bool and np.abs(amp) == 0:
                continue
            if polar_bool:
                r = np.round(np.abs(amp), dec)
                theta = np.round(np.angle(amp), dec)
                rep = f"{r} exp(1j*{theta})" if r != 0 else "0"
            else:
                rep = amp
            print(f"{rep} |{_bin_gen(i, nqubits)}>")
        return None
    else:
        return sv_rounded

def Measure(qc, c, shots=None, **kwargs):
    """
    Return basis labels and probabilities (or counts) for the classical register `c`.

    Args:
      qc    : QuantumCircuit (may contain measure ops)
      c     : ClassicalRegister instance, or list of Clbit, or single Clbit (the same object you pass when measuring)
      shots : None -> return exact probabilities from statevector
              integer -> sample 'shots' counts from the exact distribution
    kwargs:
      counts=True -> return counts (integers) instead of probabilities

    Returns:
      bases : list of binary strings (big-endian) of length = number of measured classical bits
      values: list of probabilities (floats) or counts (ints)
    """
    # Normalize 'c' to list of Clbit objects in the order of the classical register
    # Accept: ClassicalRegister (has attribute .clbits), list of Clbit, or a single Clbit object
    if hasattr(c, "clbits"):
        clbits = list(c.clbits)
    elif isinstance(c, list):
        clbits = c
    else:
        clbits = [c]

    num_measured = len(clbits)
    if num_measured == 0:
        raise ValueError("No classical bits provided to Measure().")

    # Determine which qubit indices are measured into each classical bit in clbits (in that order)
    # We'll search measurement instructions in qc.data and map classical bits to qubits
    measured_qubits = [None] * num_measured
    # create set for quick membership
    clbit_set = set(clbits)
    for instr, qargs, cargs in qc.data:
        if instr.name == 'measure':
            if len(cargs) == 0:
                continue
            clbit = cargs[0]
            if clbit in clbit_set:
                # find position i in clbits, record corresponding qubit index
                try:
                    i = clbits.index(clbit)
                except ValueError:
                    continue
                qubit = qargs[0]
                measured_qubits[i] = qc.qubits.index(qubit)

    # If any mapping not found, as a fallback assume measured_qubits = [0,1,2,...] (in order)
    if any(m is None for m in measured_qubits):
        measured_qubits = list(range(num_measured))

    # Build unitary (measurement-free) circuit and compute statevector
    qc_unitary = _strip_measurements(qc)
    sv = Statevector.from_instruction(qc_unitary)

    # Use Statevector.probabilities_dict to compute marginal probabilities on measured_qubits
    try:
        prob_dict = sv.probabilities_dict(qargs=measured_qubits)
    except Exception:
        # fallback: compute full probabilities array and marginalize manually
        probs_full = np.abs(np.array(sv.data))**2
        # Map each basis index to the bits of measured_qubits and accumulate
        prob_dict = {}
        for idx, p in enumerate(probs_full):
            # binary string of full system (big-endian)
            full_bits = _bin_gen(idx, int(np.log2(len(probs_full))))
            # extract bits at measured_qubits positions (qbit index mapping: measured_qubits are qubit indices; 
            # we need their positions in the bitstring: bitstring index = num_qubits-1 - qubit_index)
            n = len(full_bits)
            measured_bits = []
            for mq in measured_qubits:
                pos = n - 1 - mq
                measured_bits.append(full_bits[pos])
            key = ''.join(measured_bits)
            prob_dict[key] = prob_dict.get(key, 0.0) + p

    # Build sorted list of bases (big-endian lexical order)
    bases = [_bin_gen(i, num_measured) for i in range(2**num_measured)]
    probs = [ float(prob_dict.get(b, 0.0)) for b in bases ]

    # If shots specified, sample counts from multinomial and return counts or probs accordingly
    if shots is None:
        if kwargs.get('counts', False):
            # approximate counts by scaling probabilities - but better to leave to user to specify shots
            counts = list((np.array(probs) * 1_000_000).round().astype(int))
            return bases, counts
        else:
            return bases, probs
    else:
        # sample counts
        counts = np.random.multinomial(shots, probs)
        if kwargs.get('counts', False):
            return bases, list(counts)
        else:
            return bases, list(counts / float(shots))

def ProbPlot(qc, c, shots=None, **kwargs):
    """
    Plot histogram of measurement probabilities/counts for classical register `c`.
    """
    bases, vals = Measure(qc, c, shots=shots, counts=False)
    plt.figure(figsize=(max(5, len(bases) * 0.6), 3))
    plt.bar(bases, vals)
    plt.xlabel('Measurement outcome (classical register)')
    plt.ylabel('Probability')
    plt.show()
    return bases, vals
