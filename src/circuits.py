"""Noiseless rotated surface-code memory circuits."""
import stim


def noiseless_circuit(d: int, rounds: int) -> stim.Circuit:
    """Z-basis rotated surface-code memory experiment with no noise."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds)


def dem_of(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Graph-like detector error model (hyperedges decomposed for matching)."""
    return circuit.detector_error_model(decompose_errors=True)
