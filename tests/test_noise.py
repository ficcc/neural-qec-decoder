import stim

from src.circuits import noiseless_circuit
from src.noise import apply_noise, uniform_model


def test_uniform_noise_matches_stim_builtin():
    d, p = 3, 0.004
    c = noiseless_circuit(d, d)
    ours = apply_noise(c, uniform_model(c, p))
    ref = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)
    assert ours.detector_error_model(decompose_errors=True).approx_equals(
        ref.detector_error_model(decompose_errors=True), atol=1e-9)


def test_distinct_rates_match_stim_builtin():
    # Distinct values per noise channel catch placement swaps a single p would hide.
    c = noiseless_circuit(5, 5)
    m = uniform_model(c, 0.0)
    for k in m.p_1q: m.p_1q[k] = 0.001
    for k in m.p_2q: m.p_2q[k] = 0.002
    for k in m.p_reset: m.p_reset[k] = 0.003
    for k in m.p_meas: m.p_meas[k] = 0.004
    for k in m.p_data: m.p_data[k] = 0.005
    ref = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=5, rounds=5,
        after_clifford_depolarization=0.001, after_reset_flip_probability=0.003,
        before_measure_flip_probability=0.004, before_round_data_depolarization=0.005)
    # Stim uses one rate for 1q and 2q Cliffords, so compare with p_2q set equal.
    for k in m.p_2q: m.p_2q[k] = 0.001
    assert apply_noise(c, m).detector_error_model(decompose_errors=True).approx_equals(
        ref.detector_error_model(decompose_errors=True), atol=1e-9)


def test_per_qubit_rate_is_used():
    c = noiseless_circuit(3, 3)
    m = uniform_model(c, 0.001)
    q = sorted(m.p_meas)[0]
    m.p_meas[q] = 0.2
    assert f"X_ERROR(0.2) {q}" in str(apply_noise(c, m))


def test_d5_d7_compile():
    for d in (5, 7):
        c = noiseless_circuit(d, d)
        apply_noise(c, uniform_model(c, 0.003)).compile_detector_sampler().sample(10)


def _depol1_targets(circ, rate):
    out = []
    for op in circ.flattened():
        if op.name == "DEPOLARIZE1" and abs(op.gate_args_copy()[0] - rate) < 1e-12:
            out.append({t.value for t in op.targets_copy()})
    return out


def test_si1000_rates():
    from src.noise import si1000_model
    c = noiseless_circuit(3, 3)
    m = si1000_model(c, 0.002)
    assert set(m.p_2q.values()) == {0.002}
    assert set(m.p_1q.values()) == {0.0002}
    assert set(m.p_reset.values()) == {0.004}
    assert set(m.p_meas.values()) == {0.01}
    assert set(m.p_data.values()) <= {0.0}
    n_qubits = c.num_qubits - 0  # qubit indices may be sparse; compare against used qubits
    used = {t.value for op in c.flattened() for t in op.targets_copy() if t.is_qubit_target}
    assert set(m.p_idle) == used and set(m.p_idle.values()) == {0.0002}
    assert set(m.p_res_idle) == used and set(m.p_res_idle.values()) == {0.004}


def test_idle_noise_on_untouched_qubits():
    from src.noise import si1000_model
    c = noiseless_circuit(3, 3)
    noisy = apply_noise(c, si1000_model(c, 0.002))
    # First H layer acts on the 4 X-ancillas only; the other 13 qubits idle at p/10.
    h_qubits = {t.value for op in c.flattened() if op.name == "H" for t in op.targets_copy()}
    used = {t.value for op in c.flattened() for t in op.targets_copy() if t.is_qubit_target}
    # Stim fuses adjacent same-rate DEPOLARIZE1 (H gate noise + idle), so check coverage.
    first_layer = _depol1_targets(noisy, 0.0002)[0]
    assert first_layer == used


def test_resonator_idle_during_measurement():
    from src.noise import si1000_model
    c = noiseless_circuit(3, 3)
    m = si1000_model(c, 0.002)
    noisy = apply_noise(c, m)
    data = set(m.p_data)
    # Data qubits idle while ancillas are measured-and-reset each round.
    assert data in _depol1_targets(noisy, 0.004)
    zero_idle = si1000_model(c, 0.002)
    for k in zero_idle.p_idle: zero_idle.p_idle[k] = 0.0
    for k in zero_idle.p_res_idle: zero_idle.p_res_idle[k] = 0.0
    assert not noisy.detector_error_model().approx_equals(
        apply_noise(c, zero_idle).detector_error_model(), atol=1e-6)
