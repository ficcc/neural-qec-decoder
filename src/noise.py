"""Per-qubit / per-gate circuit-level noise insertion.

Mirrors the placement of Stim's built-in generated noise
(after_clifford_depolarization, after_reset_flip_probability,
before_measure_flip_probability, before_round_data_depolarization),
but every qubit and every CX pair can carry its own rate.
"""
from dataclasses import dataclass, field

import stim

ONE_QUBIT_CLIFFORDS = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "X", "Y", "Z"}
RESETS = {"R", "RZ"}


@dataclass(frozen=True)
class NoiseModel:
    p_1q: dict[int, float]                # after single-qubit Cliffords
    p_2q: dict[tuple[int, int], float]    # after CX, key = sorted pair
    p_reset: dict[int, float]             # X_ERROR after resets
    p_meas: dict[int, float]              # X_ERROR before measurements
    p_data: dict[int, float]              # DEPOLARIZE1 on data qubits at round start
    p_idle: dict[int, float] = field(default_factory=dict)      # idle in a gate moment
    p_res_idle: dict[int, float] = field(default_factory=dict)  # idle in a measure/reset moment


def _iter_ops(circuit: stim.Circuit):
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            yield from _iter_ops(op.body_copy())
        else:
            yield op


def _qubits(op: stim.CircuitInstruction) -> list[int]:
    return [t.value for t in op.targets_copy()]


def cx_pairs(circuit: stim.Circuit) -> list[tuple[int, int]]:
    """Distinct sorted CX qubit pairs, in first-appearance order."""
    seen: dict[tuple[int, int], None] = {}
    for op in _iter_ops(circuit):
        if op.name == "CX":
            q = _qubits(op)
            for a, b in zip(q[::2], q[1::2]):
                seen[tuple(sorted((a, b)))] = None
    return list(seen)


def uniform_model(circuit: stim.Circuit, p: float) -> NoiseModel:
    """Every rate equal to p, keyed by the qubits/pairs the circuit uses."""
    one_q, reset, meas, data = {}, {}, {}, {}
    for op in _iter_ops(circuit):
        q = _qubits(op)
        if op.name in ONE_QUBIT_CLIFFORDS:
            one_q.update(dict.fromkeys(q, p))
        elif op.name in RESETS:
            reset.update(dict.fromkeys(q, p))
        elif op.name == "MR":
            reset.update(dict.fromkeys(q, p))
            meas.update(dict.fromkeys(q, p))
        elif op.name in ("M", "MZ"):
            meas.update(dict.fromkeys(q, p))
            data.update(dict.fromkeys(q, p))
    return NoiseModel(
        p_1q=dict(sorted(one_q.items())),
        p_2q={pair: p for pair in sorted(cx_pairs(circuit))},
        p_reset=dict(sorted(reset.items())),
        p_meas=dict(sorted(meas.items())),
        p_data=dict(sorted(data.items())))


def si1000_model(circuit: stim.Circuit, p: float) -> NoiseModel:
    """SI1000 (Gidney et al. 2021): CX p, 1q p/10, reset 2p, measure 5p,
    idle p/10, idle during measurement/reset moments 2p."""
    base = uniform_model(circuit, p)
    used = sorted({t.value for op in _iter_ops(circuit) for t in op.targets_copy()
                   if t.is_qubit_target})
    return NoiseModel(
        p_1q={q: p / 10 for q in base.p_1q},
        p_2q=dict(base.p_2q),
        p_reset={q: 2 * p for q in base.p_reset},
        p_meas={q: 5 * p for q in base.p_meas},
        p_data={q: 0.0 for q in base.p_data},
        p_idle={q: p / 10 for q in used},
        p_res_idle={q: 2 * p for q in used})


def _grouped(name: str, targets: list[int], rate_of) -> stim.Circuit:
    """One noise instruction per distinct nonzero rate, targets in original order."""
    out = stim.Circuit()
    groups: dict[float, list[int]] = {}
    for t in targets:
        groups.setdefault(rate_of(t), []).append(t)
    for rate, ts in groups.items():
        if rate > 0:
            out.append(name, ts, rate)
    return out


def _grouped_pairs(targets: list[int], p_2q) -> stim.Circuit:
    out = stim.Circuit()
    groups: dict[float, list[int]] = {}
    for a, b in zip(targets[::2], targets[1::2]):
        groups.setdefault(p_2q[tuple(sorted((a, b)))], []).extend((a, b))
    for rate, ts in groups.items():
        if rate > 0:
            out.append("DEPOLARIZE2", ts, rate)
    return out


def apply_noise(circuit: stim.Circuit, model: NoiseModel) -> stim.Circuit:
    """Return a noisy copy of `circuit` with the rates in `model`."""
    data_qubits = list(model.p_data)
    state = {"last_gate": None, "moment": set(), "moment_mr": False}
    all_qubits = list(model.p_idle)
    NOISE_FREE = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}

    def close_moment() -> stim.Circuit:
        idle = [q for q in all_qubits if q not in state["moment"]]
        rates = model.p_res_idle if state["moment_mr"] else model.p_idle
        state["moment"], state["moment_mr"] = set(), False
        return _grouped("DEPOLARIZE1", idle, rates.__getitem__) if idle else stim.Circuit()

    def walk(block: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        for op in block:
            if isinstance(op, stim.CircuitRepeatBlock):
                out.append(stim.CircuitRepeatBlock(op.repeat_count, walk(op.body_copy())))
                continue
            name, q = op.name, _qubits(op)
            if name == "TICK":
                if all_qubits:
                    out += close_moment()
                out.append(op)
                # Round start: the TICK right after a reset or measure-reset layer.
                if state["last_gate"] in RESETS | {"MR"}:
                    out += _grouped("DEPOLARIZE1", data_qubits, model.p_data.__getitem__)
                continue
            if name not in NOISE_FREE:
                state["moment"].update(q)
                if name in RESETS | {"M", "MZ", "MR"}:
                    state["moment_mr"] = True
            if name in ("M", "MZ", "MR"):
                out += _grouped("X_ERROR", q, model.p_meas.__getitem__)
            out.append(op)
            if name in RESETS or name == "MR":
                out += _grouped("X_ERROR", q, model.p_reset.__getitem__)
            elif name in ONE_QUBIT_CLIFFORDS:
                out += _grouped("DEPOLARIZE1", q, model.p_1q.__getitem__)
            elif name == "CX":
                out += _grouped_pairs(q, model.p_2q)
            if name not in NOISE_FREE:
                state["last_gate"] = name
        return out

    return walk(circuit)
