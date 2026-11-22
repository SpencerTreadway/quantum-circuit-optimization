"""Qublitz 2026 quantum circuit optimizer."""

import copy
import math

import numpy as np


SINGLE_QUBIT_GATES = frozenset(("H", "X", "Z", "S", "SDG", "T", "TDG"))
ALLOWED_GATES = SINGLE_QUBIT_GATES | {"CX"}


def optimize(circuit, config):
    """Return a cost-aware, equivalent circuit without modifying the inputs."""
    validate_circuit(circuit)
    validate_config(config)

    gates = copy.deepcopy(circuit["gates"])
    n = circuit["n_qubits"]
    best = {"n_qubits": n, "gates": gates}
    best_cost = circuit_cost(best, config)

    # Repeated passes expose more reductions and let transformations interact.
    for _ in range(12):
        changed = False
        pre_bubble = copy.deepcopy(gates)
        _bubble_commuting_gates(gates)
        pre_bubble_cost = circuit_cost(
            {"n_qubits": n, "gates": pre_bubble}, config)
        bubble_cost = circuit_cost(
            {"n_qubits": n, "gates": gates}, config)
        if bubble_cost > pre_bubble_cost + 1e-12:
            gates[:] = pre_bubble

        candidate = {"n_qubits": n, "gates": gates}
        cost = circuit_cost(candidate, config)
        if cost < best_cost - 1e-12:
            best = copy.deepcopy(candidate)
            best_cost = cost
            changed = True

        # First simplify short one-qubit runs.  This catches phase products,
        # inverses, and general Clifford+T products.
        for i in range(len(gates)):
            for j in range(i + 1, min(len(gates) + 1, i + 8)):
                if len({q for op in gates[i:j] for q in op["qubits"]}) != 1:
                    break
                target = _one_qubit_matrix(gates[i:j])
                replacement = _synthesize_one_qubit(target)
                if replacement is None:
                    continue
                target_qubit = gates[i]["qubits"][0]
                replacement = [
                    {"gate": operation["gate"], "qubits": [target_qubit]}
                    for operation in replacement
                ]
                trial_gates = gates[:i] + replacement + gates[j:]
                trial = {"n_qubits": n, "gates": trial_gates}
                trial_cost = circuit_cost(trial, config)
                if trial_cost < circuit_cost({"n_qubits": n, "gates": gates}, config) - 1e-12:
                    gates[:] = trial_gates
                    changed = True
                    break
            if changed:
                break

        if not changed:
            # Search contiguous small two-qubit regions.  Unlike pairwise
            # rewrites, this also recognizes products of several local gates.
            for i in range(len(gates)):
                for j in range(i + 2, min(len(gates) + 1, i + 7)):
                    involved = {q for op in gates[i:j] for q in op["qubits"]}
                    if len(involved) > 2:
                        break
                    if len(involved) == 2:
                        pair = sorted(involved)
                        target = _window_two_qubit_matrix(gates[i:j], pair)
                        replacement = _synthesize_two_qubit(target, pair, config)
                        if replacement is None:
                            continue
                        trial_gates = gates[:i] + replacement + gates[j:]
                        trial = {"n_qubits": n, "gates": trial_gates}
                        trial_cost = circuit_cost(trial, config)
                        if trial_cost < circuit_cost({"n_qubits": n, "gates": gates}, config) - 1e-12:
                            gates[:] = trial_gates
                            changed = True
                            break
                if changed:
                    break

        candidate = {"n_qubits": n, "gates": gates}
        cost = circuit_cost(candidate, config)
        if cost < best_cost - 1e-12:
            best = copy.deepcopy(candidate)
            best_cost = cost

        if not changed:
            break

    return best


def validate_circuit(circuit):
    if not isinstance(circuit, dict) or set(circuit) != {"n_qubits", "gates"}:
        raise ValueError("Circuit must have exactly 'n_qubits' and 'gates'.")
    n = circuit["n_qubits"]
    if type(n) is not int or not 1 <= n <= 6:
        raise ValueError("n_qubits must be an integer from 1 through 6.")
    if not isinstance(circuit["gates"], list):
        raise ValueError("gates must be a list.")
    for i, operation in enumerate(circuit["gates"]):
        if not isinstance(operation, dict) or set(operation) != {"gate", "qubits"}:
            raise ValueError(f"Gate {i}: expected exactly 'gate' and 'qubits'.")
        name = operation["gate"]
        qubits = operation["qubits"]
        if name not in ALLOWED_GATES:
            raise ValueError(f"Gate {i}: unsupported gate {name!r}.")
        if not isinstance(qubits, list):
            raise ValueError(f"Gate {i}: qubits must be a list.")
        arity = 2 if name == "CX" else 1
        if len(qubits) != arity:
            raise ValueError(f"Gate {i}: {name} requires {arity} qubit(s).")
        if any(type(q) is not int or not 0 <= q < n for q in qubits):
            raise ValueError(f"Gate {i}: invalid qubit index.")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Gate {i}: qubit indices must be distinct.")
    return True


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"w1", "w2", "wd"}:
        raise ValueError("config must have exactly 'w1', 'w2', and 'wd'.")
    for key in ("w1", "w2", "wd"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a real number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if not any(config[key] > 0 for key in ("w1", "w2", "wd")):
        raise ValueError("At least one weight must be positive.")
    return True


_ROOT2 = math.sqrt(2)
_ONE_QUBIT_MATRICES = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / _ROOT2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "S": np.diag([1, 1j]).astype(complex),
    "SDG": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex),
    "TDG": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
}


def _full_gate_matrix(n, operation):
    name = operation["gate"]
    qubits = operation["qubits"]
    if name != "CX":
        result = np.array([[1]], dtype=complex)
        for q in range(n):
            factor = (_ONE_QUBIT_MATRICES[name]
                      if q == qubits[0] else np.eye(2, dtype=complex))
            result = np.kron(result, factor)
        return result
    control, target = qubits
    dimension = 1 << n
    result = np.zeros((dimension, dimension), dtype=complex)
    control_bit = 1 << (n - 1 - control)
    target_bit = 1 << (n - 1 - target)
    for input_state in range(dimension):
        output_state = (input_state ^ target_bit
                        if input_state & control_bit else input_state)
        result[output_state, input_state] = 1
    return result


def circuit_unitary(circuit):
    validate_circuit(circuit)
    n = circuit["n_qubits"]
    result = np.eye(1 << n, dtype=complex)
    for operation in circuit["gates"]:
        result = _full_gate_matrix(n, operation) @ result
    return result


def equivalent_up_to_global_phase(first, second, atol=1e-9):
    validate_circuit(first)
    validate_circuit(second)
    if first["n_qubits"] != second["n_qubits"]:
        return False
    a = circuit_unitary(first)
    b = circuit_unitary(second)
    pivot = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[pivot] / a[pivot]
    return (np.isclose(abs(phase), 1.0, atol=atol, rtol=0.0)
            and np.allclose(b, phase * a, atol=atol, rtol=0.0))


def circuit_metrics(circuit):
    validate_circuit(circuit)
    available = [0] * circuit["n_qubits"]
    one_qubit = 0
    two_qubit = 0
    for operation in circuit["gates"]:
        qubits = operation["qubits"]
        end = 1 + max(available[q] for q in qubits)
        for q in qubits:
            available[q] = end
        if operation["gate"] == "CX":
            two_qubit += 1
        else:
            one_qubit += 1
    return {"n_1q": one_qubit, "n_2q": two_qubit,
            "depth": max(available)}


def circuit_cost(circuit, config):
    validate_config(config)
    metrics = circuit_metrics(circuit)
    return (config["w1"] * metrics["n_1q"]
            + config["w2"] * metrics["n_2q"]
            + config["wd"] * metrics["depth"])


def evaluate_optimization(original, optimized, config):
    equivalent = equivalent_up_to_global_phase(original, optimized)
    before = circuit_cost(original, config)
    after = circuit_cost(optimized, config)
    return {
        "equivalent": equivalent,
        "input_metrics": circuit_metrics(original),
        "output_metrics": circuit_metrics(optimized),
        "input_cost": before,
        "output_cost": after,
        "cost_reduction": before - after,
    }


def _one_qubit_matrix(gates):
    matrix = np.eye(2, dtype=complex)
    for operation in gates:
        matrix = _ONE_QUBIT_MATRICES[operation["gate"]] @ matrix
    return matrix


def _equivalent_matrices(first, second, atol=1e-8):
    pivot = np.unravel_index(np.argmax(np.abs(first)), first.shape)
    if abs(first[pivot]) < 1e-12:
        return False
    phase = second[pivot] / first[pivot]
    return (np.isclose(abs(phase), 1.0, atol=atol, rtol=0.0)
            and np.allclose(second, phase * first, atol=atol, rtol=0.0))


_ONE_SYNTH_CACHE = {}


def _synthesize_one_qubit(target):
    target_key = tuple(np.round(target.reshape(-1), 9))
    if _equivalent_matrices(np.eye(2, dtype=complex), target):
        _ONE_SYNTH_CACHE[target_key] = []
        return []

    if target_key in _ONE_SYNTH_CACHE:
        return _ONE_SYNTH_CACHE[target_key]

    names = ("H", "X", "Z", "S", "SDG", "T", "TDG")
    identity = np.eye(2, dtype=complex)
    states = [(_matrix_phase_key(identity), identity, tuple())]
    by_key = {_matrix_phase_key(identity): (identity, tuple())}

    target = target
    for length in range(1, 7):
        next_states = []
        for _, matrix, sequence in states:
            for name in names:
                operation = {"gate": name, "qubits": [0]}
                next_matrix = _ONE_QUBIT_MATRICES[name] @ matrix
                if _equivalent_matrices(target, next_matrix):
                    result = [{"gate": name, "qubits": [0]} for name in
                              tuple(sequence) + (name,)]
                    _ONE_SYNTH_CACHE[target_key] = result
                    return result
                next_key = _matrix_phase_key(next_matrix)
                if next_key not in by_key:
                    by_key[next_key] = (next_matrix,
                                        tuple(sequence) + (name,))
                    next_states.append((next_key, next_matrix,
                                        tuple(sequence) + (name,)))
        states = next_states
    _ONE_SYNTH_CACHE[target_key] = None
    return None


def _matrix_phase_key(matrix):
    pivot = np.unravel_index(np.argmax(np.abs(matrix)), matrix.shape)
    value = matrix[pivot]
    if abs(value) > 1e-12:
        matrix = matrix / value
    return tuple(np.round(matrix.reshape(-1), 9))


def _gate_matrix_2q(operation):
    name = operation["gate"]
    qubits = operation["qubits"]
    if name != "CX":
        result = np.array([[1]], dtype=complex)
        for q in (0, 1):
            factor = (_ONE_QUBIT_MATRICES[name]
                      if q == qubits[0] else np.eye(2, dtype=complex))
            result = np.kron(result, factor)
        return result
    control, target = qubits
    result = np.zeros((4, 4), dtype=complex)
    control_bit = 1 << (1 - control)
    target_bit = 1 << (1 - target)
    for input_state in range(4):
        output = (input_state ^ target_bit
                  if input_state & control_bit else input_state)
        result[output, input_state] = 1
    return result


def _window_two_qubit_matrix(gates, pair):
    matrix = np.eye(4, dtype=complex)
    qmap = {pair[0]: 0, pair[1]: 1}
    for operation in gates:
        name = operation["gate"]
        normalized = {"gate": name,
                      "qubits": [qmap[q] for q in operation["qubits"]]}
        matrix = _gate_matrix_2q(normalized) @ matrix
    return matrix


_TWO_QUBIT_LUT = None


def _build_two_qubit_lut():
    candidates = {}
    names = ("H", "X", "Z", "S", "SDG", "T", "TDG")
    operations = []
    for name in names:
        operations.append({"gate": name, "qubits": [0]})
        operations.append({"gate": name, "qubits": [1]})
    operations.append({"gate": "CX", "qubits": [0, 1]})
    operations.append({"gate": "CX", "qubits": [1, 0]})

    identity = np.eye(4, dtype=complex)
    states = {_matrix_phase_key(identity): (identity, tuple())}
    for length in range(1, 4):
        next_states = {}
        for key, (matrix, sequence) in states.items():
            for operation in operations:
                next_sequence = tuple(sequence) + (
                    (operation["gate"], tuple(operation["qubits"])),)
                next_matrix = _gate_matrix_2q(operation) @ matrix
                next_key = _matrix_phase_key(next_matrix)
                candidates.setdefault(next_key, next_sequence)
                next_states.setdefault(next_key, (next_matrix, next_sequence))
        states = next_states
    return candidates


def _local_two_qubit_cost(sequence, pair, config):
    cost = 0.0
    available = [0, 0]
    for name, normalized_qubits in sequence:
        qubits = [pair[q] for q in normalized_qubits]
        if name == "CX":
            cost += config["w2"]
        else:
            cost += config["w1"]
        end = 1 + max(available[q] for q in range(2)
                       if normalized_qubits and normalized_qubits.count(q) == 1)
        for q in normalized_qubits:
            available[q] = end
    if sequence:
        cost += config["wd"] * max(available)
    return cost


def _synthesize_two_qubit(target, pair, config):
    global _TWO_QUBIT_LUT
    if _TWO_QUBIT_LUT is None:
        _TWO_QUBIT_LUT = _build_two_qubit_lut()
    if _equivalent_matrices(np.eye(4, dtype=complex), target):
        return []
    key = _matrix_phase_key(target)
    sequence = _TWO_QUBIT_LUT.get(key)
    if sequence is None:
        return None

    # Some phase-equivalent unitary keys can differ only by the pivot.
    # Resolve the exact match before returning a sequence.
    target_matrix = target
    operations_for_sequence = []
    for name, normalized_qubits in sequence:
        operations_for_sequence.append({
            "gate": name,
            "qubits": [pair[q] for q in normalized_qubits],
        })
    matrix = np.eye(4, dtype=complex)
    qmap = {pair[0]: 0, pair[1]: 1}
    for operation in operations_for_sequence:
        normalized = {"gate": operation["gate"],
                      "qubits": [qmap[q] for q in operation["qubits"]]}
        matrix = _gate_matrix_2q(normalized) @ matrix
    if not _equivalent_matrices(target_matrix, matrix):
        return None
    return operations_for_sequence


def _commute(first, second):
    if set(first["qubits"]).isdisjoint(second["qubits"]):
        return True
    if first["gate"] == "CX":
        if second["gate"] in ("Z", "S", "SDG", "T", "TDG"):
            return second["qubits"][0] == first["qubits"][0]
        if second["gate"] == "X":
            return second["qubits"][0] == first["qubits"][1]
    if second["gate"] == "CX":
        if first["gate"] in ("Z", "S", "SDG", "T", "TDG"):
            return first["qubits"][0] == second["qubits"][0]
        if first["gate"] == "X":
            return first["qubits"][0] == second["qubits"][1]
    return False


def _bubble_commuting_gates(gates):
    for i in range(1, len(gates)):
        j = i
        while j > 0 and _commute(gates[j - 1], gates[j]):
            gates[j - 1], gates[j] = gates[j], gates[j - 1]
            j -= 1


if __name__ == "__main__":
    example = {
        "n_qubits": 2,
        "gates": [
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "Z", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
    }
    weights = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
    result = optimize(example, weights)
    print("Optimized:", result)
    print("Assessment:", evaluate_optimization(example, result, weights))

