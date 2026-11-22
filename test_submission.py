import copy

from submission import (
    circuit_cost,
    equivalent_up_to_global_phase,
    evaluate_optimization,
    optimize,
)


CONFIG = {"w1": 1.0, "w2": 6.0, "wd": 0.2}


def check(name, circuit, config=CONFIG):
    original = copy.deepcopy(circuit)
    result = optimize(circuit, config)
    assessment = evaluate_optimization(original, result, config)
    assert circuit == original, name + " mutated input"
    assert assessment["equivalent"], name + " is not equivalent"
    assert assessment["cost_reduction"] >= -1e-9, name + " increased cost"
    print(name, assessment)


if __name__ == "__main__":
    check("CX-Z-CX", {
        "n_qubits": 2,
        "gates": [
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "Z", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
    })
    check("Pauli inverses", {
        "n_qubits": 3,
        "gates": [
            {"gate": "H", "qubits": [0]},
            {"gate": "H", "qubits": [0]},
            {"gate": "X", "qubits": [1]},
            {"gate": "X", "qubits": [1]},
            {"gate": "S", "qubits": [2]},
            {"gate": "SDG", "qubits": [2]},
        ],
    })
    check("phase combination", {
        "n_qubits": 1,
        "gates": [
            {"gate": "T", "qubits": [0]},
            {"gate": "T", "qubits": [0]},
            {"gate": "T", "qubits": [0]},
            {"gate": "T", "qubits": [0]},
        ],
    })
    check("adjacent CX", {
        "n_qubits": 2,
        "gates": [
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
    })
