from __future__ import annotations


VALID_Q_ANSATZ_TYPES = {
    "rot_cnot_ring",
    "rot_cnot_chain",
    "rot_cz_ring",
    "rot_none",
}


def resolve_q_ansatz_type(q_ansatz_type: str = "rot_cnot_ring") -> str:
    q_ansatz_type = str(q_ansatz_type).strip().lower()
    if q_ansatz_type not in VALID_Q_ANSATZ_TYPES:
        raise ValueError(
            f"Unknown q_ansatz_type={q_ansatz_type!r}. "
            f"Choose from {sorted(VALID_Q_ANSATZ_TYPES)}."
        )
    return q_ansatz_type


def get_q_ansatz_weight_shapes(
    q_ansatz_type: str,
    *,
    n_q_layers: int,
    n_qubits: int,
) -> dict[str, tuple[int, ...]]:
    q_ansatz_type = resolve_q_ansatz_type(q_ansatz_type)

    if q_ansatz_type in VALID_Q_ANSATZ_TYPES:
        return {"weights": (n_q_layers, n_qubits, 3)}

    raise ValueError(f"Unknown q_ansatz_type={q_ansatz_type!r}")


def _apply_xyz_rotations(qml, layer_weights, n_qubits: int) -> None:
    for i in range(n_qubits):
        qml.RX(layer_weights[i, 0], wires=i)
        qml.RY(layer_weights[i, 1], wires=i)
        qml.RZ(layer_weights[i, 2], wires=i)


def _apply_cnot_ring(qml, n_qubits: int) -> None:
    if n_qubits <= 1:
        return

    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def _apply_cnot_chain(qml, n_qubits: int) -> None:
    if n_qubits <= 1:
        return

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def _apply_cz_ring(qml, n_qubits: int) -> None:
    if n_qubits <= 1:
        return

    for i in range(n_qubits):
        qml.CZ(wires=[i, (i + 1) % n_qubits])


def apply_q_ansatz(
    qml,
    *,
    q_ansatz_type: str,
    weights,
    n_qubits: int,
    n_q_layers: int,
) -> None:
    q_ansatz_type = resolve_q_ansatz_type(q_ansatz_type)

    for layer in range(n_q_layers):
        layer_weights = weights[layer]
        _apply_xyz_rotations(qml, layer_weights, n_qubits)

        if q_ansatz_type == "rot_cnot_ring":
            _apply_cnot_ring(qml, n_qubits)
        elif q_ansatz_type == "rot_cnot_chain":
            _apply_cnot_chain(qml, n_qubits)
        elif q_ansatz_type == "rot_cz_ring":
            _apply_cz_ring(qml, n_qubits)
        elif q_ansatz_type == "rot_none":
            continue
        else:
            raise ValueError(f"Unknown q_ansatz_type={q_ansatz_type!r}")
