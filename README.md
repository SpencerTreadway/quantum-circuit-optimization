# Qublitz 2026 Quantum Circuit Golf

## Run

The implementation requires Python 3 and NumPy:

```bash
python3 -m pip install -r requirements.txt
python3 submission.py
python3 test_submission.py
```

`submission.py` exports `optimize(circuit, config)`.

## Algorithm

The optimizer repeatedly applies:

1. Exact commutation rewrites, including CX with diagonal gates on the control
   and X on the target.
2. Exhaustive synthesis of contiguous one-qubit segments up to six gates.
3. Exhaustive synthesis of contiguous two-qubit segments up to three gates.
4. Full-circuit cost comparisons, so a rewrite is accepted only if its weighted
   cost is lower.

Only matrix-equivalent local rewrites are accepted. One-qubit synthesis searches
the supported Clifford/phase gates; two-qubit synthesis includes all supported
single-qubit gates and CX in both orientations.

## Limitations

The local search does not exhaustively optimize arbitrary larger circuits, and
the two-qubit lookup is bounded to three generated gates. It is therefore
strong on local gate products, phase combinations, adjacent inverses, and
CX-commutation identities, but cannot guarantee a globally optimal result.
