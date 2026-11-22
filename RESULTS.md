# Experimentation Results

The local test suite verifies equivalence up to global phase and confirms that
no tested optimization increases the weighted cost.

| Test | Input cost | Output cost | Reduction | Equivalence |
|---|---:|---:|---:|---|
| CX-Z-CX | 13.6 | 1.2 | 12.4 | Passed |
| H/X/S inverse pairs | 6.4 | 0.0 | 6.4 | Passed |
| TTTT phase product | 4.8 | 1.2 | 3.6 | Passed |
| Adjacent CX pair | 12.4 | 0.0 | 12.4 | Passed |

The weight configuration used is `{"w1": 1.0, "w2": 6.0, "wd": 0.2}`.

## Observed strengths

- Removing redundant adjacent inverses and phase products.
- Reducing CX-Z-CX through commutation and CX cancellation.
- Replacing short exact local unitary products with fewer gates.
- Using full-circuit weighted cost to reject unhelpful rewrites.

## Known limitations

- Two-qubit synthesis is bounded to sequences of three supported operations.
- One-qubit synthesis is bounded to six-gate generated sequences.
- Arbitrary large entangling or cross-qubit structures are not exhaustively
  searched.
- The implementation does not guarantee a globally optimal circuit.
