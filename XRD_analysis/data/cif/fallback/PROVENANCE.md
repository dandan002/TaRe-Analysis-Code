# TaRe bcc Fallback CIF — Provenance

## Summary

This CIF was generated programmatically because Materials Project (as of 2026-04-10)
returns no Ta-Re binary compound with space group Im-3m (#229) via its public API.

## Structure parameters

| Parameter | Value |
|-----------|-------|
| Formula | TaRe (equiatomic, occupationally disordered) |
| Space group | Im-3m (#229) |
| Lattice parameter | a = 3.19 Å |
| Wyckoff sites | bcc sites at (0,0,0) and (0.5,0.5,0.5), each with Ta0.5/Re0.5 occupancy |
| Source | Generated with pymatgen.core.Structure (cubic Bravais lattice) |

## Rationale

The lattice parameter a ≈ 3.19 Å is consistent with published values for TaRe bcc alloys
(Shen et al. 2014; JCPDS card for elemental Ta: 3.303 Å; elemental Re: 2.761 Å hcp;
bcc TaRe interpolated). This structure is used solely for preliminary peak-position
estimation in Phase 4. Rietveld refinement in Phase 5 will use the actual refined
lattice parameters.

## Downstream usage

- Copied to `data/cif/TaRe_1_fallback.cif` at runtime when MP returns no Im-3m candidate.
- `build_phase_peaks_dict()` reads the first path in `downloaded["TaRe"]` to compute
  Cu Kα 2θ positions for the quick-look overlay.

## Regeneration

```python
from pymatgen.core import Structure, Lattice
lattice = Lattice.cubic(3.19)
mixed = {"Ta": 0.5, "Re": 0.5}
structure = Structure(lattice, [mixed, mixed], [[0, 0, 0], [0.5, 0.5, 0.5]])
structure.to(filename="data/cif/fallback/TaRe_bcc.cif")
```
