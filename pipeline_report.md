# AlphaGenome x Syn3A Bridge — Pipeline Report

## mRNA Abundance Table
- Genes: 25
- Total molecules/cell: 162.6
- Reference total: 149.0

| Gene | Function | TPM | Molecules/cell |
|------|----------|-----|----------------|
| dnaK | chaperone | 310555 | 50.50 |
| gyrB | dna_topology | 126726 | 20.61 |
| pgk | glycolysis | 120113 | 19.53 |
| atpB | energy_metabolism | 59397 | 9.66 |
| ftsZ | cell_division | 52843 | 8.59 |
| infB | translation_initiation | 48528 | 7.89 |
| ftsQ | cell_division | 37928 | 6.17 |
| ftsA | cell_division | 37802 | 6.15 |
| fusA | translation | 32502 | 5.28 |
| dnaA | replication_initiation | 29993 | 4.88 |

## Whole-Cell Phenotype Predictions

- Perturbations analysed: 9
- Mean viability: 0.867
- Predicted lethal (viability < 0.5): 0

| Phenotype | Mean predicted | Wild-type |
|-----------|---------------|-----------|
| division_time_min | 138.812 | 105.000 |
| growth_rate_per_hr | 0.352 | 0.396 |
| dna_replication_success | 0.974 | 0.980 |
| atp_flux_rel | 0.847 | 1.000 |
| ribosome_occupancy | 0.686 | 0.820 |

## References
- Avsec et al. 2026 AlphaGenome. Nature 649. doi:10.1038/s41586-025-10014-0
- Luthey-Schulten et al. 2026 Syn3A 4D model. Cell. S0092-8674(26)00174-1