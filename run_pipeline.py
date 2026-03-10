"""
run_pipeline.py
===============
End-to-end integration: AlphaGenome RNA-seq → mRNA init table → Surrogate phenotype predictions.

Demonstrates the full bridge between AlphaGenome sequence predictions and the
Syn3A 4D whole-cell model, including:

  1. Querying AlphaGenome for RNA-seq tracks over all Syn3A genes
  2. Converting to calibrated mRNA molecule counts per cell
  3. Deriving VariantEffectRecords from the abundance table
  4. Training (or loading) the surrogate model
  5. Predicting whole-cell phenotypes for all expression-shifted genes
  6. Writing a combined markdown report

Usage
-----
# Simulation mode (no external dependencies):
    python run_pipeline.py --n-train 5000 --out-dir outputs

# Real AlphaGenome predictions:
    python run_pipeline.py --api-key YOUR_KEY --n-train 5000 --out-dir outputs

# Real simulation sweep data for surrogate:
    python run_pipeline.py --sweep-data path/to/sweep.csv --out-dir outputs

# Both real sources:
    python run_pipeline.py --api-key YOUR_KEY --sweep-data sweep.csv --out-dir outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome_to_mrna import (
    AlphaGenomeToMRNAPipeline,
    MRNAAbundanceTable,
    SYN3A_REFERENCE_MRNA_COUNTS,
    SYN3A_TOTAL_MRNA_MOLECULES,
)
from surrogate_wholecell import (
    VariantToPhenotypeOracle,
    VariantEffectRecord,
    WT_PHENOTYPES,
    PHENOTYPE_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 3 helper: derive expression-shift records from abundance table
# ---------------------------------------------------------------------------

def mrna_table_to_variant_records(
    table: MRNAAbundanceTable,
    min_delta_log2: float = 0.1,
) -> list[VariantEffectRecord]:
    """
    Convert an mRNA abundance table into VariantEffectRecords by comparing
    each gene's predicted molecules/cell against the Syn3A reference counts.

    Only genes whose predicted abundance deviates by more than min_delta_log2
    (in log2 space) from the reference are included.

    Parameters
    ----------
    table : MRNAAbundanceTable
        Output of AlphaGenomeToMRNAPipeline.run().
    min_delta_log2 : float
        Minimum absolute delta-log2 deviation required to include a gene.

    Returns
    -------
    list[VariantEffectRecord]
    """
    ref_map: dict[str, float] = dict(SYN3A_REFERENCE_MRNA_COUNTS)

    records = []
    for rec in table.records:
        ref_val = ref_map.get(rec.name)
        if ref_val is None or ref_val <= 0:
            continue
        delta_log2 = np.log2(max(rec.molecules_per_cell, 1e-6) / ref_val)
        if abs(delta_log2) < min_delta_log2:
            continue

        records.append(VariantEffectRecord(
            variant_id=f"expr_shift_{rec.gene_id}",
            chromosome="chrSyn3A",
            position=0,
            ref="N",
            alt="N",
            delta_log2_expr={rec.name: delta_log2},
            affected_function=rec.function,
            tss_distance_bp=0,
            variant_type="SNP",
        ))

    log.info(
        "Derived %d expression-shift records (|delta_log2| > %.2f) from abundance table",
        len(records), min_delta_log2,
    )
    return records


# ---------------------------------------------------------------------------
# Step 4 helper: generate markdown report
# ---------------------------------------------------------------------------

def generate_report(
    mrna_table: MRNAAbundanceTable,
    predictions_df: pd.DataFrame,
    out_dir: Path,
    simulate: bool,
) -> None:
    """Write a human-readable markdown report summarising pipeline results."""

    lines = [
        "# AlphaGenome x Syn3A Whole-Cell Bridge - Pipeline Report",
        "",
        f"> **Simulation mode**: {'yes - synthetic data, not real predictions' if simulate else 'no - real AlphaGenome predictions'}",
        "",
        "---",
        "",
        "## 1. mRNA Abundance Table (AlphaGenome -> Syn3A init params)",
        "",
        f"- Genes quantified: **{len(mrna_table.records)}**",
        f"- Total predicted mRNA molecules/cell: **{sum(r.molecules_per_cell for r in mrna_table.records):.1f}**",
        f"- Syn3A reference total (Breuer et al. 2019): **{SYN3A_TOTAL_MRNA_MOLECULES:.1f}**",
        "",
        "### Top 10 expressed genes",
        "",
        "| Gene | Function | TPM | Molecules/cell |",
        "|------|----------|----:|---------------:|",
    ]

    for r in sorted(mrna_table.records, key=lambda x: x.molecules_per_cell, reverse=True)[:10]:
        lines.append(f"| {r.name} | {r.function} | {r.tpm:.0f} | {r.molecules_per_cell:.2f} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Whole-Cell Phenotype Predictions (Surrogate Model)",
        "",
    ]

    if len(predictions_df) == 0:
        lines.append(
            "_No expression shifts exceeded the minimum delta-log2 threshold -- "
            "no phenotype predictions generated._"
        )
    else:
        lethal = int((predictions_df["viability_score"] < 0.5).sum())
        lines += [
            f"- Variants / expression shifts analysed: **{len(predictions_df)}**",
            f"- Predicted likely lethal (viability < 0.5): **{lethal}**",
            f"- Mean viability score: **{predictions_df['viability_score'].mean():.3f}**",
            "",
            "### Phenotype summary (mean +/- std across all perturbations)",
            "",
            "| Phenotype | Predicted mean +/- std | Wild-type |",
            "|-----------|------------------------|----------:|",
        ]
        for col in PHENOTYPE_NAMES:
            if col in predictions_df.columns:
                wt = WT_PHENOTYPES[col]
                m  = predictions_df[col].mean()
                s  = predictions_df[col].std()
                lines.append(f"| {col} | {m:.3f} +/- {s:.3f} | {wt:.3f} |")

        lines += [
            "",
            "### Most deleterious predicted variants (lowest viability)",
            "",
            "| Variant ID | Viability | Division time (min) | Replication success |",
            "|------------|----------:|--------------------:|--------------------:|",
        ]
        for _, row in predictions_df.sort_values("viability_score").head(5).iterrows():
            lines.append(
                f"| {row.get('variant_id','?')} | {row['viability_score']:.3f} | "
                f"{row.get('division_time_min', float('nan')):.1f} | "
                f"{row.get('dna_replication_success', float('nan')):.3f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Methods",
        "",
        "**AlphaGenome API wrapper** (alphagenome_to_mrna.py):",
        "Queries AlphaGenome RNA-seq track predictions over each Syn3A gene interval.",
        "Per-nucleotide coverage is aggregated to reads-per-kilobase (RPK), then",
        "TPM-normalised. Absolute molecule counts are calibrated via OLS regression",
        "against reference mRNA counts from Breuer et al. (2019) eLife.",
        "",
        "**Surrogate emulator** (surrogate_wholecell.py):",
        "A 5-member ensemble of 3-layer MLPs (256->128->64) trained on Syn3A simulation",
        "data (synthetic physics-based or real sweep outputs via SweepDataLoader).",
        "Predicts 5 phenotypic outputs. Uncertainty estimated from ensemble disagreement.",
        "",
        "## References",
        "",
        "- Avsec et al. (2026) AlphaGenome. Nature 649. doi:10.1038/s41586-025-10014-0",
        "- Luthey-Schulten et al. (2026) Syn3A 4D whole-cell model. Cell. S0092-8674(26)00174-1",
        "- Breuer et al. (2019) Essential metabolism for a minimal cell. eLife 8:e36842.",
        "- Hutchison et al. (2016) Design and synthesis of a minimal bacterial genome. Science 351.",
    ]

    report_path = out_dir / "pipeline_report.md"
    report_path.write_text("\n".join(lines))
    log.info("Pipeline report written -> %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AlphaGenome -> Syn3A surrogate whole-cell pipeline"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="AlphaGenome API key. If omitted, module 1 runs in simulation mode.",
    )
    parser.add_argument(
        "--sweep-data", default=None, metavar="CSV",
        help=(
            "Path to a real Syn3A simulation sweep CSV for surrogate training. "
            "If omitted, surrogate trains on synthetic data. "
            "Run `python surrogate_wholecell.py --write-example-csv` to see the expected format."
        ),
    )
    parser.add_argument(
        "--n-train", type=int, default=5000,
        help="Synthetic training samples (used only when --sweep-data is not provided).",
    )
    parser.add_argument(
        "--model-path", default=None,
        help=(
            "Path to cache the trained surrogate model (.pkl). "
            "If the file already exists the model is loaded instead of re-trained. "
            "Defaults to <out-dir>/syn3a_surrogate.pkl."
        ),
    )
    parser.add_argument("--out-dir", default="outputs", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or str(out_dir / "syn3a_surrogate.pkl")
    simulate_mrna = args.api_key is None

    # Step 1: AlphaGenome -> mRNA abundance table
    log.info("=== STEP 1: AlphaGenome RNA-seq -> mRNA abundance table ===")
    mrna_pipeline = AlphaGenomeToMRNAPipeline(
        api_key=args.api_key,
        simulate=simulate_mrna,
    )
    mrna_table = mrna_pipeline.run()
    mrna_table.to_csv(out_dir / "syn3a_mrna_abundance.csv")
    mrna_table.to_wholecell_json(out_dir / "syn3a_mrna_init_params.json")

    # Step 2: Train / load surrogate model
    log.info("=== STEP 2: Train / load Syn3A surrogate model ===")
    oracle = VariantToPhenotypeOracle.build_and_train(
        n_train=args.n_train,
        data_path=args.sweep_data,
        model_path=model_path,
    )

    # Step 3: Derive expression-shift records and predict
    log.info("=== STEP 3: Expression shifts -> whole-cell phenotype predictions ===")
    variant_records = mrna_table_to_variant_records(mrna_table)

    if variant_records:
        preds = oracle.surrogate.predict(variant_records)
        preds_df = pd.DataFrame([p.to_dict() for p in preds])
        preds_df.to_csv(out_dir / "phenotype_predictions.csv", index=False)
        log.info("Phenotype predictions -> %s", out_dir / "phenotype_predictions.csv")
    else:
        preds_df = pd.DataFrame()
        log.info("No genes exceeded the delta-log2 threshold -- no phenotype predictions written.")

    # Step 4: Report
    log.info("=== STEP 4: Writing pipeline report ===")
    generate_report(mrna_table, preds_df, out_dir, simulate=simulate_mrna)

    print("\n" + "=" * 65)
    print(f"  Pipeline complete. Outputs in: {out_dir}/")
    print(f"  |- syn3a_mrna_abundance.csv       mRNA init table")
    print(f"  |- syn3a_mrna_init_params.json    whole-cell model params")
    print(f"  |- syn3a_surrogate.pkl            trained surrogate model")
    print(f"  |- phenotype_predictions.csv      variant phenotype predictions")
    print(f"  +- pipeline_report.md             human-readable report")
    print("=" * 65)


if __name__ == "__main__":
    main()
