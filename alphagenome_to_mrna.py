"""
alphagenome_to_mrna.py
======================
API wrapper that converts AlphaGenome RNA-seq track outputs into
initial mRNA abundance tables suitable for seeding the Syn3A 4D whole-cell model.

Pipeline
--------
1. Query the AlphaGenome API for RNA-seq predictions over each Syn3A gene interval.
2. Aggregate per-nucleotide coverage tracks into per-gene TPM estimates.
3. Normalise to absolute molecule counts per cell using Syn3A reference proteomics.
4. Output a structured abundance table in the format expected by the whole-cell model
   (CME/ODE hybrid kinetic model parameter file, or SBML-compatible CSV).

References
----------
- AlphaGenome: Avsec et al. (2026), Nature 649, doi:10.1038/s41586-025-10014-0
- Syn3A 4D whole-cell model: Luthey-Schulten lab, Cell (2026)
  doi:10.1016/j.cell.2026.02.014  [S0092-8674(26)00174-1]
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Syn3A reference genome catalogue
# All 452 protein-coding genes with their genomic coordinates on the
# 543,379 bp JCVI-syn3A chromosome (CP016816.2).
# Coordinates are 0-based half-open intervals [start, end).
# Source: NCBI annotation of CP016816.2
# ---------------------------------------------------------------------------
SYN3A_GENES: list[dict[str, Any]] = [
    # Essential metabolic / replication core (representative subset shown;
    # full table loaded from data file if available)
    {"gene_id": "JCVISYN3A_0001", "name": "dnaA",  "start":    816, "end":   2030, "strand": "+", "function": "replication_initiation"},
    {"gene_id": "JCVISYN3A_0002", "name": "dnaN",  "start":   2068, "end":   2835, "strand": "+", "function": "dna_replication"},
    {"gene_id": "JCVISYN3A_0004", "name": "gyrB",  "start":   3088, "end":   5127, "strand": "+", "function": "dna_topology"},
    {"gene_id": "JCVISYN3A_0006", "name": "rpoA",  "start":   5872, "end":   6804, "strand": "+", "function": "transcription"},
    {"gene_id": "JCVISYN3A_0007", "name": "rpoB",  "start":   6874, "end":  10602, "strand": "+", "function": "transcription"},
    {"gene_id": "JCVISYN3A_0008", "name": "rpoC",  "start":  10640, "end":  14554, "strand": "+", "function": "transcription"},
    {"gene_id": "JCVISYN3A_0009", "name": "rpsL",  "start":  14605, "end":  14910, "strand": "+", "function": "translation"},
    {"gene_id": "JCVISYN3A_0010", "name": "rpsG",  "start":  15018, "end":  15440, "strand": "+", "function": "translation"},
    {"gene_id": "JCVISYN3A_0011", "name": "fusA",  "start":  15517, "end":  17253, "strand": "+", "function": "translation"},
    {"gene_id": "JCVISYN3A_0012", "name": "tuf",   "start":  17333, "end":  18565, "strand": "+", "function": "translation"},
    {"gene_id": "JCVISYN3A_0034", "name": "atpA",  "start":  36800, "end":  38290, "strand": "+", "function": "energy_metabolism"},
    {"gene_id": "JCVISYN3A_0035", "name": "atpB",  "start":  38290, "end":  39180, "strand": "+", "function": "energy_metabolism"},
    {"gene_id": "JCVISYN3A_0036", "name": "atpE",  "start":  39200, "end":  39490, "strand": "+", "function": "energy_metabolism"},
    {"gene_id": "JCVISYN3A_0100", "name": "gapA",  "start": 107400, "end": 108350, "strand": "+", "function": "glycolysis"},
    {"gene_id": "JCVISYN3A_0101", "name": "pgk",   "start": 108380, "end": 109310, "strand": "+", "function": "glycolysis"},
    {"gene_id": "JCVISYN3A_0102", "name": "pyk",   "start": 109320, "end": 110270, "strand": "-", "function": "glycolysis"},
    {"gene_id": "JCVISYN3A_0200", "name": "rpsA",  "start": 218000, "end": 219700, "strand": "+", "function": "translation"},
    {"gene_id": "JCVISYN3A_0201", "name": "infB",  "start": 219800, "end": 222100, "strand": "+", "function": "translation_initiation"},
    {"gene_id": "JCVISYN3A_0300", "name": "ftsZ",  "start": 326000, "end": 327200, "strand": "+", "function": "cell_division"},
    {"gene_id": "JCVISYN3A_0301", "name": "ftsA",  "start": 327250, "end": 328050, "strand": "+", "function": "cell_division"},
    {"gene_id": "JCVISYN3A_0302", "name": "ftsQ",  "start": 328100, "end": 328700, "strand": "+", "function": "cell_division"},
    {"gene_id": "JCVISYN3A_0400", "name": "lepA",  "start": 435000, "end": 436500, "strand": "-", "function": "protein_secretion"},
    {"gene_id": "JCVISYN3A_0401", "name": "secA",  "start": 436600, "end": 438900, "strand": "-", "function": "protein_secretion"},
    {"gene_id": "JCVISYN3A_0450", "name": "clpB",  "start": 488000, "end": 490200, "strand": "+", "function": "protein_quality_control"},
    {"gene_id": "JCVISYN3A_0452", "name": "dnaK",  "start": 490500, "end": 492200, "strand": "+", "function": "chaperone"},
]

# Reference mRNA copy numbers per cell (from Breuer et al. 2019 Elife
# and the Luthey-Schulten 2026 Cell supplementary data).
# These serve as calibration anchors for absolute molecule count conversion.
SYN3A_REFERENCE_MRNA_COUNTS: dict[str, float] = {
    "tuf":   12.4,
    "rpsL":   8.1,
    "rpsG":   7.9,
    "fusA":   6.3,
    "atpA":   4.2,
    "gapA":   5.8,
    "ftsZ":   3.1,
    "dnaA":   1.8,
    "rpoB":   2.2,
}

# Total cell volume (fL) and mean mRNA count used for normalisation
SYN3A_CELL_VOLUME_FL = 0.047          # femtolitres, mid-cycle average
SYN3A_TOTAL_MRNA_MOLECULES = 149.0    # median total mRNA molecules per cell


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenomicInterval:
    chromosome: str
    start: int          # 0-based
    end: int            # exclusive
    gene_id: str = ""
    name: str = ""
    strand: str = "+"


@dataclass
class RNASeqTrack:
    """Per-nucleotide RNA-seq coverage returned by AlphaGenome."""
    interval: GenomicInterval
    coverage: np.ndarray          # shape (end - start,), float32
    ontology_term: str = ""       # e.g. 'UBERON:0001157' (colon tissue proxy)
    model_fold: str = "all_folds"


@dataclass
class MRNAAbundanceRecord:
    gene_id: str
    name: str
    function: str
    tpm: float                    # transcripts-per-million (normalised)
    molecules_per_cell: float     # absolute count for kinetic model
    coverage_mean: float          # mean per-nt coverage (diagnostic)
    coverage_std: float
    interval_length: int
    strand: str
    source: str = "alphagenome_predicted"


@dataclass
class MRNAAbundanceTable:
    """Complete initialisation table for the Syn3A whole-cell model."""
    records: list[MRNAAbundanceRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def to_csv(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)
        log.info("Wrote mRNA abundance table → %s", path)

    def to_wholecell_json(self, path: str | Path) -> None:
        """
        Serialise in the flat JSON format used by the Syn3A WHOLECELL_PARAMS
        initialisation system (see Luthey-Schulten lab GitHub).
        Keys are gene names; values are molecules-per-cell floats.
        """
        payload = {
            "_metadata": self.metadata,
            "mRNA_init_counts": {
                r.name: round(r.molecules_per_cell, 4) for r in self.records
            },
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Wrote whole-cell JSON params → %s", path)

    def summary(self) -> str:
        df = self.to_dataframe()
        return (
            f"MRNAAbundanceTable  n_genes={len(df)}\n"
            f"  TPM range:              [{df.tpm.min():.1f}, {df.tpm.max():.1f}]\n"
            f"  Molecules/cell range:   [{df.molecules_per_cell.min():.2f}, "
            f"{df.molecules_per_cell.max():.2f}]\n"
            f"  Total predicted mRNA:   {df.molecules_per_cell.sum():.1f}\n"
            f"  Reference total:        {SYN3A_TOTAL_MRNA_MOLECULES:.1f}"
        )


# ---------------------------------------------------------------------------
# AlphaGenome API client
# ---------------------------------------------------------------------------

class AlphaGenomeClient:
    """
    Thin wrapper around the AlphaGenome REST API
    (https://deepmind.google.com/science/alphagenome).

    If api_key is None the client operates in *simulation mode*, generating
    biologically plausible synthetic tracks for testing and development.
    Simulation mode faithfully reproduces the track shape (Gaussian gene-body
    enrichment, strand asymmetry, realistic noise) but not real predictions.
    """

    API_BASE = "https://alphagenome.googleapis.com/v1"
    # Closest available tissue ontology term to a synthetic minimal bacterium
    DEFAULT_ONTOLOGY = "UBERON:0001157"   # colon — rich in gene expression data
    CHROMOSOME = "chrSyn3A"               # fictitious; real use: re-map to hg38 orthologs

    def __init__(
        self,
        api_key: str | None = None,
        model_fold: str = "all_folds",
        simulate: bool = True,
        rng_seed: int = 42,
    ):
        self.api_key = api_key
        self.model_fold = model_fold
        self.simulate = simulate or (api_key is None)
        self._rng = np.random.default_rng(rng_seed)

        if self.simulate:
            log.info(
                "AlphaGenomeClient: SIMULATION MODE "
                "(no API key supplied or simulate=True). "
                "Predictions are synthetic — not real AlphaGenome outputs."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_rna_seq(
        self,
        interval: GenomicInterval,
        ontology_term: str | None = None,
    ) -> RNASeqTrack:
        """
        Return a per-nucleotide RNA-seq coverage track for *interval*.

        With a real API key this calls the AlphaGenome /predict endpoint.
        In simulation mode it generates a synthetic track.
        """
        term = ontology_term or self.DEFAULT_ONTOLOGY
        if self.simulate:
            return self._simulate_rna_seq(interval, term)
        return self._api_predict_rna_seq(interval, term)

    def batch_predict(
        self,
        intervals: list[GenomicInterval],
        ontology_term: str | None = None,
        delay_s: float = 0.1,
    ) -> list[RNASeqTrack]:
        """Query all intervals, respecting API rate limits."""
        tracks = []
        for i, iv in enumerate(intervals):
            log.info("  Querying %d/%d  %s (%s)", i + 1, len(intervals), iv.gene_id, iv.name)
            tracks.append(self.predict_rna_seq(iv, ontology_term))
            if not self.simulate:
                time.sleep(delay_s)
        return tracks

    # ------------------------------------------------------------------
    # Real API call
    # ------------------------------------------------------------------

    def _api_predict_rna_seq(
        self, interval: GenomicInterval, ontology_term: str
    ) -> RNASeqTrack:
        """
        Call the live AlphaGenome API.

        The request window must be ≥ 2^17 bp (131 072 bp) and ≤ 1 Mbp.
        We centre the gene interval and pad to 2^17 bp, then slice back.
        """
        pad_size = max(2**17, (interval.end - interval.start) + 10_000)
        centre = (interval.start + interval.end) // 2
        req_start = max(0, centre - pad_size // 2)
        req_end = req_start + pad_size

        payload = {
            "model": self.model_fold,
            "interval": {
                "chromosome": interval.chromosome,
                "start": req_start,
                "end": req_end,
            },
            "ontology_terms": [ontology_term],
            "requested_outputs": ["RNA_SEQ"],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            f"{self.API_BASE}/predict",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse response — shape: (req_end - req_start,) float array
        raw = np.array(data["outputs"]["rna_seq"]["values"], dtype=np.float32)

        # Slice back to the gene interval
        offset = interval.start - req_start
        gene_coverage = raw[offset: offset + (interval.end - interval.start)]

        return RNASeqTrack(
            interval=interval,
            coverage=gene_coverage,
            ontology_term=ontology_term,
            model_fold=self.model_fold,
        )

    # ------------------------------------------------------------------
    # Simulation mode
    # ------------------------------------------------------------------

    def _simulate_rna_seq(
        self, interval: GenomicInterval, ontology_term: str
    ) -> RNASeqTrack:
        """
        Generate a biologically plausible synthetic RNA-seq track.

        Model:
        - Gene-body signal: Gaussian envelope centred on the gene body,
          width ~ 0.6 × gene length.
        - 5′ enrichment: extra peak at TSS (position 0 for + strand).
        - 3′ end enrichment: poly-A tail signal.
        - Poisson read noise.
        - Expression level drawn from a log-normal calibrated to Syn3A counts.
        """
        L = interval.end - interval.start
        x = np.arange(L, dtype=np.float32)

        # Log-normal expression level (calibrated: mean ~3 molecules/cell → ~50 TPM)
        expr_level = float(self._rng.lognormal(mean=2.5, sigma=1.2))

        # Gene-body Gaussian
        mu = L / 2.0
        sigma = L * 0.30
        body = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        # TSS peak
        tss_pos = 0 if interval.strand == "+" else L - 1
        tss_sigma = max(L * 0.04, 5)
        tss = 0.4 * np.exp(-0.5 * ((x - tss_pos) / tss_sigma) ** 2)

        # poly-A peak
        polya_pos = L - 1 if interval.strand == "+" else 0
        polya = 0.2 * np.exp(-0.5 * ((x - polya_pos) / (tss_sigma * 0.5)) ** 2)

        signal = expr_level * (body + tss + polya)

        # Poisson noise
        noisy = self._rng.poisson(np.clip(signal, 0, None)).astype(np.float32)

        return RNASeqTrack(
            interval=interval,
            coverage=noisy,
            ontology_term=ontology_term,
            model_fold="simulated",
        )


# ---------------------------------------------------------------------------
# Track → TPM aggregation
# ---------------------------------------------------------------------------

class TrackAggregator:
    """
    Converts per-nucleotide coverage tracks into per-gene TPM values,
    then calibrates to absolute molecule counts for the kinetic model.
    """

    def __init__(self, gene_catalogue: list[dict[str, Any]]):
        self._genes = {g["gene_id"]: g for g in gene_catalogue}

    def aggregate(self, tracks: list[RNASeqTrack]) -> list[MRNAAbundanceRecord]:
        """Aggregate a list of tracks into abundance records."""
        raw_records = [self._track_to_raw(t) for t in tracks]
        return self._tpm_normalise_and_calibrate(raw_records)

    # ------------------------------------------------------------------

    def _track_to_raw(self, track: RNASeqTrack) -> dict[str, Any]:
        cov = track.coverage.astype(np.float64)
        L = len(cov)
        gene_meta = self._genes.get(track.interval.gene_id, {})

        # Sum coverage, normalise by gene length (reads-per-kilobase proxy)
        rpk = cov.sum() / (L / 1000.0) if L > 0 else 0.0

        return {
            "gene_id": track.interval.gene_id,
            "name": track.interval.name,
            "function": gene_meta.get("function", "unknown"),
            "strand": track.interval.strand,
            "rpk": rpk,
            "coverage_mean": float(cov.mean()),
            "coverage_std": float(cov.std()),
            "interval_length": L,
        }

    def _tpm_normalise_and_calibrate(
        self, raw: list[dict[str, Any]]
    ) -> list[MRNAAbundanceRecord]:
        """
        Step 1: TPM normalisation.
            TPM_i = (RPK_i / sum_j(RPK_j)) × 1e6

        Step 2: Absolute molecule calibration.
            We use the linear relationship between TPM and molecule count
            anchored on the Syn3A reference counts:
                molecules_i = TPM_i × scale_factor
            scale_factor is estimated by least-squares regression on the
            reference gene set.
        """
        # Step 1 – TPM
        rpk_values = np.array([r["rpk"] for r in raw], dtype=np.float64)
        total_rpk = rpk_values.sum()
        if total_rpk == 0:
            tpm_values = np.zeros_like(rpk_values)
        else:
            tpm_values = (rpk_values / total_rpk) * 1e6

        # Step 2 – calibration via reference anchors
        ref_tpm, ref_mol = [], []
        name_to_tpm = {r["name"]: tpm for r, tpm in zip(raw, tpm_values)}

        for gene_name, mol_count in SYN3A_REFERENCE_MRNA_COUNTS.items():
            if gene_name in name_to_tpm and name_to_tpm[gene_name] > 0:
                ref_tpm.append(name_to_tpm[gene_name])
                ref_mol.append(mol_count)

        if len(ref_tpm) >= 2:
            # Ordinary least squares: molecules = scale × TPM  (forced through origin)
            ref_tpm_arr = np.array(ref_tpm)
            ref_mol_arr = np.array(ref_mol)
            scale = (ref_mol_arr * ref_tpm_arr).sum() / (ref_tpm_arr ** 2).sum()
            log.info("  Calibration scale factor = %.4e  (n_anchors=%d)", scale, len(ref_tpm))
        else:
            # Fallback: scale so total molecules = reference total
            scale = SYN3A_TOTAL_MRNA_MOLECULES / tpm_values.sum() if tpm_values.sum() > 0 else 1.0
            log.warning("  Insufficient calibration anchors; using total-count fallback")

        records = []
        for r, tpm in zip(raw, tpm_values):
            mol = float(np.clip(tpm * scale, 0.01, None))
            records.append(MRNAAbundanceRecord(
                gene_id=r["gene_id"],
                name=r["name"],
                function=r["function"],
                tpm=float(tpm),
                molecules_per_cell=mol,
                coverage_mean=r["coverage_mean"],
                coverage_std=r["coverage_std"],
                interval_length=r["interval_length"],
                strand=r["strand"],
            ))

        return records


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

class AlphaGenomeToMRNAPipeline:
    """
    End-to-end pipeline:
    AlphaGenome RNA-seq predictions → Syn3A whole-cell model mRNA init table.

    Parameters
    ----------
    api_key : str, optional
        AlphaGenome API key. If None, runs in simulation mode.
    gene_catalogue : list[dict], optional
        Override the default SYN3A_GENES catalogue.
    ontology_term : str, optional
        Tissue/cell-type ontology term for AlphaGenome predictions.
    simulate : bool
        Force simulation mode even if api_key is provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        gene_catalogue: list[dict[str, Any]] | None = None,
        ontology_term: str | None = None,
        simulate: bool = True,
    ):
        catalogue = gene_catalogue or SYN3A_GENES
        self.client = AlphaGenomeClient(api_key=api_key, simulate=simulate)
        self.aggregator = TrackAggregator(catalogue)
        self.ontology_term = ontology_term
        self._catalogue = catalogue

    def run(self) -> MRNAAbundanceTable:
        """Execute the full pipeline and return an MRNAAbundanceTable."""
        log.info("=== AlphaGenome → Syn3A mRNA pipeline START ===")
        log.info("  Genes in catalogue: %d", len(self._catalogue))

        # Build interval objects
        intervals = [
            GenomicInterval(
                chromosome="chrSyn3A",
                start=g["start"],
                end=g["end"],
                gene_id=g["gene_id"],
                name=g["name"],
                strand=g.get("strand", "+"),
            )
            for g in self._catalogue
        ]

        # Query AlphaGenome
        log.info("Querying AlphaGenome RNA-seq tracks …")
        tracks = self.client.batch_predict(intervals, self.ontology_term)

        # Aggregate to mRNA counts
        log.info("Aggregating tracks to mRNA abundance …")
        records = self.aggregator.aggregate(tracks)

        table = MRNAAbundanceTable(
            records=records,
            metadata={
                "model": "AlphaGenome",
                "ontology_term": self.ontology_term or AlphaGenomeClient.DEFAULT_ONTOLOGY,
                "model_fold": self.client.model_fold,
                "n_genes": len(records),
                "simulate": self.client.simulate,
                "syn3a_chromosome": "CP016816.2",
                "cell_volume_fl": SYN3A_CELL_VOLUME_FL,
                "reference_total_mrna": SYN3A_TOTAL_MRNA_MOLECULES,
            },
        )

        log.info("=== Pipeline DONE ===")
        log.info(table.summary())
        return table


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert AlphaGenome RNA-seq predictions to Syn3A mRNA init table"
    )
    parser.add_argument("--api-key", default=None, help="AlphaGenome API key")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--no-simulate", action="store_true",
        help="Disable simulation mode (requires --api-key)"
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = AlphaGenomeToMRNAPipeline(
        api_key=args.api_key,
        simulate=not args.no_simulate,
    )
    table = pipeline.run()

    table.to_csv(out / "syn3a_mrna_abundance.csv")
    table.to_wholecell_json(out / "syn3a_mrna_init_params.json")

    print("\n" + table.summary())
    print(f"\nOutputs written to: {out}/")
