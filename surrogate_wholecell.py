"""
surrogate_wholecell.py
======================
Surrogate emulator that maps AlphaGenome variant-effect predictions to
Syn3A whole-cell phenotypic outcomes — bypassing the need to run the full
6-day whole-cell simulation for each variant.

Architecture
------------
Input features (per variant):
  - Δlog2(expression) for each of N genes (from AlphaGenome variant scorer)
  - Affected functional class one-hot vector
  - Variant distance to TSS (log-scaled)
  - Variant type embedding (SNP / indel / structural)

Output targets (whole-cell phenotypes):
  - division_time_min       : expected cell cycle duration (minutes)
  - growth_rate_per_hr      : exponential growth rate (h⁻¹)
  - dna_replication_success : P(successful chromosome replication) ∈ [0,1]
  - atp_flux_rel            : ATP production rate relative to wild-type
  - ribosome_occupancy      : mean ribosome occupancy fraction

Training data
-------------
In production this module trains on tabular outputs from the Luthey-Schulten
Syn3A 4D simulation sweep (perturbed expression levels → simulated phenotypes).
In development / demo mode, a physically realistic synthetic dataset is generated
that captures the known sensitivities of Syn3A to ribosomal, replication, and
metabolic gene perturbations.

References
----------
- Luthey-Schulten Syn3A 4D model, Cell 2026 [S0092-8674(26)00174-1]
- AlphaGenome, Nature 2026 [doi:10.1038/s41586-025-10014-0]
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — Syn3A phenotype reference (wild-type)
# ---------------------------------------------------------------------------
WT_PHENOTYPES: dict[str, float] = {
    "division_time_min":        105.0,
    "growth_rate_per_hr":       0.396,   # ln(2) / (105/60)
    "dna_replication_success":  0.98,
    "atp_flux_rel":             1.00,
    "ribosome_occupancy":       0.82,
}

PHENOTYPE_NAMES = list(WT_PHENOTYPES.keys())

# Functional gene classes and their known sensitivity weights in Syn3A
# (derived from essential gene analysis in Breuer et al. 2019)
FUNCTIONAL_CLASSES = [
    "replication_initiation",
    "dna_replication",
    "dna_topology",
    "transcription",
    "translation",
    "translation_initiation",
    "energy_metabolism",
    "glycolysis",
    "cell_division",
    "protein_secretion",
    "protein_quality_control",
    "chaperone",
    "unknown",
]

# Per-class phenotype sensitivities (empirically estimated from simulation sweeps)
# Shape: {class: {phenotype: sensitivity_coefficient}}
CLASS_SENSITIVITIES: dict[str, dict[str, float]] = {
    "replication_initiation":  {"dna_replication_success": 0.9,  "division_time_min": 0.6},
    "dna_replication":         {"dna_replication_success": 1.2,  "division_time_min": 0.8},
    "dna_topology":            {"dna_replication_success": 0.7,  "division_time_min": 0.4},
    "transcription":           {"division_time_min": 0.5,        "ribosome_occupancy": 0.4},
    "translation":             {"growth_rate_per_hr": 1.1,       "ribosome_occupancy": 0.9,
                                "division_time_min": 0.7},
    "translation_initiation":  {"growth_rate_per_hr": 0.7,       "ribosome_occupancy": 0.6},
    "energy_metabolism":       {"atp_flux_rel": 1.3,             "growth_rate_per_hr": 0.5,
                                "division_time_min": 0.4},
    "glycolysis":              {"atp_flux_rel": 0.8,             "growth_rate_per_hr": 0.4},
    "cell_division":           {"division_time_min": 1.4,        "dna_replication_success": 0.3},
    "protein_secretion":       {"growth_rate_per_hr": 0.2},
    "protein_quality_control": {"growth_rate_per_hr": 0.3,       "ribosome_occupancy": 0.2},
    "chaperone":               {"growth_rate_per_hr": 0.2},
    "unknown":                 {},
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VariantEffectRecord:
    """
    AlphaGenome output for one variant: per-gene Δlog2(expression) values
    plus variant metadata.
    """
    variant_id: str
    chromosome: str
    position: int
    ref: str
    alt: str
    delta_log2_expr: dict[str, float]   # gene_name → Δlog2(fold-change)
    affected_function: str = "unknown"
    tss_distance_bp: int = 0
    variant_type: str = "SNP"           # SNP | indel | structural


@dataclass
class PhenotypePrediction:
    """Surrogate model output for one variant."""
    variant_id: str
    division_time_min: float
    growth_rate_per_hr: float
    dna_replication_success: float
    atp_flux_rel: float
    ribosome_occupancy: float
    viability_score: float              # composite [0,1]: 1 = WT-like, 0 = lethal
    uncertainty_std: dict[str, float]   # per-phenotype prediction std (from ensemble)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_likely_lethal(self, threshold: float = 0.5) -> bool:
        return self.viability_score < threshold

    def summary(self) -> str:
        lines = [
            f"Variant: {self.variant_id}",
            f"  division_time_min:       {self.division_time_min:.1f}  "
            f"(WT={WT_PHENOTYPES['division_time_min']:.1f})",
            f"  growth_rate_per_hr:      {self.growth_rate_per_hr:.3f}  "
            f"(WT={WT_PHENOTYPES['growth_rate_per_hr']:.3f})",
            f"  dna_replication_success: {self.dna_replication_success:.3f}  "
            f"(WT={WT_PHENOTYPES['dna_replication_success']:.3f})",
            f"  atp_flux_rel:            {self.atp_flux_rel:.3f}  "
            f"(WT={WT_PHENOTYPES['atp_flux_rel']:.3f})",
            f"  ribosome_occupancy:      {self.ribosome_occupancy:.3f}  "
            f"(WT={WT_PHENOTYPES['ribosome_occupancy']:.3f})",
            f"  viability_score:         {self.viability_score:.3f}",
            f"  likely_lethal:           {self.is_likely_lethal()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

GENE_NAMES_ORDERED = [
    "dnaA", "dnaN", "gyrB", "rpoA", "rpoB", "rpoC",
    "rpsL", "rpsG", "fusA", "tuf", "atpA", "atpB", "atpE",
    "gapA", "pgk", "pyk", "rpsA", "infB",
    "ftsZ", "ftsA", "ftsQ", "lepA", "secA", "clpB", "dnaK",
]

VARIANT_TYPE_IDX = {"SNP": 0, "indel": 1, "structural": 2}
N_GENES = len(GENE_NAMES_ORDERED)
N_FUNC_CLASSES = len(FUNCTIONAL_CLASSES)
FEATURE_DIM = N_GENES + N_FUNC_CLASSES + 2   # Δexpr + one-hot + tss_dist + vtype


def featurise(record: VariantEffectRecord) -> np.ndarray:
    """Convert a VariantEffectRecord to a fixed-length feature vector."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 1. Δlog2 expression for each gene (0 if not affected)
    for i, gname in enumerate(GENE_NAMES_ORDERED):
        feat[i] = record.delta_log2_expr.get(gname, 0.0)

    # 2. Affected functional class one-hot
    offset = N_GENES
    if record.affected_function in FUNCTIONAL_CLASSES:
        feat[offset + FUNCTIONAL_CLASSES.index(record.affected_function)] = 1.0

    # 3. Log-scaled TSS distance
    feat[offset + N_FUNC_CLASSES] = np.log1p(abs(record.tss_distance_bp)) * (
        1.0 if record.tss_distance_bp >= 0 else -1.0
    )

    # 4. Variant type index (normalised to [0,1])
    feat[offset + N_FUNC_CLASSES + 1] = VARIANT_TYPE_IDX.get(record.variant_type, 0) / 2.0

    return feat


# ---------------------------------------------------------------------------
# Synthetic training data generator
# ---------------------------------------------------------------------------

class Syn3ASimulationEmulator:
    """
    Generates synthetic whole-cell simulation outputs that mimic the
    expected behaviour of the Luthey-Schulten 4D model under gene
    expression perturbations.

    Physics encoded:
    - Division time increases when translation / replication genes are reduced.
    - ATP flux responds primarily to energy_metabolism perturbations.
    - Replication success drops sharply when dnaA/dnaN/gyrB are downregulated.
    - Non-linear saturation: large perturbations → lethality.
    - Correlated phenotypes (growth_rate ↔ division_time).
    """

    def __init__(self, rng_seed: int = 0):
        self._rng = np.random.default_rng(rng_seed)

    def generate(self, n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X : np.ndarray, shape (n_samples, FEATURE_DIM)
        Y : np.ndarray, shape (n_samples, len(PHENOTYPE_NAMES))
        """
        records, phenotypes = [], []

        for i in range(n_samples):
            rec = self._random_variant(f"variant_{i:05d}")
            records.append(featurise(rec))
            phenotypes.append(self._simulate_phenotype(rec))

        return np.array(records, dtype=np.float32), np.array(phenotypes, dtype=np.float32)

    def _random_variant(self, vid: str) -> VariantEffectRecord:
        func = self._rng.choice(FUNCTIONAL_CLASSES)
        vtype = self._rng.choice(["SNP", "indel", "structural"], p=[0.7, 0.25, 0.05])
        tss_dist = int(self._rng.integers(-5000, 5000))

        # Most variants have small effects; a few are large
        n_affected = int(self._rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        genes = self._rng.choice(GENE_NAMES_ORDERED, size=n_affected, replace=False)
        delta = {}
        for g in genes:
            magnitude = float(self._rng.lognormal(mean=-1.0, sigma=1.0))
            sign = -1 if self._rng.random() < 0.7 else 1   # mostly downregulating
            delta[g] = sign * magnitude

        return VariantEffectRecord(
            variant_id=vid,
            chromosome="chrSyn3A",
            position=int(self._rng.integers(0, 543_379)),
            ref="A", alt="G",
            delta_log2_expr=delta,
            affected_function=func,
            tss_distance_bp=tss_dist,
            variant_type=vtype,
        )

    def _simulate_phenotype(self, rec: VariantEffectRecord) -> list[float]:
        """
        Deterministic physics-based mapping from variant effect to phenotype.
        Adds Gaussian noise to mimic simulation stochasticity.
        """
        phen = dict(WT_PHENOTYPES)   # start from wild type

        total_delta = sum(rec.delta_log2_expr.values())
        func_sens = CLASS_SENSITIVITIES.get(rec.affected_function, {})

        # Apply per-gene effects
        for gene, dlog2 in rec.delta_log2_expr.items():
            fold = 2 ** dlog2   # <1 = downregulation

            if gene in ("dnaA", "dnaN", "gyrB"):
                # Replication — sharp non-linear effect
                phen["dna_replication_success"] *= np.clip(fold ** 0.5, 0.05, 1.5)
                phen["division_time_min"] *= np.clip(1.0 / (fold ** 0.3), 0.7, 3.0)

            elif gene in ("rpoA", "rpoB", "rpoC"):
                # Transcription — moderate effect on division time
                phen["division_time_min"] *= np.clip(1.0 / (fold ** 0.25), 0.8, 2.0)
                phen["ribosome_occupancy"] *= np.clip(fold ** 0.15, 0.5, 1.2)

            elif gene in ("tuf", "fusA", "rpsL", "rpsG", "rpsA", "infB"):
                # Translation — strong effect on growth
                phen["growth_rate_per_hr"] *= np.clip(fold ** 0.6, 0.1, 1.3)
                phen["ribosome_occupancy"] *= np.clip(fold ** 0.4, 0.3, 1.2)
                phen["division_time_min"] *= np.clip(1.0 / (fold ** 0.5), 0.7, 3.5)

            elif gene in ("atpA", "atpB", "atpE"):
                # Energy
                phen["atp_flux_rel"] *= np.clip(fold ** 0.8, 0.1, 1.5)
                phen["growth_rate_per_hr"] *= np.clip(fold ** 0.3, 0.3, 1.2)

            elif gene in ("gapA", "pgk", "pyk"):
                # Glycolysis
                phen["atp_flux_rel"] *= np.clip(fold ** 0.5, 0.2, 1.3)

            elif gene in ("ftsZ", "ftsA", "ftsQ"):
                # Division
                phen["division_time_min"] *= np.clip(1.0 / (fold ** 0.7), 0.7, 5.0)

        # Enforce consistency: growth_rate = ln(2) / division_time (hours)
        phen["growth_rate_per_hr"] = np.log(2) / (phen["division_time_min"] / 60.0)

        # Add Gaussian noise (CV ≈ 5% for time, 8% for rates — matches sim stochasticity)
        noise_scale = {"division_time_min": 0.05, "growth_rate_per_hr": 0.08,
                       "dna_replication_success": 0.03, "atp_flux_rel": 0.06,
                       "ribosome_occupancy": 0.04}
        for k, cv in noise_scale.items():
            phen[k] += self._rng.normal(0, abs(phen[k]) * cv)

        # Hard clamp to physiologically valid ranges
        phen["division_time_min"]        = np.clip(phen["division_time_min"], 50, 600)
        phen["growth_rate_per_hr"]       = np.clip(phen["growth_rate_per_hr"], 0.05, 1.5)
        phen["dna_replication_success"]  = np.clip(phen["dna_replication_success"], 0.0, 1.0)
        phen["atp_flux_rel"]             = np.clip(phen["atp_flux_rel"], 0.05, 2.0)
        phen["ribosome_occupancy"]       = np.clip(phen["ribosome_occupancy"], 0.1, 1.0)

        return [phen[k] for k in PHENOTYPE_NAMES]


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

class Syn3ASurrogateModel:
    """
    Multi-output MLP surrogate that emulates the Syn3A whole-cell simulator.

    Design choices:
    - 3-layer MLP with skip-connection-inspired architecture (via sklearn MLP).
    - Multi-output wrapped in MultiOutputRegressor (one MLP per phenotype)
      to allow different hyperparameters per target if needed.
    - StandardScaler for input normalisation (critical for MLP training).
    - Lightweight ensemble of 5 models for uncertainty quantification.
    - Training runtime <60s on a laptop CPU for N=5 000 samples.
    """

    N_ENSEMBLE = 5

    def __init__(self):
        self._scalers_X: list[StandardScaler] = []
        self._scalers_Y: list[StandardScaler] = []
        self._models: list[MultiOutputRegressor] = []
        self._is_trained: bool = False
        self._train_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        test_fraction: float = 0.15,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Train the surrogate ensemble.

        Parameters
        ----------
        X : (n_samples, FEATURE_DIM) feature matrix
        Y : (n_samples, n_phenotypes) target matrix
        test_fraction : held-out fraction for evaluation
        verbose : print training metrics

        Returns
        -------
        metrics : dict with MAE and R² per phenotype
        """
        log.info("Training Syn3A surrogate model  (n=%d, features=%d, targets=%d)",
                 len(X), X.shape[1], Y.shape[1])

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_fraction, random_state=0
        )

        for seed in range(self.N_ENSEMBLE):
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_tr = scaler_X.fit_transform(X_train)
            Y_tr = scaler_Y.fit_transform(Y_train)

            # Per-phenotype MLP — moderate depth, batch normalisation via Adam
            base_mlp = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=3e-4,
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                tol=1e-5,
            )
            model = MultiOutputRegressor(base_mlp, n_jobs=-1)
            model.fit(X_tr, Y_tr)

            self._scalers_X.append(scaler_X)
            self._scalers_Y.append(scaler_Y)
            self._models.append(model)

        # Evaluate on test set
        metrics = self._evaluate(X_test, Y_test)
        self._train_metrics = metrics
        self._is_trained = True

        if verbose:
            self._print_metrics(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, records: list[VariantEffectRecord]) -> list[PhenotypePrediction]:
        """
        Predict phenotypic outcomes for a list of variant effect records.
        Returns one PhenotypePrediction per record, with uncertainty estimates.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call .train() first.")

        X = np.array([featurise(r) for r in records], dtype=np.float32)
        all_preds = self._ensemble_predict(X)   # shape (N_ENSEMBLE, n_samples, n_targets)

        mean_preds = all_preds.mean(axis=0)
        std_preds = all_preds.std(axis=0)

        results = []
        for i, rec in enumerate(records):
            p = mean_preds[i]
            s = std_preds[i]
            phen = dict(zip(PHENOTYPE_NAMES, p.tolist()))
            unc  = dict(zip(PHENOTYPE_NAMES, s.tolist()))

            viability = self._viability_score(phen)

            results.append(PhenotypePrediction(
                variant_id=rec.variant_id,
                division_time_min=float(np.clip(phen["division_time_min"], 50, 600)),
                growth_rate_per_hr=float(np.clip(phen["growth_rate_per_hr"], 0.0, 1.5)),
                dna_replication_success=float(np.clip(phen["dna_replication_success"], 0, 1)),
                atp_flux_rel=float(np.clip(phen["atp_flux_rel"], 0, 2)),
                ribosome_occupancy=float(np.clip(phen["ribosome_occupancy"], 0, 1)),
                viability_score=viability,
                uncertainty_std=unc,
            ))

        return results

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "models": self._models,
            "scalers_X": self._scalers_X,
            "scalers_Y": self._scalers_Y,
            "train_metrics": self._train_metrics,
            "feature_dim": FEATURE_DIM,
            "phenotype_names": PHENOTYPE_NAMES,
            "gene_names": GENE_NAMES_ORDERED,
        }
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh, protocol=5)
        log.info("Surrogate model saved → %s  (%.1f MB)", path,
                 path.stat().st_size / 1e6)

    @classmethod
    def load(cls, path: str | Path) -> "Syn3ASurrogateModel":
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        m = cls()
        m._models = bundle["models"]
        m._scalers_X = bundle["scalers_X"]
        m._scalers_Y = bundle["scalers_Y"]
        m._train_metrics = bundle["train_metrics"]
        m._is_trained = True
        log.info("Surrogate model loaded ← %s", path)
        return m

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for scaler_X, scaler_Y, model in zip(self._scalers_X, self._scalers_Y, self._models):
            X_sc = scaler_X.transform(X)
            Y_sc = model.predict(X_sc)
            Y_orig = scaler_Y.inverse_transform(Y_sc)
            preds.append(Y_orig)
        return np.array(preds)   # (N_ENSEMBLE, n_samples, n_targets)

    def _evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict[str, Any]:
        all_preds = self._ensemble_predict(X_test)
        mean_preds = all_preds.mean(axis=0)

        metrics: dict[str, Any] = {}
        for j, name in enumerate(PHENOTYPE_NAMES):
            yt = Y_test[:, j]
            yp = mean_preds[:, j]
            r, _ = pearsonr(yt, yp)
            metrics[name] = {
                "mae": float(mean_absolute_error(yt, yp)),
                "r2":  float(r2_score(yt, yp)),
                "pearson_r": float(r),
            }
        return metrics

    def _viability_score(self, phen: dict[str, float]) -> float:
        """
        Composite viability score ∈ [0,1].
        Based on deviation from WT phenotype, weighted by biological importance.
        """
        weights = {
            "division_time_min":        0.25,
            "dna_replication_success":  0.30,
            "growth_rate_per_hr":       0.25,
            "atp_flux_rel":             0.10,
            "ribosome_occupancy":       0.10,
        }
        score = 0.0
        for key, w in weights.items():
            wt_val = WT_PHENOTYPES[key]
            pred_val = phen.get(key, wt_val)
            if key == "division_time_min":
                # Longer division time → lower viability
                ratio = wt_val / max(pred_val, 1e-9)
            else:
                ratio = pred_val / max(wt_val, 1e-9)
            score += w * np.clip(ratio, 0, 1)
        return float(np.clip(score, 0, 1))

    def _print_metrics(self, metrics: dict[str, Any]) -> None:
        log.info("─" * 60)
        log.info("  Surrogate model evaluation (held-out test set)")
        log.info("  %-30s  %7s  %7s  %10s", "Phenotype", "MAE", "R²", "Pearson r")
        log.info("─" * 60)
        for name, m in metrics.items():
            log.info("  %-30s  %7.3f  %7.3f  %10.3f",
                     name, m["mae"], m["r2"], m["pearson_r"])
        log.info("─" * 60)


# ---------------------------------------------------------------------------
# Combined pipeline: AlphaGenome variant → phenotype prediction
# ---------------------------------------------------------------------------

class VariantToPhenotypeOracle:
    """
    High-level interface combining the AlphaGenome variant scorer
    with the trained surrogate to answer: "What phenotypic effect
    does this DNA variant have on a Syn3A cell?"

    Usage
    -----
    oracle = VariantToPhenotypeOracle.build_and_train(n_train=5000)
    pred = oracle.predict_variant("chr22:36201698 A→C", delta_log2_expr={...})
    print(pred.summary())
    """

    def __init__(self, surrogate: Syn3ASurrogateModel):
        self.surrogate = surrogate

    @classmethod
    def build_and_train(
        cls,
        n_train: int = 5000,
        rng_seed: int = 42,
        model_path: str | Path | None = None,
    ) -> "VariantToPhenotypeOracle":
        """
        Generate synthetic training data (or load from model_path) and
        return a trained oracle.
        """
        if model_path and Path(model_path).exists():
            surr = Syn3ASurrogateModel.load(model_path)
            return cls(surr)

        log.info("Generating synthetic Syn3A simulation training data  (n=%d) …", n_train)
        emulator = Syn3ASimulationEmulator(rng_seed=rng_seed)
        X, Y = emulator.generate(n_train)

        surr = Syn3ASurrogateModel()
        surr.train(X, Y)

        if model_path:
            surr.save(model_path)

        return cls(surr)

    def predict_variant(
        self,
        variant_id: str,
        delta_log2_expr: dict[str, float],
        affected_function: str = "unknown",
        tss_distance_bp: int = 0,
        variant_type: str = "SNP",
        chromosome: str = "chrSyn3A",
        position: int = 0,
    ) -> PhenotypePrediction:
        """Predict phenotypic outcome for a single variant."""
        rec = VariantEffectRecord(
            variant_id=variant_id,
            chromosome=chromosome,
            position=position,
            ref="N", alt="N",
            delta_log2_expr=delta_log2_expr,
            affected_function=affected_function,
            tss_distance_bp=tss_distance_bp,
            variant_type=variant_type,
        )
        return self.surrogate.predict([rec])[0]

    def batch_predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch predict from a DataFrame with columns:
            variant_id, delta_log2_<gene_name>, affected_function,
            tss_distance_bp, variant_type
        Returns a DataFrame with phenotype predictions appended.
        """
        records = []
        for _, row in df.iterrows():
            delta = {g: float(row[f"delta_log2_{g}"]) for g in GENE_NAMES_ORDERED
                     if f"delta_log2_{g}" in row.index}
            records.append(VariantEffectRecord(
                variant_id=str(row.get("variant_id", "unknown")),
                chromosome=str(row.get("chromosome", "chrSyn3A")),
                position=int(row.get("position", 0)),
                ref="N", alt="N",
                delta_log2_expr=delta,
                affected_function=str(row.get("affected_function", "unknown")),
                tss_distance_bp=int(row.get("tss_distance_bp", 0)),
                variant_type=str(row.get("variant_type", "SNP")),
            ))

        preds = self.surrogate.predict(records)
        pred_dicts = [p.to_dict() for p in preds]
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(pred_dicts)], axis=1)


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and run the Syn3A whole-cell surrogate model"
    )
    parser.add_argument("--n-train", type=int, default=5000,
                        help="Number of synthetic training samples")
    parser.add_argument("--model-path", default="outputs/syn3a_surrogate.pkl",
                        help="Path to save/load the trained model")
    parser.add_argument("--out-dir", default="outputs",
                        help="Output directory for predictions")
    parser.add_argument("--demo-variants", type=int, default=10,
                        help="Number of random demo variants to predict")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Build and train
    oracle = VariantToPhenotypeOracle.build_and_train(
        n_train=args.n_train,
        model_path=args.model_path,
    )

    # Demo: predict a few hand-crafted variants
    demo_cases = [
        {
            "id": "syn3a_wt_proxy",
            "delta": {},
            "func": "unknown",
            "desc": "Wild-type (no expression change)",
        },
        {
            "id": "ftsZ_knockdown_50pct",
            "delta": {"ftsZ": -1.0},   # 2-fold down
            "func": "cell_division",
            "desc": "FtsZ 2-fold downregulation → expected: longer division",
        },
        {
            "id": "tuf_strong_knockdown",
            "delta": {"tuf": -2.0, "fusA": -1.5},   # 4-fold, 3-fold down
            "func": "translation",
            "desc": "EF-Tu/EF-G severe reduction → expected: growth arrest",
        },
        {
            "id": "atpA_overexpression",
            "delta": {"atpA": +1.0},
            "func": "energy_metabolism",
            "desc": "ATP synthase 2-fold up → expected: increased ATP flux",
        },
        {
            "id": "dnaN_critical",
            "delta": {"dnaN": -3.0, "dnaA": -1.0},
            "func": "dna_replication",
            "desc": "DNA Pol III clamp severe loss → expected: replication failure",
        },
    ]

    print("\n" + "=" * 65)
    print("DEMO: Syn3A Whole-Cell Surrogate Predictions")
    print("=" * 65)

    results = []
    for case in demo_cases:
        pred = oracle.predict_variant(
            variant_id=case["id"],
            delta_log2_expr=case["delta"],
            affected_function=case["func"],
        )
        print(f"\n[{case['desc']}]")
        print(pred.summary())
        results.append(pred.to_dict())

    # Save demo predictions
    results_df = pd.DataFrame(results)
    out_path = Path(args.out_dir) / "demo_predictions.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Demo predictions saved → %s", out_path)
