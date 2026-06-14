"""Typed configuration models for clustro experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KNNImputerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"

    @model_validator(mode="after")
    def validate_n_neighbors(self) -> KNNImputerConfig:
        if self.n_neighbors < 1:
            raise ValueError("data.missingness.knn.n_neighbors must be at least 1.")
        return self


class IterativeImputerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iter: int = 10
    initial_strategy: Literal["mean", "median"] = "median"
    sample_posterior: bool = False
    random_state: int | None = None
    estimator: Literal["bayesian_ridge"] = "bayesian_ridge"

    @model_validator(mode="after")
    def validate_max_iter(self) -> IterativeImputerConfig:
        if self.max_iter < 1:
            raise ValueError("data.missingness.iterative.max_iter must be at least 1.")
        return self


class ContinuousMissingnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continuous_imputer: Literal["median", "knn", "iterative"] = "median"
    categorical_imputer: Literal["most_frequent"] = "most_frequent"
    add_missing_indicators: bool = True
    knn: KNNImputerConfig = Field(default_factory=KNNImputerConfig)
    iterative: IterativeImputerConfig = Field(default_factory=IterativeImputerConfig)


class ColumnSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continuous: list[str] = Field(default_factory=list)
    binary: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    ordinal: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_disjoint(self) -> ColumnSchemaConfig:
        seen: set[str] = set()
        for group in (self.continuous, self.binary, self.categorical, self.ordinal):
            overlap = seen.intersection(group)
            if overlap:
                raise ValueError(f"Column schema overlaps detected: {sorted(overlap)}")
            seen.update(group)
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    id_column: str | None = None
    id_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    column_schema: ColumnSchemaConfig
    ordinal_maps: dict[str, list[Any]] = Field(default_factory=dict)
    missingness: ContinuousMissingnessConfig = Field(default_factory=ContinuousMissingnessConfig)

    @model_validator(mode="after")
    def validate_schema_exclusions(self) -> DataConfig:
        schema_columns = (
            self.column_schema.continuous
            + self.column_schema.binary
            + self.column_schema.categorical
            + self.column_schema.ordinal
        )
        schema_set = set(schema_columns)
        if self.id_column is not None and self.id_column in schema_set:
            raise ValueError("id_column must not be included in column_schema.")
        id_overlap = set(self.id_columns).intersection(schema_set)
        if id_overlap:
            raise ValueError(
                f"id_columns must not be included in column_schema: {sorted(id_overlap)}"
            )
        target_overlap = set(self.target_columns).intersection(schema_set)
        if target_overlap:
            raise ValueError(
                f"target_columns must not be included in column_schema: {sorted(target_overlap)}"
            )

        ordinal_columns = set(self.column_schema.ordinal)
        for column in self.column_schema.ordinal:
            if column not in self.ordinal_maps:
                raise ValueError(
                    f"Ordinal column '{column}' requires an explicit level order in "
                    "data.ordinal_maps."
                )
            levels = self.ordinal_maps[column]
            if not levels:
                raise ValueError(f"data.ordinal_maps['{column}'] must contain at least one level.")
            try:
                unique_count = len(set(levels))
            except TypeError:
                normalized = [repr(level) for level in levels]
                unique_count = len(set(normalized))
            if unique_count != len(levels):
                raise ValueError(f"data.ordinal_maps['{column}'] contains duplicate levels.")
        extra_maps = set(self.ordinal_maps).difference(ordinal_columns)
        if extra_maps:
            raise ValueError(
                f"data.ordinal_maps includes non-ordinal columns: {sorted(extra_maps)}"
            )
        return self


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sampler: str = "TPESampler"
    pruner: str = "MedianPruner"
    n_trials_per_family: int = 50


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pilot_sample_fraction: float = 0.35
    pilot_min_rows: int = 250
    seeds_pilot: list[int] = Field(default_factory=lambda: [101, 102])
    seeds_full: list[int] = Field(default_factory=lambda: [101, 102, 103, 104, 105])
    perturbations_full: int = 5
    perturbation_type: Literal["bootstrap", "subsample"] = "bootstrap"
    stability_mode: Literal["full_pipeline", "processed_matrix"] = "full_pipeline"
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)


class VarianceThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold: float = 0.0


class RareCategoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_frequency: int | float = 0.01
    replacement: str = "__RARE__"

    @model_validator(mode="after")
    def validate_min_frequency(self) -> RareCategoryConfig:
        if self.min_frequency <= 0:
            raise ValueError("rare_category_collapse.min_frequency must be positive.")
        return self


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continuous_transforms: list[str] = Field(default_factory=lambda: ["standard"])
    categorical_encoding: list[str] = Field(default_factory=lambda: ["onehot"])
    rare_category_collapse: RareCategoryConfig = Field(default_factory=RareCategoryConfig)
    variance_threshold: VarianceThresholdConfig = Field(default_factory=VarianceThresholdConfig)


class MethodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RepresentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methods: list[MethodConfig] = Field(default_factory=lambda: [MethodConfig(name="none")])


class ClusteringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methods: list[MethodConfig]


class StructureConstraintsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_clusters: int = 2
    max_clusters: int = 10
    min_cluster_fraction: float = 0.03
    max_noise_fraction: float = 0.35
    dominant_cluster_cap: float = 0.85


class AcceptanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_thresholds: dict[str, float] = Field(default_factory=dict)
    weighted_score: dict[str, float] = Field(default_factory=dict)
    accept_top_fraction_if_above: float = 1.0


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_metrics: list[str] = Field(default_factory=list)
    structure_constraints: StructureConstraintsConfig = Field(
        default_factory=StructureConstraintsConfig
    )
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)


class RunWeightingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "final_weighted_score"
    normalize: bool = True
    floor: float = 0.01


class UncertaintyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bootstrap_repeats: int = 25
    ambiguous_top2_gap_threshold: float = 0.10
    ambiguous_entropy_quantile: float = 0.90


class ConsensusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_only_accepted: bool = True
    run_weighting: RunWeightingConfig = Field(default_factory=RunWeightingConfig)
    consensus_method: str = "hierarchical_on_coassociation"
    final_k_strategy: str = "weighted_mode"
    coassociation_storage: Literal["auto", "dense", "sparse"] = "auto"
    max_dense_n: int = 10000
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)


class ReportingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Controls whether PNG visualisations are written to reports/.
    generate_figures: bool = True
    # Controls whether optional summary CSVs are written to reports/.
    # Core pipeline artefacts (candidate_registry.parquet, consensus_labels.csv, …)
    # are always written regardless of this flag.
    generate_tables: bool = True
    # Controls whether the manuscript_bundle directory is populated.
    manuscript_bundle: bool = True


class ExperimentSectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: str
    random_seed: int = 2026
    n_jobs: int = -1
    use_ray: bool = False
    use_mlflow: bool = False
    use_gpu_if_available: bool = False
    deterministic_mode: Literal["strict", "fast"] = "fast"


class InterpretationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surrogate_model: Literal["xgboost", "random_forest"] = "xgboost"
    feature_space: Literal[
        "original_imputed_scaled",
        "best_candidate_preprocessing",
        "consensus_majority_preprocessing",
    ] = "original_imputed_scaled"
    continuous_transform: str = "standard"
    categorical_encoding: str = "onehot"
    cross_validation_folds: int = 5
    repeated_cv_repeats: int = 3
    use_shap: bool = True
    use_permutation_importance: bool = True
    top_n_features: int = 30
    grouped_correlation_threshold: float = 0.85


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentSectionConfig
    data: DataConfig
    search: SearchConfig = Field(default_factory=SearchConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    representation: RepresentationConfig = Field(default_factory=RepresentationConfig)
    clustering: ClusteringConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    interpretation: InterpretationConfig = Field(default_factory=InterpretationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    config_path: Path | None = None
    resolved_data_path: Path | None = None
    resolved_output_dir: Path | None = None

    def with_runtime_paths(
        self,
        *,
        config_path: Path,
        resolved_data_path: Path,
        resolved_output_dir: Path,
    ) -> ExperimentConfig:
        return self.model_copy(
            update={
                "config_path": config_path,
                "resolved_data_path": resolved_data_path,
                "resolved_output_dir": resolved_output_dir,
            }
        )
