# SynthoHive AI Agent Context

> **Version**: 4.0 (Exhaustive)  
> **Purpose**: The "Single Source of Truth" technical reference for AI agents.    
> **Scope**: Implementation details, Internal APIs, Testing Patterns, Dependencies, and Architecture.

## 1. Project Specifications

### 1.1 Dependency Graph (`pyproject.toml`)
*   **Core**: `numpy>=1.21.0`, `pandas>=1.3.0`, `torch>=1.10.0`
*   **Distributed**: `pyspark>=3.2.0`, `delta-spark>=2.0.0`, `pyarrow>=8.0.0`
*   **Validation**: `scipy>=1.7.0`, `scikit-learn>=1.0.0`
*   **Utilities**: `pydantic>=2.0.0`, `faker>=13.0.0`, `structlog>=21.1.0`
*   **Dev**: `pytest>=7.0.0`

### 1.2 Directory Structure
```
syntho_hive/
├── interface/          # Public API Surface
│   ├── synthesizer.py  # Main Entry Point
│   └── config.py       # Pydantic Schemas (Metadata, Privacy)
├── core/
│   ├── models/         # Generative Backends
│   │   ├── ctgan.py    # WGAN-GP Implementation
│   │   ├── layers.py   # Neural Building Blocks
│   │   └── base.py     # Abstract Base Classes
│   └── data/
│       └── transformer.py # Invertible Data Transforms
├── relational/         # Orchestration Logic
│   ├── orchestrator.py # Training/Generation Coordinator
│   ├── graph.py        # Dependency DAG
│   └── linkage.py      # Cardinality Modeling (GMM)
├── privacy/            # PII Handling
│   ├── sanitizer.py    # Detection & Scrubbing
│   └── faker_contextual.py # Context-aware Generation
├── validation/         # QA
│   ├── report_generator.py # HTML Reporting
│   └── statistical.py      # KS/TVD/Correlation Checks
└── connectors/         # IO & Integration
    ├── spark_io.py     # Spark/Delta Wrappers
    └── sampling.py     # Relational Stratified Sampling
```

## 2. API Specifications (Strict)

### 2.1 Interface Layer

#### `Synthesizer`
**File**: `syntho_hive/interface/synthesizer.py`  
**Purpose**: Orchestrates the fit-sample-validate lifecycle.

*   `__init__(self, metadata: Metadata, privacy_config: PrivacyConfig, spark_session: Optional[SparkSession] = None, backend: str = "CTGAN", embedding_threshold: int = 50)`
    *   `embedding_threshold`: Columns with unique values > this are treated as embeddings.
    *   **Side Effect**: Initializes `StagedOrchestrator` if `spark_session` is provided.

*   `fit(self, data: Union[str, Dict[str, str]], sampling_strategy: str = "relational_stratified", sample_size: int = 5_000_000, validate: bool = False, epochs: int = 300, batch_size: int = 500, **model_kwargs)`
    *   `data`: Either a Hive Database Name (str) or a logical map `{table: uri}`.
    *   `epochs`: Default 300. Controls CTGAN training duration.
    *   **Raises**: `ValueError` if Spark is missing.

*   `sample(self, num_rows: Dict[str, int], output_format: str = "delta", output_path: Optional[str] = None) -> Dict[str, str]`
    *   `num_rows`: Count for **Root Tables** only. Child counts are derived.
    *   **Returns**: Map of `{table_name: absolute_output_path}`.

#### `Metadata` (Pydantic Model)
**File**: `syntho_hive/interface/config.py`

*   `add_table(name: str, pk: str, **kwargs)`
    *   `kwargs`:
        *   `fk`: `Dict[str, str]` -> `{local_col: "parent_table.parent_col"}`
        *   `constraints`: `Dict[str, Constraint]`
        *   `parent_context_cols`: `List[str]` (Attributes to condition on)

### 2.2 Core Internal APIs

#### `CTGAN`
**File**: `syntho_hive/core/models/ctgan.py`

*   `fit(data: pd.DataFrame, context: Optional[pd.DataFrame], ...)`
    *   **Input Shape**: `data` (N, D), `context` (N, C)
    *   **Logic**:
        1.  `DataTransformer.fit_transform` -> `data_tensor`
        2.  `generator = ResidualLayer(noise_dim + context_dim)`
        3.  `discriminator = Discriminator(data_dim + context_dim)`
        4.  Training Loop (WGAN-GP):
            *   Critic (D) updates 5 times (`discriminator_steps`) per 1 Generator update.
            *   Gradient Penalty is applied to enforce 1-Lipschitz continuity.

*   `sample(num_rows: int, context: Optional[pd.DataFrame]) -> pd.DataFrame`
    *   **Input**: `context` must be provided (N, C) if trained with context.
    *   **Logic**:
        1.  `noise = randn(N, embedding_dim)`
        2.  `fake_raw = G(cat(noise, context))`
        3.  `DataTransformer.inverse_transform(fake_raw)`

#### `DataTransformer`
**File**: `syntho_hive/core/data/transformer.py`

*   **Continuous Columns**:
    *   **Model**: `BayesianGaussianMixture` (VGM) with `n_components=10`.
    *   **Encoded**: `[OneHot(Cluster_Assignment) | Scalar(Normalized_Value)]`.
    *   **Dim**: $10 + 1 = 11$ dimensions per column.
*   **Categorical Columns**:
    *   **Low Card**: `OneHotEncoder`. Dim = $K$.
    *   **High Card**: `LabelEncoder`. Dim = $1$ (Integer Index).

#### `EntityEmbeddingLayer`
**File**: `syntho_hive/core/models/layers.py`

*   **Forward**: Standard `nn.Embedding`.
*   **ForwardSoft**: `matmul(probs, weights)`. Used during generator training to allow backprop through categorical sampling (Discrete -> Continuous relaxation).

#### `LinkageModel`
**File**: `syntho_hive/relational/linkage.py`

*   **Method**: `GaussianMixture` for distribution of child counts $N_c$.
*   **Fit**: Aggregates `child.groupby(FK).size()`. Joins with Parent PKs to include Zeros.
*   **Sample**: Samples from GMM. $N_c = \text{clip}(\text{round}(x), 0, \infty)$.

### 2.3 Orchestration & Connectors

#### `StagedOrchestrator`
**File**: `syntho_hive/relational/orchestrator.py`

*   **Algorithm**:
    1.  `graph.get_generation_order()` -> Topological Sort.
    2.  For `table` in order:
        *   **Training**:
            *   If Child: Load Driver Parent + Child. Join. Train Linkage. Train CTGAN with Context.
        *   **Generation**:
            *   If Child: Load generated Parent. Sample Counts. Replicate Context. Sample CTGAN. Assign FKs.
    3.  **Secondary FKs**: Assigned via random sampling from valid keys in secondary parent tables (Independent Assumption).

#### `SparkIO`
**File**: `syntho_hive/connectors/spark_io.py`

*   **Format Handling**: Auto-detects `file://`, `.csv`, `.parquet`.
*   **Write**: Defaults to `parquet` but supports arbitrary types via kwargs.

#### `RelationalSampler`
**File**: `syntho_hive/connectors/sampling.py`

*   `sample_relational(root_table, sample_size, stratify_by)`
    *   **Root**: Stratified Sample or Random Sample.
    *   **Cascading**: `Child = Child.join(SampledRoot, on=FK, how="semi")`.
    *   **Constraint**: Assumes tree structure; complex DAGs might lose records if not careful with join direction.

### 2.4 Privacy & Validation

#### `PIISanitizer`
**File**: `syntho_hive/privacy/sanitizer.py`

*   **Actions**:
    *   `mask`: Keeps last 4 chars.
    *   `hash`: SHA256.
    *   `fake`: Uses `ContextualFaker`.
*   **ContextualFaker**:
    *   Checks row context: `country`, `locale`, or `region`.
    *   Loads locale-specific `Faker` (e.g., `ja_JP` for `country="Japan"`).

#### `StatisticalValidator`
**File**: `syntho_hive/validation/statistical.py`

*   **KS Test**: $P > 0.05$ (Pass).
*   **TVD**: $Distance < 0.1$ (Pass).

## 3. Testing Patterns (From `tests/`)
*   **E2E Scenarios** (`tests/e2e_scenarios/`):
    *   Full lifecycle tests: Generate Fake Ground Truth -> Sanitize -> Train -> Generate -> Validate.
    *   Uses absolute paths in Metadata: `file:///.../data.csv`.
*   **Unit Tests**:
    *   **Privacy**: `test_privacy.py` checks detection accuracy.
    *   **Models**: `test_models.py` checks tensor shapes and gradient flows.
*   **Spark Fixtures**: Tests typically check for `pyspark` presence and skip if missing.

## 4. Known Constraints & Edge Cases
1.  **Memory**: `CTGAN` fits entire transformers in memory. High cardinality (>1k categories) on non-embedded columns can OOM.
2.  **Parent Context**: Currently only joins with the *Driver Parent*. Context from secondary parents is not fed to the model (only FK integrity is maintained).
3.  **Linkage**: Assumes conditional independence of child counts given parent attributes.
4.  **Data Types**: Complex types (Array, Map) are not supported by `DataTransformer`. Flatten them before input.
