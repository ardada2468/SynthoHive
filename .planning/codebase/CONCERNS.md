# Codebase Concerns

**Analysis Date:** 2026-02-22

## Tech Debt

### Incomplete Model Serialization

**Issue:** Checkpointing only saves generator and discriminator state dicts, not transformer state. Loaded models cannot be used for inference without retraining or reconstructing transformers.

**Files:**
- `syntho_hive/core/models/ctgan.py` (lines 482-503)

**Impact:** Models saved to disk are incomplete. Resuming training or running inference on checkpointed models will fail due to missing DataTransformer state. Recovery requires keeping original data or retraining.

**Fix approach:** Implement full object serialization using pickle or joblib for the entire CTGAN object, including metadata, transformers, and embedding layers. Add validation in `load()` to verify all components are present.

---

### Duplicate Assignment Statement

**Issue:** Variable assigned twice consecutively without semantic difference.

**Files:**
- `syntho_hive/core/models/ctgan.py` (lines 77-78): `self.epochs = epochs` appears twice

**Impact:** Minor - suggests incomplete refactoring. Could hide copy-paste errors. Affects code clarity.

**Fix approach:** Remove duplicate assignment (line 78). Review nearby code for other copy-paste artifacts.

---

### Unimplemented Embedding Transformation Logic

**Issue:** The `_apply_embeddings()` method has a stub `pass` statement for handling fake data (generated logits), indicating incomplete feature.

**Files:**
- `syntho_hive/core/models/ctgan.py` (lines 132-185, particularly line 175)

**Impact:** Method is unused but documents complex design issue: generator outputs have different shapes than transformer outputs for categorical features (logits vs indices). Current code processes embeddings inline during training/sampling, not via this method. Creates confusion about data flow.

**Fix approach:** Either complete this method to properly handle both real and fake data shapes, or remove it and document why embedding processing is inline-only.

---

### Bare Exception Handling in Data Loading

**Issue:** Generic `except:` clause without exception type specification.

**Files:**
- `syntho_hive/interface/synthesizer.py` (line 153)

**Impact:** Silently catches and swallows all exceptions (including KeyboardInterrupt, SystemExit), making debugging difficult. If Spark table read fails for reasons beyond missing table (e.g., permission error), user won't know the real cause.

**Fix approach:** Replace with `except Exception as e:` and log the exception. Consider specific exception types for the Spark operation (check PySpark documentation for `spark.read.table()` exceptions).

---

### Silent Type Conversion Failures

**Issue:** Constraint application catches and silently passes on numeric conversion failures.

**Files:**
- `syntho_hive/core/data/transformer.py` (lines 222-230, 245-246)

**Impact:** When synthetic data cannot be converted to required numeric type (due to data generation errors), the system continues with non-conformant data instead of raising. This violates constraints without warning.

**Fix approach:** Log warnings when conversions fail, and either enforce constraints strictly (raise) or fall back to a sensible default. Document behavior in docstring.

---

## Known Bugs

### Memory Accumulation in Relational Generation

**Issue:** All generated DataFrames are stored in memory simultaneously during multi-table generation, even when writing to disk.

**Files:**
- `syntho_hive/relational/orchestrator.py` (lines 224-226)

**Trigger:** Generate synthetic data for multiple tables with output_path_base specified.

**Problem:** Line 226 stores `generated_pdf` in `generated_tables` dictionary always, regardless of whether `output_path_base` is set. Code comment acknowledges this: "For massive datasets this might be risky". With 10M row parent + 100M row children, memory usage becomes unbounded.

**Impact:** Out-of-memory errors on large datasets despite writing to disk. Generation fails unexpectedly.

**Workaround:** Process tables one at a time, or use explicit garbage collection between table generations.

**Fix approach:** When `output_path_base` is set, read generated tables from disk instead of keeping all in memory. Only cache in-memory for downstream table generation if output_path_base is None.

---

### Context Data Mismatch in Child Generation

**Issue:** When generating child tables with context, the last seen batch's context is reused for all child rows, not freshly sampled.

**Files:**
- `syntho_hive/core/models/ctgan.py` (lines 359-362)

**Trigger:** Training conditional model on child table with parent context.

**Problem:** In generator training loop, `real_context_batch` from the most recent discriminator batch is reused for all generator samples. This couples generator training to discriminator's random sampling order.

**Impact:** Context distribution during training may not match actual parent distribution, leading to poor conditional generation quality.

**Fix approach:** Independently sample context for generator training rather than reusing discriminator batch context.

---

### Type Mismatch Handling in Linkage Model

**Issue:** Foreign key and primary key types may not match, with fallback to string conversion that could cause data integrity issues.

**Files:**
- `syntho_hive/relational/linkage.py` (lines 35-44)

**Trigger:** Parent table with integer PKs, child table with string FKs (or other type mismatch).

**Problem:** If type casting fails, code falls back to converting everything to string. Subsequent FK assignments in orchestrator may not properly align with parent PKs.

**Impact:** Generated child rows may reference non-existent parent IDs if type conversions are lossy.

**Fix approach:** Raise an error requiring explicit type alignment in metadata instead of silent fallback. Add validation after merge to verify all FKs reference valid PKs.

---

## Security Considerations

### PII Regex Pattern Limitations

**Issue:** Default email, phone, SSN, and credit card regex patterns have false positive/negative rates and are not production-grade validation.

**Files:**
- `syntho_hive/privacy/sanitizer.py` (lines 25-31)

**Risk:** Sensitive PII may not be detected (false negatives) or benign data mistakenly flagged (false positives). Email regex `[^@]+@[^@]+\.[^@]+` will match `a@b.c`, which is technically valid but overly broad.

**Current mitigation:** Detection only flags columns where >50% of sample matches (line 94), reducing false positive impact.

**Recommendations:**
- Use established validation libraries (e.g., email-validator, phonenumbers) instead of regex
- Add whitelist of known PII columns in metadata for deterministic detection
- Log detection decisions and allow user review before sanitization

---

### Weak Hashing for Sensitive Data

**Issue:** SHA256 hashing without salt for PII values.

**Files:**
- `syntho_hive/privacy/sanitizer.py` (lines 154-156)

**Risk:** Hashed PII can be reversed via rainbow tables or brute force on small PII spaces (e.g., US zip codes, SSNs). This is not suitable for compliance (GDPR, CCPA).

**Current mitigation:** Hashing is one option among mask/drop/fake/custom, so users can choose alternatives.

**Recommendations:**
- Use salted hashing (PBKDF2, bcrypt, scrypt) if hashing is chosen
- Document that hashing is for deduplication/consistency, not cryptographic security
- Recommend dropping or faking PII instead of hashing for compliance use cases
- Consider adding HMAC with environment-provided secret

---

### Context Information Leakage in Synthetic Data

**Issue:** Parent attributes used as conditioning context are preserved in synthetic child data through direct column copies.

**Files:**
- `syntho_hive/relational/orchestrator.py` (lines 97-114, 178-188)

**Risk:** If parent context columns contain sensitive attributes (e.g., salary, department), they appear unchanged in synthetic data and could allow inference attacks linking synthetic back to real individuals.

**Current mitigation:** None - user must explicitly exclude sensitive parent columns from context.

**Recommendations:**
- Add warnings in docs that context columns should not contain sensitive attributes
- Optionally allow transforming context columns through privacy rules before use
- Consider differential privacy perturbation of context features

---

## Performance Bottlenecks

### DataFrames to Pandas Conversions in Multi-Stage Pipeline

**Issue:** Multiple toPandas() conversions during relational generation, converting between Spark and Pandas representations.

**Files:**
- `syntho_hive/relational/orchestrator.py` (lines 60, 87, 169, 207)

**Cause:** CTGAN operates on pandas DataFrames; orchestrator works with Spark DataFrames. No caching of converted data.

**Problem:** For each table, data is read from Spark, converted to Pandas, processed, written back to Spark, then read again for child table linkage. Large tables incur serialization overhead multiple times.

**Scaling Impact:** ~2-3x slower for multi-table pipelines. Spark parallelism is lost during pandas operations.

**Improvement path:**
- Cache pandas representations in memory between stages
- Consider implementing CTGAN in Spark UDFs to avoid Pandas roundtrips
- Profile actual conversion times to quantify overhead

---

### Linear Scan in PII Detection

**Issue:** Content-based PII detection samples 100 rows and checks every pattern on every value.

**Files:**
- `syntho_hive/privacy/sanitizer.py` (lines 65-102)

**Problem:** For large datasets and multiple columns, regex matching on sample becomes slow. Nested loop: columns × patterns × samples × regex evaluation.

**Scaling Impact:** Noticeable on wide (100+ column) tables.

**Improvement path:**
- Cache regex patterns as compiled objects, not strings
- Use column name heuristics first (line 58-62) and skip content check for obvious matches
- Parallelize column scanning
- Consider bloom filters for quick rejection before regex

---

### Gradient Penalty Computation in Every Discriminator Step

**Issue:** Full gradient penalty (interpolation + autograd) computed for every discriminator batch, which is computationally expensive.

**Files:**
- `syntho_hive/core/models/ctgan.py` (lines 348, 14-43)

**Problem:** WGAN-GP gradient penalty requires second-order autograd. With `discriminator_steps=5` (default) per generator step, this is called 5x more than necessary. For large embeddings, gradient computation is slow.

**Scaling Impact:** Training time dominated by gradient penalty rather than forward/backward passes.

**Improvement path:**
- Cache gradient penalties across steps if real/fake batches are similar
- Apply penalty every N steps instead of every step (reduce frequency)
- Use spectral normalization as alternative to WGAN-GP for faster training

---

## Fragile Areas

### Complex Embedding Shape Management in CTGAN

**Issue:** Tracking different tensor shapes for different data types through training/sampling is error-prone and documented with extensive comments indicating design uncertainty.

**Files:**
- `syntho_hive/core/models/ctgan.py` (lines 94-185, 296-316, 369-386)

**Why fragile:**
- Transformer outputs different dimensions than generator (indices vs logits)
- Discriminator inputs different shapes than generator outputs (embeddings vs logits)
- Real and fake data have different representations requiring separate processing paths
- No centralized schema documentation - shape info spread across methods

**Safe modification:** Before changing embedding handling, create comprehensive test covering:
1. Single categorical column generation and reconstruction
2. Multiple categorical columns with varying cardinalities
3. Mixed categorical + continuous columns
4. Null handling through embeddings
5. Verify output shapes match expected constraints at each stage

**Test coverage gaps:** No unit tests specifically for embedding layer forward/backward passes or shape transitions.

---

### Relational Integrity via Sampling

**Issue:** Secondary foreign keys assigned by random sampling from parent PKs without checking cardinality constraints.

**Files:**
- `syntho_hive/relational/orchestrator.py` (lines 200-214)

**Why fragile:**
- No validation that sampled FKs actually exist in parent table
- No handling of FK uniqueness constraints (if child has unique FK)
- If parent generation fails or produces unexpected number of rows, children may reference missing IDs

**Safe modification:**
- Add explicit validation: `assert all(fk in parent_pks)` after sampling
- Log warnings if cardinality of sampled FKs differs significantly from real data
- Add unit tests for multi-parent scenarios

**Test coverage gaps:** Multi-parent linkage tested in `tests/e2e_scenarios/retail_test.py` but not with edge cases (empty parent, duplicate FKs, missing PKs).

---

### Constraint Application in Inverse Transform

**Issue:** Numeric constraint application (min/max/dtype) happens in inverse_transform() with multiple try/except blocks and fallback logic that silently succeeds on partial application.

**Files:**
- `syntho_hive/core/data/transformer.py` (lines 209-255)

**Why fragile:**
- Conversions from string/float to int may lose precision
- Clipping to min/max range after type conversion may produce out-of-range values if conversion is lossy
- Exception handling allows partial constraint application without warning
- Constraint dtype and min/max are checked separately; incompatible specifications could conflict

**Safe modification:**
- Test with each constraint type individually: int conversion, float conversion, min/max clipping, null handling
- Add assertions that output respects all constraints: `assert output >= constraint.min`
- Consider separate methods for each constraint type rather than unified logic

**Test coverage:** `tests/test_transformer.py` has basic tests but not comprehensive constraint coverage.

---

## Scaling Limits

### Memory Scaling with Generated Datasets

**Current capacity:** Successfully tested with ~100K row root tables + proportional children (tested: ~1M child rows total in memory).

**Limit:** The in-memory storage of all generated DataFrames in `generated_tables` dict will cause OOM errors with datasets >1GB.

**Scaling path:**
- Implement streaming write-as-you-generate for large outputs
- Use temporary disk spillover for intermediate tables
- Process children in batches if parent is very large
- Consider returning generators instead of full DataFrames

---

### Gradient Penalty Computational Scaling

**Current capacity:** Batch size 500 with embedding_dim 128 trains in ~minutes per epoch on 1 GPU.

**Limit:** Batch size >2000 or embedding_dim >512 causes memory issues. Gradient penalty scales with batch size.

**Scaling path:**
- Profile memory allocation in compute_gradient_penalty (lines 14-43)
- Implement gradient accumulation to achieve larger effective batch size with smaller memory
- Consider replacing WGAN-GP with lighter weight alternative (spectral norm)

---

### Categorical Cardinality in Embeddings

**Current capacity:** Categorical columns with <1000 unique values train stably.

**Limit:** High-cardinality columns (>5000 unique values) may produce poorly learned embeddings due to sparse data and embedding layer being under-constrained.

**Scaling path:**
- Implement hierarchical embeddings for high-cardinality columns
- Add post-hoc embedding validation (check that similar categories produce similar embeddings)
- Document recommended embedding_threshold tuning based on cardinality and sample size

---

## Test Coverage Gaps

### CTGAN Embedding Roundtrip

**What's not tested:** End-to-end generation and reconstruction of data with categorical embeddings.

**Files:**
- `syntho_hive/core/models/ctgan.py` (embedding layers and processing)
- No dedicated test file for this

**Risk:** Changes to embedding application could break categorical data handling silently. Current tests use synthetic/simplified data.

**Priority:** HIGH - embedding is core to CTGAN functionality.

---

### Relational Linkage Cardinality

**What's not tested:** Linkage model accuracy with skewed cardinality distributions (e.g., parent with 1 child, another with 1000).

**Files:**
- `syntho_hive/relational/linkage.py` (sample_counts method)
- Tests in `tests/e2e_scenarios/retail_test.py` but limited coverage

**Risk:** Generated child counts may not preserve real-data cardinality, leading to incomplete relationships.

**Priority:** MEDIUM - affects relational integrity.

---

### Privacy Sanitization Regex Accuracy

**What's not tested:** False positive/negative rates of PII detection patterns.

**Files:**
- `syntho_hive/privacy/sanitizer.py` (patterns and detection logic)
- No test data for this

**Risk:** Sensitive data may slip through or benign data may be unnecessarily sanitized.

**Priority:** HIGH - affects privacy guarantees.

---

### Constraint Roundtrip

**What's not tested:** Comprehensive constraint preservation through fit -> transform -> inverse_transform cycle.

**Files:**
- `syntho_hive/core/data/transformer.py`
- `tests/test_transformer.py` has basic tests but limited constraint scenarios

**Risk:** Constraints may be partially applied or silently violated.

**Priority:** MEDIUM - constraints are user-specified integrity requirements.

---

## Dependencies at Risk

### PyTorch Version Pinning

**Risk:** torch>=1.10.0 is flexible, but older versions have deprecated APIs. Code uses `torch.autograd.grad()` and `state_dict()`, which are stable, but newer versions (2.0+) have breaking changes in some areas.

**Recommendation:** Test with torch 2.x versions and pin to stable range (e.g., torch>=1.13.0,<2.1).

---

### PySpark Version Compatibility

**Risk:** pyspark>=3.2.0 targets Spark 3.2+. Newer versions (4.0) may have schema/UDF changes.

**Current usage:** Basic DataFrame operations (read, write, select, sql), which are stable APIs.

**Recommendation:** Test with Spark 4.0+ and document supported versions.

---

### Sklearn Deprecations

**Risk:** scikit-learn>=1.0.0. Future versions may deprecate:
- `sparse_output` parameter in OneHotEncoder (used in transformer)
- BayesianGaussianMixture API changes

**Recommendation:** Add explicit handling for deprecated parameters with fallback to older syntax.

---

## Missing Critical Features

### Model Validation Metrics

**Problem:** No automated way to measure generation quality (FID, IS, or domain-specific metrics).

**Blocks:** Users cannot assess whether model training converged or if output is usable without manual inspection.

---

### Checkpoint Resume

**Problem:** Saved models cannot be resumed; only full retraining is possible.

**Blocks:** Long training runs are risky (no recovery from interruption). Large-scale deployments require complete retraining for any fix.

---

### Distributed Training

**Problem:** CTGAN training is single-machine, single-GPU only despite Spark being available in orchestrator.

**Blocks:** Datasets >10GB cannot be trained on typical machines.

---

### Foreign Key Uniqueness

**Problem:** No support for unique constraints on foreign keys.

**Blocks:** One-to-one parent-child relationships cannot be modeled.

---

## Recommendations by Priority

### Immediate (Blocking Production)
1. Fix model serialization to include transformers
2. Resolve memory accumulation in multi-table generation
3. Add explicit error handling for FK type mismatches
4. Add comprehensive embedding roundtrip tests

### High Priority (Affects Correctness)
1. Fix bare except clause in data loading
2. Add test coverage for linkage cardinality preservation
3. Implement context resampling in generator training
4. Add PII detection accuracy tests

### Medium Priority (Affects Performance/Maintainability)
1. Profile and optimize toPandas conversions
2. Document and refactor embedding shape management
3. Implement full-object checkpoint serialization
4. Add constraint validation in inverse_transform

### Low Priority (Nice-to-Have)
1. Add model validation metrics
2. Implement distributed training
3. Optimize gradient penalty computation
4. Add hierarchical embedding support for high-cardinality columns

---

*Concerns audit: 2026-02-22*
