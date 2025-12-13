---
title: Architecture
---

# Architecture

SynthoHive is organized into clear packages:

- **interface**: `Synthesizer` façade, `Metadata`, `TableConfig`, `PrivacyConfig` entry points.
- **core**: `DataTransformer` for normalization/encoding, and `CTGAN` (Conditional WGAN-GP) for deep generative modeling. [See API](api/core.md).
- **relational**: `StagedOrchestrator` managing the generation DAG, and `LinkageModel` for parent-child cardinality learning. [See API](api/relational.md).
- **privacy**: `PIISanitizer` with regex-based detection, and `ContextualFaker` for locale-aware obfuscation. [See API](api/privacy.md).
- **validation**: `ValidationReport` and `StatisticalValidator` measuring KS/TVD metrics. [See API](api/validation.md).
- **connectors**: `SparkIO` for scalable I/O.

## Key flows
1. **Fit**: transformers profile tables, CTGAN trains (optionally conditioned on parent context), linkage models learn child counts.
2. **Sample**: generators produce rows, linkage models drive child counts, referential integrity enforced via FK assignment.
3. **Privacy**: sanitizer detects/masks/fakes PII; contextual faker injects locale-aware replacements.
4. **Validation**: KS/TVD per column, correlation distance, preview tables, HTML/JSON report.

See [Data Flow](data-flow.md) for a stepwise diagram and [Guides](guides/fitting.md) for hands-on steps. 
