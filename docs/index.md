---
title: SynthoHive
---

# SynthoHive Documentation

SynthoHive is a production-grade synthetic data engine that generates high-fidelity, privacy-preserving synthetic data for complex relational databases. It maintains referential integrity across multi-table schemas, preserves statistical correlations, and provides automated PII handling.

## What you'll find

- [**Getting Started**](getting-started.md): Install SynthoHive and run your first synthesis in minutes.
- **Concepts**: Understand the [architecture](architecture.md) and [data flow](data-flow.md) behind the engine.
- **Guides**: Step-by-step instructions for [fitting models](guides/fitting.md), [relational generation](guides/relational.md), [embeddings](guides/embeddings.md), [sampling](guides/sampling.md), [privacy](guides/privacy.md), and [validation](guides/validation.md).
- [**Demos**](demos/overview.md): Runnable walk-throughs mirroring the `examples/demos` folder.
- **API Reference**: Auto-generated documentation for the [interface](api/interface.md), [core models](api/core.md), [privacy](api/privacy.md), [relational](api/relational.md), [validation](api/validation.md), and [connectors](api/connectors.md) modules.
- [**Config Examples**](reference/config-examples.md): Copy-paste configurations for common scenarios.
- [**Troubleshooting**](reference/troubleshooting.md): Solutions for common issues.

## Quick install

```bash
pip install synthohive pyspark pandas pyarrow
```

## Build docs locally

```bash
pip install .[docs]
mkdocs serve
```

Deploy to GitHub Pages:

```bash
mkdocs gh-deploy --force
```
