---
title: Relational
---

# Relational Orchestration

This module handles the complexity of multi-table generation, ensuring referential integrity and statistical correlation between tables.

## Orchestrator

::: syntho_hive.relational.orchestrator.StagedOrchestrator
    options:
      members:
        - fit_all
        - generate

## Linkage

::: syntho_hive.relational.linkage.LinkageModel
    options:
      members:
        - fit
        - sample_counts

## Graph & Schema

::: syntho_hive.relational.graph.SchemaGraph
