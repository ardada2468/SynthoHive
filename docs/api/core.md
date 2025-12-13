---
title: Core & Models
---

# Core Models & Data

The core module contains the deep learning implementations and data transformation logic.

## Models

::: syntho_hive.core.models.ctgan.CTGAN
    options:
      members:
        - fit
        - sample
        - save
        - load

::: syntho_hive.core.models.base.ConditionalGenerativeModel

## Data Transformation

::: syntho_hive.core.data.transformer.DataTransformer
    options:
      members:
        - fit
        - transform
        - inverse_transform

::: syntho_hive.core.data.transformer.ClusterBasedNormalizer
