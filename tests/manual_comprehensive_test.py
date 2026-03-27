"""
Comprehensive manual end-to-end test of the SynthoHive library.

Generates a realistic e-commerce dataset and exercises:
  A. Privacy Sanitization (PIISanitizer)
  B. Single-Table CTGAN Training & Generation
  C. Statistical Validation
  D. DataTransformer Edge Cases
  E. Seed Reproducibility

Run with:
    .venv/bin/python tests/manual_comprehensive_test.py
"""

import sys
import traceback
import warnings
import logging

import numpy as np
import pandas as pd

# Suppress noisy warnings and structlog output during test
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.disable(logging.CRITICAL)

# Silence structlog
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

# ─────────────────────────────────────────────────────────────
# 1. BUILD THE DATASET
# ─────────────────────────────────────────────────────────────

print("=" * 70)
print("BUILDING COMPLEX E-COMMERCE DATASET")
print("=" * 70)

np.random.seed(42)

FIRST_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "Frank",
    "Grace",
    "Hank",
    "Ivy",
    "Jack",
    "Karen",
    "Leo",
    "Mona",
    "Nate",
    "Olive",
    "Paul",
    "Quinn",
    "Rita",
    "Sam",
    "Tina",
    "Uma",
    "Vic",
    "Wendy",
    "Xander",
    "Yara",
    "Zane",
    "Aria",
    "Blake",
    "Cleo",
    "Drew",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "company.io"]
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]

N_CUSTOMERS = 500
N_ORDERS = 2000

# --- customers ---
customer_ids = list(range(1, N_CUSTOMERS + 1))
first_names = np.random.choice(FIRST_NAMES, N_CUSTOMERS)
emails = [
    f"{fn.lower()}.{ln.lower()}{np.random.randint(1, 999)}@{dom}"
    for fn, ln, dom in zip(
        first_names,
        np.random.choice(LAST_NAMES, N_CUSTOMERS),
        np.random.choice(DOMAINS, N_CUSTOMERS),
    )
]
ages = np.random.normal(35, 12, N_CUSTOMERS).clip(18, 85).astype(int)
incomes = np.random.lognormal(mean=10.8, sigma=0.7, size=N_CUSTOMERS).clip(
    20_000, 500_000
)
regions = np.random.choice(REGIONS, N_CUSTOMERS, p=[0.2, 0.2, 0.25, 0.2, 0.15])
signup_dates = pd.date_range("2020-01-01", "2024-12-31", periods=N_CUSTOMERS).date
is_premium = np.random.choice([True, False], N_CUSTOMERS, p=[0.2, 0.8])

customers = pd.DataFrame(
    {
        "customer_id": customer_ids,
        "first_name": first_names,
        "email": emails,
        "age": ages.astype(float),  # float to allow NaN injection
        "income": incomes,
        "region": regions,
        "signup_date": [str(d) for d in signup_dates],
        "is_premium": pd.Series(is_premium).map({True: "yes", False: "no"}),
    }
)

# Inject nulls
null_age_idx = np.random.choice(N_CUSTOMERS, int(N_CUSTOMERS * 0.05), replace=False)
null_income_idx = np.random.choice(N_CUSTOMERS, int(N_CUSTOMERS * 0.03), replace=False)
customers.loc[null_age_idx, "age"] = np.nan
customers.loc[null_income_idx, "income"] = np.nan

# --- orders ---
order_ids = list(range(1, N_ORDERS + 1))
order_customer_ids = np.random.choice(customer_ids, N_ORDERS)
order_dates = pd.date_range("2020-06-01", "2024-12-31", periods=N_ORDERS).date
customer_income_map = customers.set_index("customer_id")["income"].to_dict()
base_amounts = np.array(
    [customer_income_map.get(cid, 50_000) for cid in order_customer_ids], dtype=float
)
total_amounts = (base_amounts * np.random.uniform(0.001, 0.05, N_ORDERS)).round(2)
statuses = np.random.choice(
    ["completed", "pending", "cancelled", "refunded"],
    N_ORDERS,
    p=[0.6, 0.2, 0.1, 0.1],
)
payment_methods = np.random.choice(
    ["credit_card", "debit_card", "paypal", "bank_transfer"],
    N_ORDERS,
    p=[0.4, 0.25, 0.2, 0.15],
)

orders = pd.DataFrame(
    {
        "order_id": order_ids,
        "customer_id": order_customer_ids,
        "order_date": [str(d) for d in order_dates],
        "total_amount": total_amounts,
        "status": statuses,
        "payment_method": payment_methods,
    }
)

null_amount_idx = np.random.choice(N_ORDERS, int(N_ORDERS * 0.02), replace=False)
orders.loc[null_amount_idx, "total_amount"] = np.nan

print(f"  customers: {customers.shape}")
print(f"  orders:    {orders.shape}")
print(
    f"  customer null rates  -> age: {customers['age'].isna().mean():.2%}, "
    f"income: {customers['income'].isna().mean():.2%}"
)
print(
    f"  order null rates     -> total_amount: {orders['total_amount'].isna().mean():.2%}"
)
print()

# ─────────────────────────────────────────────────────────────
# RESULT TRACKER
# ─────────────────────────────────────────────────────────────
results = {}
synth_customers = None  # Will be set in Test B, used by Test C


# ─────────────────────────────────────────────────────────────
# TEST A — Privacy Sanitization
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST A: Privacy Sanitization")
print("=" * 70)
try:
    from syntho_hive.privacy.sanitizer import PIISanitizer

    sanitizer = PIISanitizer()

    # Analyze
    pii_map = sanitizer.analyze(customers)
    print(f"  Detected PII columns: {pii_map}")

    # The sanitizer maps column names to rule names:
    #   email    -> "email"
    #   first_name -> "name"
    detected_email = pii_map.get("email") == "email"
    detected_name = pii_map.get("first_name") == "name"
    print(f"  email detected as PII:      {detected_email}")
    print(f"  first_name detected as PII: {detected_name}")

    # Sanitize
    sanitized = sanitizer.sanitize(customers, pii_map=pii_map)
    print(f"  Sanitized shape: {sanitized.shape}")

    # Verify PII was handled
    pii_handled = True
    for col, rule_name in pii_map.items():
        if col in sanitized.columns:
            # Column still present — check it was transformed (fake/mask)
            if sanitized[col].equals(customers[col]):
                print(f"  WARNING: column '{col}' appears unchanged after sanitization")
                pii_handled = False
            else:
                print(
                    f"  Column '{col}' ({rule_name}): values differ after sanitization — OK"
                )
        else:
            print(f"  Column '{col}' ({rule_name}): dropped — OK")

    passed_a = detected_email and detected_name and pii_handled
    results["Privacy Sanitization"] = passed_a
    print(f"\n  => {'PASS' if passed_a else 'FAIL'}")

except Exception as exc:
    print(f"  EXCEPTION: {exc}")
    traceback.print_exc()
    results["Privacy Sanitization"] = False

print()

# ─────────────────────────────────────────────────────────────
# TEST B — CTGAN Training & Generation
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST B: CTGAN Training & Generation (50 epochs)")
print("=" * 70)
try:
    from syntho_hive.interface.config import Metadata
    from syntho_hive.core.models.ctgan import CTGAN

    meta = Metadata()
    meta.add_table("customers", pk="customer_id")

    model = CTGAN(
        metadata=meta,
        batch_size=256,
        epochs=50,
        embedding_dim=64,
        generator_dim=(128, 128),
        discriminator_dim=(128, 128),
    )

    # Fit — pass the customers dataframe directly (no Spark needed for CTGAN.fit)
    print("  Training CTGAN on 'customers' table ...")
    model.fit(customers, table_name="customers", seed=42, progress_bar=True)

    # Generate 200 synthetic customers
    print("  Generating 200 synthetic customers ...")
    synth_customers = model.sample(200)
    print(f"  Generated shape: {synth_customers.shape}")
    print(f"  Generated columns: {list(synth_customers.columns)}")

    # Compare statistics
    compare_cols_num = ["age", "income"]
    compare_cols_cat = ["region", "is_premium"]

    print("\n  --- Numeric Column Comparison ---")
    print(
        f"  {'Column':<12} {'Real Mean':>12} {'Synth Mean':>12} "
        f"{'Real Std':>12} {'Synth Std':>12}"
    )
    for col in compare_cols_num:
        if col in synth_customers.columns:
            rm = customers[col].mean()
            sm = synth_customers[col].mean()
            rs = customers[col].std()
            ss = synth_customers[col].std()
            print(f"  {col:<12} {rm:>12.2f} {sm:>12.2f} {rs:>12.2f} {ss:>12.2f}")

    print("\n  --- Categorical Column Comparison ---")
    for col in compare_cols_cat:
        if col in synth_customers.columns:
            real_dist = customers[col].value_counts(normalize=True).sort_index()
            synth_dist = synth_customers[col].value_counts(normalize=True).sort_index()
            print(f"  {col}:")
            all_vals = sorted(set(real_dist.index) | set(synth_dist.index), key=str)
            # Limit output to top 20 categories
            for v in all_vals[:20]:
                rv = real_dist.get(v, 0)
                sv = synth_dist.get(v, 0)
                print(f"    {str(v):<15} Real={rv:.3f}  Synth={sv:.3f}")
            if len(all_vals) > 20:
                print(f"    ... ({len(all_vals) - 20} more categories omitted)")
        else:
            print(f"  {col}: NOT in synthetic output")

    # Check null rate preservation
    print("\n  --- Null Rate Comparison ---")
    for col in ["age", "income"]:
        if col in synth_customers.columns:
            real_null = customers[col].isna().mean()
            synth_null = synth_customers[col].isna().mean()
            print(f"  {col:<12} Real={real_null:.3f}  Synth={synth_null:.3f}")
        else:
            print(f"  {col:<12} NOT in synthetic output (excluded as PK/FK — expected)")

    # Basic sanity: generated data has rows, columns roughly match
    passed_b = (
        synth_customers.shape[0] == 200
        and synth_customers.shape[1] >= 3  # At least some non-PK columns
    )
    results["CTGAN Training & Generation"] = passed_b
    print(f"\n  => {'PASS' if passed_b else 'FAIL'}")

except Exception as exc:
    print(f"  EXCEPTION: {exc}")
    traceback.print_exc()
    results["CTGAN Training & Generation"] = False

print()

# ─────────────────────────────────────────────────────────────
# TEST C — Statistical Validation
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST C: Statistical Validation")
print("=" * 70)
try:
    from syntho_hive.validation.statistical import StatisticalValidator

    if synth_customers is None:
        raise RuntimeError("Test B did not produce synthetic data — cannot validate")

    validator = StatisticalValidator()

    # Need to align columns: synth_customers doesn't have PK columns
    common_cols = [c for c in customers.columns if c in synth_customers.columns]
    real_subset = customers[common_cols].copy()
    synth_subset = synth_customers[common_cols].copy()

    col_results = validator.compare_columns(real_subset, synth_subset)

    print("  --- Column-wise Test Results ---")
    for col, res in col_results.items():
        if isinstance(res, dict) and "test" in res:
            if res["test"] == "ks_test":
                print(
                    f"  {col:<15} KS stat={res['statistic']:.4f}  "
                    f"p={res['p_value']:.4f}  passed={res['passed']}"
                )
            elif res["test"] == "tvd":
                print(f"  {col:<15} TVD={res['statistic']:.4f}  passed={res['passed']}")
        elif isinstance(res, dict) and "error" in res:
            print(f"  {col:<15} error: {res['error']}")

    # Correlation comparison
    corr_dist = validator.check_correlations(real_subset, synth_subset)
    print(f"\n  Correlation Frobenius norm: {corr_dist:.4f}")

    # Pass if validator didn't crash and we got results for at least some columns
    valid_results = [
        v for v in col_results.values() if isinstance(v, dict) and "test" in v
    ]
    passed_c = len(valid_results) >= 1
    results["Statistical Validation"] = passed_c
    print(f"\n  => {'PASS' if passed_c else 'FAIL'}")

except Exception as exc:
    print(f"  EXCEPTION: {exc}")
    traceback.print_exc()
    results["Statistical Validation"] = False

print()

# ─────────────────────────────────────────────────────────────
# TEST D — DataTransformer Edge Cases
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST D: DataTransformer Edge Cases")
print("=" * 70)
try:
    from syntho_hive.core.data.transformer import DataTransformer
    from syntho_hive.interface.config import Metadata as MetadataD

    edge_pass = True
    edge_details = []

    # D1: High cardinality (> 50 unique values -> should use embeddings)
    print("  D1: High cardinality column (>50 unique) ...")
    meta_d1 = MetadataD()
    meta_d1.add_table("edge1", pk="id")
    df_d1 = pd.DataFrame(
        {
            "id": range(200),
            "high_card": [f"cat_{i}" for i in range(200)],  # 200 unique values
            "value": np.random.randn(200),
        }
    )
    t_d1 = DataTransformer(meta_d1, embedding_threshold=50)
    t_d1.fit(df_d1, table_name="edge1")
    hc_info = t_d1._column_info.get("high_card", {})
    hc_type = hc_info.get("type", "unknown")
    d1_ok = hc_type == "categorical_embedding"
    edge_details.append(
        f"    type={hc_type} (expected categorical_embedding): {'OK' if d1_ok else 'FAIL'}"
    )
    if not d1_ok:
        edge_pass = False
    # Verify transform/inverse_transform round-trip
    transformed_d1 = t_d1.transform(df_d1)
    inv_d1 = t_d1.inverse_transform(transformed_d1)
    edge_details.append(
        f"    round-trip shape: {transformed_d1.shape} -> {inv_d1.shape}"
    )
    print("\n".join(edge_details))
    edge_details.clear()
    print()

    # D2: Column with only 1 unique value
    print("  D2: Single unique value column ...")
    meta_d2 = MetadataD()
    meta_d2.add_table("edge2", pk="id")
    df_d2 = pd.DataFrame(
        {
            "id": range(100),
            "constant": ["always_same"] * 100,
            "value": np.random.randn(100),
        }
    )
    t_d2 = DataTransformer(meta_d2, embedding_threshold=50)
    t_d2.fit(df_d2, table_name="edge2")
    transformed_d2 = t_d2.transform(df_d2)
    inv_d2 = t_d2.inverse_transform(transformed_d2)
    # The constant column should reconstruct to "always_same" everywhere
    d2_ok = (inv_d2["constant"].dropna() == "always_same").all()
    edge_details.append(f"    constant column preserved: {'OK' if d2_ok else 'FAIL'}")
    if not d2_ok:
        edge_pass = False
    print("\n".join(edge_details))
    edge_details.clear()
    print()

    # D3: All-null numeric column
    print("  D3: All-null numeric column ...")
    meta_d3 = MetadataD()
    meta_d3.add_table("edge3", pk="id")
    df_d3 = pd.DataFrame(
        {
            "id": range(50),
            "all_null": pd.array([np.nan] * 50, dtype="float64"),
            "value": np.random.randn(50),
        }
    )
    t_d3 = DataTransformer(meta_d3, embedding_threshold=50)
    t_d3.fit(df_d3, table_name="edge3")
    transformed_d3 = t_d3.transform(df_d3)
    inv_d3 = t_d3.inverse_transform(transformed_d3)
    d3_ok = inv_d3["all_null"].isna().all()
    edge_details.append(f"    all_null remains all NaN: {'OK' if d3_ok else 'FAIL'}")
    if not d3_ok:
        edge_pass = False
    print("\n".join(edge_details))
    edge_details.clear()
    print()

    # D4: Re-fitting transformer on different data (verifying C4 reset fix)
    print("  D4: Re-fit transformer on different data (reset fix) ...")
    meta_d4 = MetadataD()
    meta_d4.add_table("edge4", pk="id")
    df_d4a = pd.DataFrame(
        {
            "id": range(100),
            "color": np.random.choice(["red", "green", "blue"], 100),
            "score": np.random.randn(100),
        }
    )
    df_d4b = pd.DataFrame(
        {
            "id": range(80),
            "color": np.random.choice(["alpha", "beta", "gamma", "delta"], 80),
            "score": np.random.uniform(0, 1, 80),
        }
    )
    t_d4 = DataTransformer(meta_d4, embedding_threshold=50)

    # First fit
    t_d4.fit(df_d4a, table_name="edge4")
    out_dim_a = t_d4.output_dim
    transformed_a = t_d4.transform(df_d4a)

    # Re-fit on different data
    t_d4.fit(df_d4b, table_name="edge4")
    out_dim_b = t_d4.output_dim
    transformed_b = t_d4.transform(df_d4b)

    # After re-fit, transformer should work with new data and potentially different output_dim
    d4_ok = transformed_b.shape[0] == 80 and transformed_b.shape[1] == out_dim_b
    edge_details.append(
        f"    fit A: output_dim={out_dim_a}, shape={transformed_a.shape}"
    )
    edge_details.append(
        f"    fit B: output_dim={out_dim_b}, shape={transformed_b.shape}"
    )
    edge_details.append(f"    re-fit works correctly: {'OK' if d4_ok else 'FAIL'}")
    if not d4_ok:
        edge_pass = False
    print("\n".join(edge_details))
    edge_details.clear()

    results["Transformer Edge Cases"] = edge_pass
    print(f"\n  => {'PASS' if edge_pass else 'FAIL'}")

except Exception as exc:
    print(f"  EXCEPTION: {exc}")
    traceback.print_exc()
    results["Transformer Edge Cases"] = False

print()

# ─────────────────────────────────────────────────────────────
# TEST E — Seed Reproducibility
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST E: Seed Reproducibility")
print("=" * 70)
try:
    from syntho_hive.interface.config import Metadata as MetadataE
    from syntho_hive.core.models.ctgan import CTGAN as CTGAN_E

    # Use a small dataset for speed — no nulls to keep it simple
    repro_rng = np.random.RandomState(999)
    small_df = pd.DataFrame(
        {
            "pk": range(200),
            "val_a": repro_rng.randn(200),
            "cat_b": repro_rng.choice(["x", "y", "z"], 200),
        }
    )

    def train_and_sample(seed: int, label: str) -> pd.DataFrame:
        meta = MetadataE()
        meta.add_table("small", pk="pk")
        m = CTGAN_E(
            metadata=meta,
            batch_size=100,
            epochs=15,
            embedding_dim=32,
            generator_dim=(64, 64),
            discriminator_dim=(64, 64),
        )
        print(f"  Training model '{label}' with seed={seed} ...")
        m.fit(small_df, table_name="small", seed=seed, progress_bar=False)
        out = m.sample(50, seed=seed)
        return out

    # Same seed — should produce identical output
    out_1 = train_and_sample(12345, "A (seed=12345)")
    out_2 = train_and_sample(12345, "B (seed=12345)")

    same_seed_match = True
    for col in out_1.columns:
        if pd.api.types.is_numeric_dtype(out_1[col]):
            if not np.allclose(
                out_1[col].fillna(-999).values,
                out_2[col].fillna(-999).values,
                atol=1e-4,
            ):
                same_seed_match = False
                print(f"  MISMATCH in numeric column '{col}' for same seed")
                # Show first few values for debugging
                print(f"    A: {out_1[col].head(5).tolist()}")
                print(f"    B: {out_2[col].head(5).tolist()}")
        else:
            if not (
                out_1[col].fillna("__NULL__").values
                == out_2[col].fillna("__NULL__").values
            ).all():
                same_seed_match = False
                print(f"  MISMATCH in categorical column '{col}' for same seed")
                print(f"    A: {out_1[col].head(5).tolist()}")
                print(f"    B: {out_2[col].head(5).tolist()}")

    print(f"  Same seed produces identical output: {same_seed_match}")

    # Different seed — should produce different output
    out_3 = train_and_sample(99999, "C (seed=99999)")

    diff_seed_differs = False
    for col in out_1.columns:
        if pd.api.types.is_numeric_dtype(out_1[col]):
            if not np.allclose(
                out_1[col].fillna(-999).values,
                out_3[col].fillna(-999).values,
                atol=1e-4,
            ):
                diff_seed_differs = True
                break
        else:
            if not (
                out_1[col].fillna("__NULL__").values
                == out_3[col].fillna("__NULL__").values
            ).all():
                diff_seed_differs = True
                break

    print(f"  Different seed produces different output: {diff_seed_differs}")

    passed_e = same_seed_match and diff_seed_differs
    results["Seed Reproducibility"] = passed_e
    print(f"\n  => {'PASS' if passed_e else 'FAIL'}")

except Exception as exc:
    print(f"  EXCEPTION: {exc}")
    traceback.print_exc()
    results["Seed Reproducibility"] = False

print()

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("=== SYNTHOHIVE COMPREHENSIVE TEST REPORT ===")
print("=" * 70)

test_order = [
    "Privacy Sanitization",
    "CTGAN Training & Generation",
    "Statistical Validation",
    "Transformer Edge Cases",
    "Seed Reproducibility",
]

n_passed = 0
for t in test_order:
    status = results.get(t, False)
    label = "PASS" if status else "FAIL"
    n_passed += int(status)
    print(f"  [{label}] {t}")

print(f"\n=== {n_passed}/{len(test_order)} tests passed ===")
print("=" * 70)

sys.exit(0 if n_passed == len(test_order) else 1)
