import pytest
import pandas as pd
import numpy as np
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.interface.config import Metadata
import torch
import shutil
import tempfile
import os

def test_ctgan_full_cycle():
    """Test CTGAN fit, sample, save, and load lifecycle."""
    # 1. Setup Data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.normal(30, 5, 100),
        'income': np.random.exponential(50000, 100),
        'city': np.random.choice(['NY', 'SF', 'LA'], 100)
    })
    
    # 2. Setup Metadata
    meta = Metadata()
    meta.add_table("users", "id")
    
    # 3. Init Model
    # Use small dims for speed
    model = CTGAN(
        metadata=meta,
        batch_size=20,
        epochs=2,
        embedding_dim=16,
        generator_dim=(32, 32),
        discriminator_dim=(32, 32)
    )
    
    # 4. Fit
    model.fit(data, table_name="users")
    
    assert model.generator is not None
    assert model.discriminator is not None
    assert model.transformer.output_dim > 0
    
    # 5. Sample
    n_samples = 50
    synthetic_data = model.sample(n_samples)
    
    assert len(synthetic_data) == n_samples
    assert list(synthetic_data.columns) == list(data.columns)
    
    # Check simple statistical property (mean should be vaguely close even after 2 epochs, just not crashing)
    # Just checking it's not all zeros
    assert synthetic_data['income'].mean() != 0
    
    # 6. Save and Load
    # save() now writes a directory checkpoint (7-file bundle) instead of a single .pt file
    tmp_path = "test_ctgan_model.pth"
    try:
        model.save(tmp_path)
        assert os.path.isdir(tmp_path), "save() should produce a directory checkpoint"

        # Cold load — fresh instance, no prior fit needed
        new_model = CTGAN(
            metadata=meta,
            embedding_dim=16,
            generator_dim=(32, 32),
            discriminator_dim=(32, 32)
        )
        new_model.load(tmp_path)

        sample_loaded = new_model.sample(10)
        assert len(sample_loaded) == 10

    finally:
        # save() writes a directory now, so use shutil.rmtree instead of os.remove
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)
        elif os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_ctgan_conditional():
    """Test CTGAN with parent context."""
    # 1. Setup Data
    # Child: Orders
    orders = pd.DataFrame({
        'amount': np.random.normal(100, 20, 100)
    })
    
    # Context: User Income (Parent)
    context = pd.DataFrame({
        'user_income': np.random.normal(50000, 10000, 100)
    })
    
    meta = Metadata()
    meta.add_table("orders", "id")
    
    model = CTGAN(metadata=meta, epochs=1, batch_size=20)
    
    # 2. Fit with context
    model.fit(orders, context=context, table_name="orders")
    
    # 3. Sample with context
    # Need new context for sampling
    new_context = pd.DataFrame({
        'user_income': [60000, 40000]
    })
    
    synthetic_orders = model.sample(2, context=new_context)
    
    assert len(synthetic_orders) == 2
    assert 'amount' in synthetic_orders.columns

def test_multi_table_generation_print(capsys):
    """
    Simulate a Users -> Transactions scenario.
    Train on Real Data.
    Generate Synthetic Data.
    Print 5 rows of each.
    """
    np.random.seed(42)
    
    # --- 1. Create Real Data ---
    num_users = 100
    
    # Parent Table: Users
    users = pd.DataFrame({
        'user_id': range(num_users),
        'income': np.random.choice([20000, 50000, 100000], num_users),
        'region': np.random.choice(['US', 'EU', 'APAC'], num_users)
    })
    
    # Child Table: Transactions
    # Logic: High income -> Higher transaction amounts
    records = []
    for _ in range(500): # 500 transactions
        user = users.sample(1).iloc[0]
        base_amt = 100 if user['income'] > 50000 else 20
        amount = np.random.normal(base_amt, 10)
        records.append({
            'transaction_id': len(records),
            'user_id': user['user_id'],
            'amount': abs(amount),
            'currency': 'USD' if user['region'] == 'US' else 'EUR'
        })
    transactions = pd.DataFrame(records)
    
    # --- 2. Prepare Training Data ---
    # To train the child model, we need the Child Data AND the corresponding Parent Context for each row
    merged = transactions.merge(users, on='user_id')
    
    target_data = merged[['amount', 'currency']]
    context_data = merged[['income', 'region']] # Context used to condition the generation
    
    # Metadata setup
    meta = Metadata()
    meta.add_table("transactions", "transaction_id", fk={'user_id': 'users.user_id'})
    
    # --- 3. Train Model ---
    # Using more epochs to ideally see some pattern, though 5 is still low for convergence
    model = CTGAN(
        metadata=meta,
        batch_size=50,
        epochs=5,
        embedding_dim=16,
        generator_dim=(64, 64),
        discriminator_dim=(64, 64)
    )
    
    print("\nTraining Model...")
    model.fit(target_data, context=context_data, table_name="transactions")
    
    # --- 4. Generate Synthetic Data ---
    # We want to generate transactions for specific users (Context).
    # Let's pick 5 random users from our user base to generate transactions for.
    sample_users = users.sample(5)
    sample_context = sample_users[['income', 'region']]
    
    print("\nGenerating Synthetic Data based on sampled User Context...")
    synthetic_data = model.sample(5, context=sample_context)
    
    # Add the user info back for display clarity
    real_display = transactions[['amount', 'currency']].head(5)
    syn_display = synthetic_data.copy()
    
    # --- 5. Print Results ---
    with capsys.disabled():
        print("\n" + "="*50)
        print("REAL DATA (First 5 rows of Transactions)")
        print("="*50)
        print(real_display.to_string(index=False))
        print("\n" + "="*50)
        print("GENERATED DATA (Conditioned on 5 Random Users)")
        print("="*50)
        print(syn_display.to_string(index=False))
        print("="*50 + "\n")
        
    assert len(synthetic_data) == 5

