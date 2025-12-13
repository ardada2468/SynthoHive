from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from syntho_hive.interface.config import Metadata
from syntho_hive.core.models.ctgan import CTGAN
from syntho_hive.relational.linkage import LinkageModel


def make_parent_child_seed_data(num_parents: int = 300, max_children: int = 5):
    """Create a small parent/child dataset to train linkage + CTGAN."""
    rng = np.random.default_rng(10)
    regions = ["NE", "SE", "MW", "W"]

    parents = pd.DataFrame(
        {
            "user_id": np.arange(1, num_parents + 1),
            "region": rng.choice(regions, size=num_parents, p=[0.3, 0.2, 0.25, 0.25]),
            "age": rng.integers(20, 70, size=num_parents),
        }
    )

    child_rows = []
    order_id = 1
    for _, row in parents.iterrows():
        n_orders = rng.integers(0, max_children + 1)
        for _ in range(n_orders):
            child_rows.append(
                {
                    "order_id": order_id,
                    "user_id": row["user_id"],
                    "basket_value": max(5, rng.normal(80, 25)),
                    "channel": rng.choice(["web", "store", "mobile"], p=[0.5, 0.3, 0.2]),
                }
            )
            order_id += 1

    children = pd.DataFrame(child_rows)
    return parents, children


def build_metadata() -> Metadata:
    meta = Metadata()
    meta.add_table(name="users", pk="user_id", pii_cols=[], high_cardinality_cols=["region"])
    meta.add_table(
        name="orders",
        pk="order_id",
        fk={"user_id": "users.user_id"},
        parent_context_cols=["region"],
        constraints={"basket_value": {"dtype": "float", "min": 1.0}},
    )
    return meta


def train_models(meta: Metadata, parents: pd.DataFrame, children: pd.DataFrame, epochs: int) -> tuple[CTGAN, CTGAN, LinkageModel]:
    """Train CTGAN for parents, linkage + conditional CTGAN for children."""
    users_model = CTGAN(meta, batch_size=128, epochs=epochs, generator_dim=(128, 128), discriminator_dim=(128, 128), embedding_dim=64)
    print("Training users CTGAN...")
    users_model.fit(parents, table_name="users")

    linkage = LinkageModel()
    print("Training linkage model...")
    linkage.fit(parents, children, fk_col="user_id", pk_col="user_id")

    # Build context dataframe for child training
    joined = children.merge(parents[["user_id", "region"]], on="user_id", how="left")
    context_df = joined[["region"]].copy()

    orders_model = CTGAN(meta, batch_size=128, epochs=epochs, generator_dim=(128, 128), discriminator_dim=(128, 128), embedding_dim=64)
    print("Training orders CTGAN with parent context...")
    orders_model.fit(children, context=context_df, table_name="orders")

    return users_model, orders_model, linkage


def generate(meta: Metadata, users_model: CTGAN, orders_model: CTGAN, linkage: LinkageModel, num_parents: int, output_dir: Path):
    print(f"Generating {num_parents} synthetic parents...")
    users = users_model.sample(num_parents)
    users.insert(0, "user_id", range(1, len(users) + 1))

    counts = linkage.sample_counts(users)
    total_children = int(counts.sum())
    print(f"Generating {total_children} synthetic children conditioned on parents...")

    # Build repeated context rows for each parent
    context_rows = []
    fk_values = []
    for idx, parent in users.iterrows():
        repeat = counts[idx]
        if repeat <= 0:
            continue
        fk_values.extend([parent["user_id"]] * repeat)
        context_rows.extend([{"region": parent["region"]}] * repeat)

    if total_children > 0:
        context_df = pd.DataFrame(context_rows)
        orders = orders_model.sample(total_children, context=context_df)
        orders.insert(0, "order_id", range(1, len(orders) + 1))
        orders["user_id"] = fk_values
    else:
        orders = pd.DataFrame(columns=["order_id", "user_id", "basket_value", "channel", "region"])

    users_path = output_dir / "users.csv"
    orders_path = output_dir / "orders.csv"

    users.to_csv(users_path, index=False)
    orders.to_csv(orders_path, index=False)

    print(f"Wrote {len(users)} users to {users_path}")
    print(f"Wrote {len(orders)} orders to {orders_path}")
    print(orders.head())


def main():
    parser = argparse.ArgumentParser(description="Relational generation demo without Spark.")
    parser.add_argument("--parents", type=int, default=150, help="Number of synthetic parents to generate.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for both GANs.")
    parser.add_argument(
        "--output-dir",
        default="examples/demos/04_relational_linkage_ctgan/outputs",
        help="Directory to place generated CSVs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = build_metadata()
    parent_df, child_df = make_parent_child_seed_data()
    users_model, orders_model, linkage = train_models(meta, parent_df, child_df, epochs=args.epochs)

    generate(meta, users_model, orders_model, linkage, num_parents=args.parents, output_dir=output_dir)


if __name__ == "__main__":
    main()

