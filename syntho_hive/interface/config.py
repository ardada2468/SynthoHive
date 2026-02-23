from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from syntho_hive.exceptions import SchemaValidationError

class PrivacyConfig(BaseModel):
    """Configuration for privacy guardrails applied during synthesis."""
    enable_differential_privacy: bool = False
    epsilon: float = 1.0
    pii_strategy: Literal["mask", "faker", "context_aware_faker"] = "context_aware_faker"
    k_anonymity_threshold: int = 5
    pii_columns: List[str] = Field(default_factory=list)

class Constraint(BaseModel):
    """Configuration object describing numeric constraints for a column."""
    dtype: Optional[Literal["int", "float"]] = None
    min: Optional[float] = None
    max: Optional[float] = None

class TableConfig(BaseModel):
    """Configuration for a single table, including keys and constraints."""
    name: str
    pk: str
    pii_cols: List[str] = Field(default_factory=list)
    high_cardinality_cols: List[str] = Field(default_factory=list)
    fk: Dict[str, str] = Field(default_factory=dict, description="Map of local_col -> parent_table.parent_col")
    parent_context_cols: List[str] = Field(default_factory=list, description="List of parent attributes to condition on (e.g., 'users.region')")
    constraints: Dict[str, Constraint] = Field(default_factory=dict, description="Map of col_name -> Constraint")
    linkage_method: Literal["empirical", "negbinom"] = "empirical"

    @property
    def has_dependencies(self) -> bool:
        """Whether the table declares any foreign key dependencies."""
        return bool(self.fk)


def _dtypes_compatible(dtype_a: str, dtype_b: str) -> bool:
    """Return True if both dtypes belong to the same broad category (integer or string/object).

    Uses numpy kind codes:
      - 'i' / 'u' : signed / unsigned integer
      - 'f'       : floating-point
      - 'U' / 'O' / 'S' : unicode / object / byte-string

    Pandas extension types (e.g. StringDtype, Int64Dtype) produce a TypeError
    when passed to np.dtype(); those are treated as compatible (True) to avoid
    false positives.
    """
    try:
        kind_a = np.dtype(dtype_a).kind
        kind_b = np.dtype(dtype_b).kind
    except TypeError:
        # pandas extension types — be conservative and assume compatible.
        return True
    integer_kinds = {'i', 'u'}
    string_kinds = {'U', 'O', 'S'}
    if kind_a in integer_kinds and kind_b in integer_kinds:
        return True
    if kind_a in string_kinds and kind_b in string_kinds:
        return True
    if kind_a == 'f' and kind_b == 'f':
        return True
    return False


class Metadata(BaseModel):
    """Schema definition for the entire dataset."""
    tables: Dict[str, TableConfig] = Field(default_factory=dict)

    def add_table(self, name: str, pk: str, **kwargs: Union[List[str], Dict[str, str], Dict[str, Constraint]]):
        """Register a table configuration.

        Args:
            name: Table name.
            pk: Primary key column name.
            **kwargs: Additional fields to populate ``TableConfig``.

        Raises:
            ValueError: If a table with the same name already exists.
        """
        if name in self.tables:
             raise ValueError(f"Table '{name}' already exists in metadata.")
        self.tables[name] = TableConfig(name=name, pk=pk, **kwargs)

    def get_table(self, name: str) -> Optional[TableConfig]:
        """Fetch a table configuration by name.

        Args:
            name: Table name to retrieve.

        Returns:
            Corresponding ``TableConfig`` or ``None`` if missing.
        """
        return self.tables.get(name)

    def validate_schema(self, real_data: Optional[Dict[str, "pd.DataFrame"]] = None) -> None:
        """Validate schema integrity, focusing on foreign key references.

        Collects all errors before raising so callers see the complete problem
        list in a single exception.

        Args:
            real_data: Optional mapping of table name to DataFrame. When provided,
                FK type compatibility and column existence checks are performed in
                addition to structural (table-existence, FK-format) checks.

        Raises:
            SchemaValidationError: When one or more FK references are malformed,
                target a missing table, have type mismatches, or reference missing
                columns. The exception message lists all detected problems.
        """
        errors: List[str] = []

        for table_name, table_config in self.tables.items():
            for local_col, parent_ref in table_config.fk.items():
                if "." not in parent_ref:
                    errors.append(
                        f"Invalid FK reference '{parent_ref}' in table '{table_name}'."
                        f" Format should be 'parent_table.parent_col'."
                    )
                    continue

                parent_table, parent_col = parent_ref.split(".", 1)

                if parent_table not in self.tables:
                    errors.append(
                        f"Table '{table_name}' references non-existent parent table '{parent_table}'."
                    )
                    continue

                # Optional: data-level type and column checks.
                if real_data is not None:
                    if table_name not in real_data or parent_table not in real_data:
                        # Skip type check when data is only partially provided.
                        continue

                    child_df = real_data[table_name]
                    parent_df = real_data[parent_table]

                    if local_col not in child_df.columns:
                        errors.append(
                            f"FK column '{local_col}' missing from table '{table_name}'."
                            f" Add column '{local_col}' to child table '{table_name}'."
                        )
                    elif parent_col not in parent_df.columns:
                        errors.append(
                            f"Parent PK column '{parent_col}' missing from table '{parent_table}'."
                        )
                    else:
                        child_dtype = str(child_df[local_col].dtype)
                        parent_dtype = str(parent_df[parent_col].dtype)
                        if not _dtypes_compatible(child_dtype, parent_dtype):
                            errors.append(
                                f"FK type mismatch: '{table_name}.{local_col}' is {child_dtype}"
                                f" but '{parent_table}.{parent_col}' is {parent_dtype}."
                                f" Fix: cast '{table_name}.{local_col}' to {parent_dtype}"
                                f" or cast '{parent_table}.{parent_col}' to {child_dtype}."
                            )

        if errors:
            raise SchemaValidationError("\n".join(errors))
