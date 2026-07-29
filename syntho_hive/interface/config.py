from typing import List, Dict, Optional, Tuple, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import numpy as np
import pandas as pd
from syntho_hive.exceptions import SchemaError, SchemaValidationError


def parse_fk_ref(ref: str) -> Tuple[str, str]:
    """Parse a foreign-key reference of the form ``'parent_table.parent_col'``.

    Args:
        ref: FK reference string.

    Raises:
        SchemaError: If the reference is not exactly ``table.column``.

    Returns:
        Tuple of ``(parent_table, parent_col)``.
    """
    parts = ref.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SchemaError(
            f"Invalid FK reference '{ref}'. Expected format 'parent_table.parent_col'."
        )
    return parts[0], parts[1]


class PrivacyConfig(BaseModel):
    """Configuration for privacy guardrails applied during synthesis."""

    model_config = ConfigDict(extra="forbid")

    enable_differential_privacy: bool = False
    epsilon: float = 1.0
    pii_strategy: Literal["mask", "faker", "context_aware_faker"] = (
        "context_aware_faker"
    )
    k_anonymity_threshold: int = 5
    pii_columns: List[str] = Field(default_factory=list)

    @field_validator("epsilon")
    @classmethod
    def validate_epsilon(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("epsilon must be positive")
        return v

    @field_validator("k_anonymity_threshold")
    @classmethod
    def validate_k_anonymity(cls, v: int) -> int:
        if v < 1:
            raise ValueError("k_anonymity_threshold must be >= 1")
        return v


class Constraint(BaseModel):
    """Configuration object describing numeric constraints for a column."""

    model_config = ConfigDict(extra="forbid")

    dtype: Optional[Literal["int", "float"]] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "Constraint":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"Constraint min ({self.min}) must be <= max ({self.max})")
        return self


class TableConfig(BaseModel):
    """Configuration for a single table, including keys and constraints."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pk: str
    driver_fk: Optional[str] = Field(
        default=None,
        description=(
            "FK column that selects the 'driver' parent for cardinality modeling "
            "and conditional context. Defaults to the alphabetically-first FK."
        ),
    )
    pii_cols: List[str] = Field(default_factory=list)
    high_cardinality_cols: List[str] = Field(default_factory=list)
    fk: Dict[str, str] = Field(
        default_factory=dict, description="Map of local_col -> parent_table.parent_col"
    )
    parent_context_cols: List[str] = Field(
        default_factory=list,
        description="List of parent attributes to condition on (e.g., 'users.region')",
    )
    constraints: Dict[str, Constraint] = Field(
        default_factory=dict, description="Map of col_name -> Constraint"
    )
    linkage_method: Literal["empirical", "negbinom"] = "empirical"

    @model_validator(mode="after")
    def validate_driver_fk(self) -> "TableConfig":
        if self.driver_fk is not None and self.driver_fk not in self.fk:
            raise ValueError(
                f"driver_fk '{self.driver_fk}' is not one of the declared FK columns "
                f"{sorted(self.fk)}"
            )
        return self

    @property
    def has_dependencies(self) -> bool:
        """Whether the table declares any foreign key dependencies."""
        return bool(self.fk)

    def get_driver_fk(self) -> str:
        """Return the FK column driving cardinality/context (explicit or sorted-first)."""
        if not self.fk:
            raise SchemaError(f"Table '{self.name}' has no FK dependencies")
        return self.driver_fk or sorted(self.fk)[0]


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
    if kind_a == kind_b:
        # Identical kinds (incl. datetime 'M', bool 'b') are always compatible.
        return True
    numeric_kinds = {"i", "u", "f"}
    string_kinds = {"U", "O", "S"}
    if kind_a in numeric_kinds and kind_b in numeric_kinds:
        return True
    if kind_a in string_kinds and kind_b in string_kinds:
        return True
    return False


class Metadata(BaseModel):
    """Schema definition for the entire dataset."""

    model_config = ConfigDict(extra="forbid")

    tables: Dict[str, TableConfig] = Field(default_factory=dict)

    def add_table(
        self,
        name: str,
        pk: str,
        **kwargs: Union[List[str], Dict[str, str], Dict[str, Constraint]],
    ):
        """Register a table configuration.

        Args:
            name: Table name.
            pk: Primary key column name.
            **kwargs: Additional fields to populate ``TableConfig``.

        Raises:
            SchemaError: If a table with the same name already exists.
        """
        if name in self.tables:
            raise SchemaError(f"Table '{name}' already exists in metadata.")
        self.tables[name] = TableConfig(name=name, pk=pk, **kwargs)

    def get_table(self, name: str) -> Optional[TableConfig]:
        """Fetch a table configuration by name.

        Args:
            name: Table name to retrieve.

        Returns:
            Corresponding ``TableConfig`` or ``None`` if missing.
        """
        return self.tables.get(name)

    def validate_schema(
        self, real_data: Optional[Dict[str, "pd.DataFrame"]] = None
    ) -> None:
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
                try:
                    parent_table, parent_col = parse_fk_ref(parent_ref)
                except SchemaError:
                    errors.append(
                        f"Invalid FK reference '{parent_ref}' in table '{table_name}'."
                        f" Format should be 'parent_table.parent_col'."
                    )
                    continue

                if parent_table == table_name:
                    errors.append(
                        f"Table '{table_name}' has a self-referencing FK "
                        f"'{local_col}' -> '{parent_ref}'. Self-references are not "
                        f"supported by the relational synthesis pipeline."
                    )
                    continue

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

        # PK existence check when data is provided.
        if real_data is not None:
            for table_name, table_config in self.tables.items():
                if table_name in real_data:
                    if table_config.pk not in real_data[table_name].columns:
                        errors.append(
                            f"Declared PK column '{table_config.pk}' missing from "
                            f"table '{table_name}'."
                        )

        # Cycle detection — a cyclic FK graph would otherwise only fail at
        # generation time, after all models have already been trained.
        if not errors:
            cycle = self._find_cycle()
            if cycle:
                errors.append(
                    f"FK relationships form a cycle: {' -> '.join(cycle)}. "
                    f"Relational synthesis requires an acyclic schema."
                )

        if errors:
            raise SchemaValidationError("\n".join(errors))

    def _find_cycle(self) -> Optional[List[str]]:
        """Return one FK cycle as a table-name path, or None if the graph is acyclic."""
        children: Dict[str, List[str]] = {name: [] for name in self.tables}
        for table_name, config in self.tables.items():
            for parent_ref in config.fk.values():
                parent_table = parent_ref.split(".")[0]
                if parent_table in children and parent_table != table_name:
                    children[parent_table].append(table_name)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self.tables}

        def dfs(node: str, trail: List[str]) -> Optional[List[str]]:
            color[node] = GRAY
            trail.append(node)
            for child in children[node]:
                if color[child] == GRAY:
                    return trail[trail.index(child) :] + [child]
                if color[child] == WHITE:
                    found = dfs(child, trail)
                    if found:
                        return found
            trail.pop()
            color[node] = BLACK
            return None

        for name in sorted(self.tables):
            if color[name] == WHITE:
                found = dfs(name, [])
                if found:
                    return found
        return None
