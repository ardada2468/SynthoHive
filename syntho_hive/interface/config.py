from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field

class PrivacyConfig(BaseModel):
    """Configuration for privacy guardrails."""
    enable_differential_privacy: bool = False
    epsilon: float = 1.0
    pii_strategy: Literal["mask", "faker", "context_aware_faker"] = "context_aware_faker"
    k_anonymity_threshold: int = 5
    pii_columns: List[str] = Field(default_factory=list)

class Constraint(BaseModel):
    """Configuration for column constraints."""
    dtype: Optional[Literal["int", "float"]] = None
    min: Optional[float] = None
    max: Optional[float] = None

class TableConfig(BaseModel):
    """Configuration for a single table."""
    name: str
    pk: str
    pii_cols: List[str] = Field(default_factory=list)
    high_cardinality_cols: List[str] = Field(default_factory=list)
    fk: Dict[str, str] = Field(default_factory=dict, description="Map of local_col -> parent_table.parent_col")
    parent_context_cols: List[str] = Field(default_factory=list, description="List of parent attributes to condition on (e.g., 'users.region')")
    constraints: Dict[str, Constraint] = Field(default_factory=dict, description="Map of col_name -> Constraint")

    @property
    def has_dependencies(self) -> bool:
        return bool(self.fk)

class Metadata(BaseModel):
    """Schema definition for the entire dataset."""
    tables: Dict[str, TableConfig] = Field(default_factory=dict)

    def add_table(self, name: str, pk: str, **kwargs):
        if name in self.tables:
             raise ValueError(f"Table '{name}' already exists in metadata.")
        self.tables[name] = TableConfig(name=name, pk=pk, **kwargs)

    def get_table(self, name: str) -> Optional[TableConfig]:
        return self.tables.get(name)

    def validate_schema(self):
        """Validates the schema integrity specifically Foreign Keys."""
        for table_name, table_config in self.tables.items():
            for local_col, parent_ref in table_config.fk.items():
                if "." not in parent_ref:
                    raise ValueError(f"Invalid FK reference '{parent_ref}' in table '{table_name}'. Format should be 'parent_table.parent_col'.")
                
                parent_table, parent_col = parent_ref.split(".", 1)
                
                if parent_table not in self.tables:
                    raise ValueError(f"Table '{table_name}' references non-existent parent table '{parent_table}'.")
                
                # Ideally we check if parent_col exists in parent config, but we don't store column lists explicitly in TableConfig yet (except PII/Card/PK).
                # Start with just checking table existence.
