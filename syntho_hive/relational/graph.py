from typing import List, Dict, Set
from syntho_hive.interface.config import Metadata, parse_fk_ref
from syntho_hive.exceptions import SchemaError


class SchemaGraph:
    """DAG representation of table dependencies derived from metadata."""

    def __init__(self, metadata: Metadata):
        """Create a dependency graph from table metadata.

        Args:
            metadata: Dataset metadata containing FK relationships.
        """
        self.metadata = metadata
        self.adj_list: Dict[str, Set[str]] = {}
        self._build_graph()

    def _build_graph(self):
        """Build an adjacency list from FK relationships."""
        for table_name in self.metadata.tables:
            self.adj_list[table_name] = set()

        for table_name, config in self.metadata.tables.items():
            for ref_col, ref_path in config.fk.items():
                parent_table, _ = parse_fk_ref(ref_path)
                if parent_table == table_name:
                    raise SchemaError(
                        f"Table '{table_name}' has a self-referencing FK "
                        f"'{ref_col}' -> '{ref_path}', which is not supported."
                    )
                if parent_table not in self.adj_list:
                    raise SchemaError(
                        f"Table '{table_name}' references unknown parent table "
                        f"'{parent_table}' via FK '{ref_col}'."
                    )
                # Dependency: Parent -> Child (we generate Parent first)
                self.adj_list[parent_table].add(table_name)

    def get_generation_order(self) -> List[str]:
        """Return a topologically sorted list of tables.

        Returns:
            List of table names ordered for parent-before-child generation.

        Raises:
            SchemaError: If a cycle is detected in FK relationships.
        """
        visited = set()
        stack = []
        path = set()

        def visit(node):
            if node in path:
                raise SchemaError(f"Cycle detected in FK relationships involving '{node}'")
            if node in visited:
                return

            path.add(node)
            visited.add(node)

            # Note: For generation order (Parent -> Child), we want to visit parents, then children.
            # Standard topological sort gives reverse dependency order if edge is Dependency -> Dependent
            # Here Edge is Parent -> Child. So generic topological sort:
            # Visit Parent, allow it to finish, add to stack? No.
            # If A -> B (A is parent of B).
            # We want [A, B].
            # Normal DFS topo sort on A -> B puts B on stack, then A. Stack: [A, B] (LIFO) -> Pop A, Pop B.
            # Yes, standard topological sort on (Parent -> Child) edges returns [Parent, Child].

            for neighbor in self.adj_list.get(node, []):
                visit(neighbor)

            path.remove(node)
            stack.append(node)

        # Iterate over all nodes, not just roots, to catch disconnected components
        # Sort keys for deterministic order
        for node in sorted(self.adj_list.keys()):
            visit(node)

        return stack[::-1]  # Reverse stack to get topological order
