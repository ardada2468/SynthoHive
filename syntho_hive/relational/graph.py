from typing import List, Dict, Set
from syntho_hive.interface.config import Metadata

class SchemaGraph:
    """
    DAG representation of table dependencies.
    """
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
        self.adj_list: Dict[str, Set[str]] = {}
        self._build_graph()
        
    def _build_graph(self):
        """Build adjacency list from FKs."""
        for table_name in self.metadata.tables:
            self.adj_list[table_name] = set()
            
        for table_name, config in self.metadata.tables.items():
            for ref_col, ref_path in config.fk.items():
                parent_table, _ = ref_path.split(".")
                # Dependency: Parent -> Child (we generate Parent first)
                if parent_table in self.adj_list:
                    self.adj_list[parent_table].add(table_name)
                    
    def get_generation_order(self) -> List[str]:
        """
        Return topological sort of tables.
        """
        visited = set()
        stack = []
        path = set()
        
        def visit(node):
            if node in path:
                raise ValueError(f"Cycle detected involving {node}")
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
            
        return stack[::-1] # Reverse stack to get topological order
