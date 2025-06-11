from typing import List, Tuple


# Represents a single packet classification rule
class Rule:
    def __init__(self, priority: int, ranges: List[Tuple[int, int]]):
        self.priority = priority
        self.ranges = ranges

    def __repr__(self) -> str:
        return f"Rule(priority={self.priority}, ranges={self.ranges})"

# Represents a node in the classification tree
class Node:
    def __init__(self, depth: int, rules: List[Rule], boundary: Rule, children: List['Node']):
        self.depth = depth
        self.rules = rules
        self.boundary = boundary
        self.children = children

    def __repr__(self) -> str:
        return f"Node(depth={self.depth}, rules={len(self.rules)}, children={len(self.children)})"

# Represents a TCAM node in the classification tree
class TCAMNode(Node):
    # Represents the memory cost of a TCAM entry, mirroring the MAT_SIZE
    # constant from the original C implementation's fine-grained model.
    # This value is a heuristic.
    MAT_SIZE = 1024

    def __init__(self, depth: int, rules: List[Rule], boundary: Rule, children: List['Node'], cut_dim: int = -1, cut_points: List[int] = None):
        super().__init__(depth, rules, boundary, children)
        self.cut_dim = cut_dim
        self.cut_points = cut_points if cut_points is not None else []

    def __repr__(self) -> str:
        return f"TCAMNode(depth={self.depth}, rules={len(self.rules)}, children={len(self.children)}, cut_dim={self.cut_dim})"