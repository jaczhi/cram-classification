import re
from typing import List
from cramcuts.structures import Node, Rule, TCAMNode


def ip_to_range(ip_str: str) -> tuple[int, int]:
    """
    Converts a CIDR IP address string into a tuple representing the integer range.
    This is a port of the `IP2Range` function from `src-efficuts/cutio.c`.
    """
    match = re.match(r'(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})/(\d{1,2})', ip_str)
    if not match:
        raise ValueError(f"Invalid IP address format: {ip_str}")

    parts = [int(p) for p in match.groups()]
    ip_int = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]
    prefix_len = parts[4]

    if not (0 <= prefix_len <= 32):
        raise ValueError(f"Invalid prefix length: {prefix_len}")

    mask = (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF
    low = ip_int & mask
    high = low | (~mask & 0xFFFFFFFF)
    return low, high


def parse_rule(rule_str: str, priority: int) -> 'Rule':
    """
    Parses a single rule string from a ClassBench file.
    This is a port of the `loadrule` function from `src-efficuts/cutio.c`.
    """
    parts = rule_str.strip().split('\t')
    
    # Source IP
    src_ip_str = parts[0][1:]  # remove @
    src_ip_low, src_ip_high = ip_to_range(src_ip_str)
    
    # Destination IP
    dest_ip_str = parts[1]
    dest_ip_low, dest_ip_high = ip_to_range(dest_ip_str)
    
    # Source Port
    src_port_low, src_port_high = [int(p) for p in parts[2].split(' : ')]
    
    # Destination Port
    dest_port_low, dest_port_high = [int(p) for p in parts[3].split(' : ')]
    
    # Protocol
    proto_parts = parts[4].split('/')
    proto_val = int(proto_parts[0], 16)
    proto_mask = int(proto_parts[1], 16)
    
    if proto_mask == 0xFF:
        proto_low, proto_high = proto_val, proto_val
    else:
        proto_low, proto_high = 0, 0xFF

    ranges = [
        (src_ip_low, src_ip_high),
        (dest_ip_low, dest_ip_high),
        (src_port_low, src_port_high),
        (dest_port_low, dest_port_high),
        (proto_low, proto_high),
    ]
    return Rule(priority=priority, ranges=ranges)


def load_rules(filename: str) -> List['Rule']:
    """
    Loads rules from a ClassBench file.
    """
    rules = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():
                rules.append(parse_rule(line, i))
    return rules
def is_present(node_boundary: 'Rule', rule: 'Rule') -> bool:
    """
    Checks if a given rule intersects with a node's boundary.
    This is a translation of the `is_present` function from `src-efficuts/checks.c`.
    """
    for i in range(len(rule.ranges)):
        rule_low, rule_high = rule.ranges[i]
        boundary_low, boundary_high = node_boundary.ranges[i]

        # Check for intersection
        if not (
            (rule_low <= boundary_low and rule_high >= boundary_low) or
            (rule_high >= boundary_high and rule_low <= boundary_high) or
            (rule_low >= boundary_low and rule_high <= boundary_high)
        ):
            return False
    return True


def do_rules_intersect(rule1: 'Rule', rule2: 'Rule') -> bool:
    """
    Checks if two rules overlap.
    This is a translation of the `DoRulesIntersect` function from `src-efficuts/checks.c`.
    """
    for i in range(len(rule1.ranges)):
        r1_low, r1_high = rule1.ranges[i]
        r2_low, r2_high = rule2.ranges[i]

        if r1_high < r2_low or r1_low > r2_high:
            return False
    return True
# Define memory constants based on the C implementation's assumptions for 64-bit architecture
PTR_SIZE = 8  # 8 bytes for a pointer
LEAF_NODE_SIZE = 8  # Simplified size for a leaf node structure
INTERNAL_NODE_SIZE = 48  # Simplified size for an internal node structure


class TreeStat:
    """A data class to hold statistics for a single tree."""
    def __init__(self, tree_id: int, num_rules: int):
        self.id = tree_id
        self.num_rules = num_rules
        self.node_count = 0
        self.leaf_node_count = 0
        self.tcam_node_count = 0
        self.total_rule_size = 0
        self.total_array_size = 0
        self.max_depth = 0
        self.standard_node_memory = 0
        self.tcam_node_memory = 0
        self.total_memory = 0


def _collect_stats_recursive(node: 'Node', stat: 'TreeStat', depth: int):
    """Recursively traverses the tree to collect statistics."""
    stat.node_count += 1
    stat.max_depth = max(stat.max_depth, depth)
    if isinstance(node, TCAMNode):
        stat.tcam_node_count += 1
        # A TCAM node is an internal node, not a leaf.
        # It has its own memory cost, but it also has children that need to be traversed.
        stat.total_array_size += len(node.children)
        for child in node.children:
            _collect_stats_recursive(child, stat, depth + 1)
    elif not node.children:  # It's a standard leaf node
        stat.leaf_node_count += 1
        stat.total_rule_size += len(node.rules)
    else:  # It's a standard internal node
        stat.total_array_size += len(node.children)
        for child in node.children:
            _collect_stats_recursive(child, stat, depth + 1)


def print_stats(trees: List['Node'], overall_depth: int):
    """
    Calculates and prints statistics for a list of decision trees,
    replicating the output of `PrintStats` from `cutio.c`.
    """
    print(f"Number of trees after merge: {len(trees)}")

    overall_total_mem = 0
    overall_std_mem = 0
    overall_tcam_mem = 0

    for i, tree_root in enumerate(trees):
        stat = TreeStat(tree_id=i, num_rules=len(tree_root.rules))
        _collect_stats_recursive(tree_root, stat, 1)

        # Memory for rule pointers in leaf nodes
        ruleptr_memory = PTR_SIZE * stat.total_rule_size
        # Memory for children pointers in internal nodes
        array_memory = PTR_SIZE * stat.total_array_size
        
        # Memory for the node structures themselves
        # Memory for the node structures themselves
        # TCAM nodes are counted separately, so standard internal nodes are what's left.
        internal_node_count = stat.node_count - stat.leaf_node_count - stat.tcam_node_count
        standard_leaf_count = stat.leaf_node_count
        
        stat.standard_node_memory = (
            (LEAF_NODE_SIZE * standard_leaf_count) +
            (INTERNAL_NODE_SIZE * internal_node_count) +
            array_memory +
            ruleptr_memory
        )
        
        # Calculate TCAM memory separately
        stat.tcam_node_memory = TCAMNode.MAT_SIZE * stat.tcam_node_count
        
        stat.total_memory = stat.standard_node_memory + stat.tcam_node_memory

        print(f"  Tree {i}:")
        print(f"    Standard Node Memory: {stat.standard_node_memory / 1024:.2f} KB")
        print(f"    TCAM Node Memory:     {stat.tcam_node_memory / 1024:.2f} KB")
        print(f"    Total Memory:         {stat.total_memory / 1024:.2f} KB")
        
        overall_total_mem += stat.total_memory
        overall_std_mem += stat.standard_node_memory
        overall_tcam_mem += stat.tcam_node_memory

    print("\n  Overall:")
    print(f"    Standard Node Memory: {overall_std_mem / 1024:.2f} KB")
    print(f"    TCAM Node Memory:     {overall_tcam_mem / 1024:.2f} KB")
    print(f"    Total Memory:         {overall_total_mem / 1024:.2f} KB")
    print(f"    Overall Depth: {overall_depth}")


def range_to_ternary(num_range: tuple[int, int], bit_width: int) -> List[str]:
    """
    Converts a numerical range to a minimal list of ternary strings.
    This is a port of the `range2prefix` function often found in packet
    classification literature.
    """
    start, end = num_range
    result = []
    while start <= end:
        # Find the largest block that starts at 'start' and does not exceed 'end'.
        # This is equivalent to finding the longest prefix of 'start' that also
        # covers a range within 'end'.
        # 'l' is the number of trailing '*' in the ternary string.
        for l in range(bit_width + 1):
            mask = (1 << l) - 1
            # Check if the block starting at 'start' is aligned and doesn't exceed 'end'
            if (start | mask) > end or (start & mask) != 0:
                l -= 1
                break

        # Convert the block to a ternary string
        prefix = start >> l
        ternary = ""
        for i in range(bit_width - l):
            ternary += '1' if (prefix >> (bit_width - l - 1 - i)) & 1 else '0'
        ternary += '*' * l
        result.append(ternary)

        start += (1 << l)
    return result


UPPER_BOUNDS = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF, 0xFFFF, 0xFF]


def check_tree_correctness(trees: List['Node'], rules: List['Rule']):
    """
    Verifies the correctness of the CramCuts trees by generating test packets
    at the boundaries of each rule and comparing the tree's search result with
    a linear search of the original rules. This is a port of `CheckTrees` from
    `src-efficuts/checks.c`.
    """
    print("Verifying tree correctness...")

    def search_trees(packet: List[int]) -> 'Rule':
        """Helper to search across all trees and return the best match."""
        best_match = None
        for tree in trees:
            match = search_tree(tree, packet)
            if match:
                if best_match is None or match.priority < best_match.priority:
                    best_match = match
        return best_match

    def linear_search(packet: List[int]) -> 'Rule':
        """Helper for linear search over all rules."""
        best_match = None
        for rule in rules:
            matches = True
            for i in range(len(packet)):
                if not (rule.ranges[i][0] <= packet[i] <= rule.ranges[i][1]):
                    matches = False
                    break
            if matches:
                if best_match is None or rule.priority < best_match.priority:
                    best_match = rule
        return best_match

    def generate_test_points(rule: 'Rule', dim: int, current_packet: List[int]):
        """Recursively generates test points for a rule."""
        if dim >= len(rule.ranges):
            yield list(current_packet)
            return

        low, high = rule.ranges[dim]

        # Points to test for this dimension
        points_to_test = {low, high}
        if low > 0:
            points_to_test.add(low - 1)
        if high < UPPER_BOUNDS[dim]:
            points_to_test.add(high + 1)

        for point in sorted(list(points_to_test)):
            current_packet[dim] = point
            yield from generate_test_points(rule, dim + 1, current_packet)


    for i, rule in enumerate(rules):
        if (i + 1) % 100 == 0:
            print(f"  Checking rule {i+1}/{len(rules)}...")

        test_packet = [0] * len(rule.ranges)
        for packet in generate_test_points(rule, 0, test_packet):
            tree_match = search_trees(packet)
            linear_match = linear_search(packet)

            tree_priority = tree_match.priority if tree_match else -1
            linear_priority = linear_match.priority if linear_match else -1

            assert tree_priority == linear_priority, (
                f"Mismatch found!\n"
                f"Packet: {packet}\n"
                f"Rule: {rule}\n"
                f"Tree search found rule with priority: {tree_priority}\n"
                f"Linear search found rule with priority: {linear_priority}"
            )

    print("Tree correctness verification passed!")
def search_tree(root: 'Node', packet: List[int]) -> 'Rule':
    """
    Traverses the CramCuts tree to find the highest-priority matching rule for a packet.

    Args:
        root: The root node of the CramCuts tree.
        packet: A list of 5 integer values representing the packet headers.

    Returns:
        The highest-priority matching Rule, or None if no match is found.
    """
    current_node = root

    # 1. Traverse the tree until a leaf or TCAMNode is reached
    # A TCAMNode is also a leaf (no children), so this loop handles the descent.
    while current_node.children:
        found_child = False
        for child in current_node.children:
            # Check if the packet falls within the child's boundary
            in_boundary = True
            # A packet is a 5-tuple, corresponding to the 5 rule fields
            for i in range(len(packet)):
                if not (child.boundary.ranges[i][0] <= packet[i] <= child.boundary.ranges[i][1]):
                    in_boundary = False
                    break
            
            if in_boundary:
                current_node = child
                found_child = True
                break
        
        if not found_child:
            # This case means the packet doesn't fit into any more-specific child.
            # We must therefore search the rules contained at the current level.
            break

    # 2. At a leaf (standard or TCAMNode), perform a linear search of its rules
    best_match = None
    highest_priority = -1  # Assuming priority is always non-negative

    # This logic handles standard leaf nodes and TCAMNodes, as both store rules to be checked.
    # For a TCAMNode, this represents the lookup of its contents.
    for rule in current_node.rules:
        # Check if the packet matches the rule's ranges
        matches = True
        for i in range(len(packet)):
            if not (rule.ranges[i][0] <= packet[i] <= rule.ranges[i][1]):
                matches = False
                break
        
        if matches:
            # Higher priority value is considered better
            if rule.priority > highest_priority:
                highest_priority = rule.priority
                best_match = rule

    return best_match