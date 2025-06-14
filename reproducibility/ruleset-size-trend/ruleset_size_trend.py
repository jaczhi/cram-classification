import sys
import os
import matplotlib.pyplot as plt

# Add the project root to sys.path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cramcuts import *


def run_analysis(use_cramcuts_heuristic: bool):
    """
    Runs the full analysis for a given heuristic setting.
    """
    rule_sizes = []
    total_memories = []
    max_depths = []

    heuristic_label = "With Heuristic" if use_cramcuts_heuristic else "Without Heuristic"
    print(f"--- Running Analysis {heuristic_label} ---")

    for i in range(5, 101, 5):
        num_rules = i * 1000
        rules_file = os.path.join('acl-list', f'acl-list-{i:03d}')
        
        print(f"\nProcessing {rules_file} ({num_rules} rules)...")

        if not os.path.exists(rules_file):
            print(f"Warning: Rules file not found: {rules_file}. Skipping.")
            continue
            
        rule_sizes.append(num_rules)

        # 1. Load rules
        rules = load_rules(rules_file)

        # 2. Process rules (binning and merging)
        binned_rules = bin_rules(rules)
        if len(binned_rules) > 1:
            final_rule_sets = merge_trees(binned_rules)
        else:
            final_rule_sets = binned_rules if binned_rules else [rules]

        # 3. Build trees
        all_trees = []
        current_max_depth = 0
        for j, rule_set in enumerate(final_rule_sets):
            tree, depth = create_cramcuts_tree(
                rules=rule_set,
                use_tcam_heuristic=use_cramcuts_heuristic,
            )
            all_trees.append(tree)
            if depth > current_max_depth:
                current_max_depth = depth
        max_depths.append(current_max_depth)

        # 4. Get stats and record memory
        stats = get_stats(all_trees)
        total_mem_bytes = sum(s.total_memory for s in stats)
        total_mem_kb = total_mem_bytes / 1024
        total_memories.append(total_mem_kb)
        print(f"  Total memory: {total_mem_kb:.2f} KB, Max Depth: {current_max_depth}")

    return rule_sizes, total_memories, max_depths


def main():
    """
    Generates plots of memory usage and max depth vs. ruleset size
    for CramCuts, comparing with and without the TCAM heuristic.
    """
    # Run analysis for both scenarios
    rule_sizes_h, mem_h, depth_h = run_analysis(use_cramcuts_heuristic=True)
    rule_sizes_nh, mem_nh, depth_nh = run_analysis(use_cramcuts_heuristic=False)

    # 5. Plotting
    print("\nGenerating plots...")
    # Create a figure with two subplots, sharing the x-axis
    # Figure size suitable for a single-column paper format (e.g., 3.5 inches wide)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 6), sharex=True)

    # Plot 1: Total Memory vs. Ruleset Size
    ax1.plot(rule_sizes_h, mem_h, marker='o', linestyle='-', label='CramCuts')
    ax1.plot(rule_sizes_nh, mem_nh, marker='x', linestyle='--', label='Efficuts')
    #ax1.set_title('CramCuts Performance vs. Ruleset Size')
    ax1.set_ylabel('Total Memory (KB)')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Max Depth vs. Ruleset Size
    ax2.plot(rule_sizes_h, depth_h, marker='o', linestyle='-', label='CramCuts')
    ax2.plot(rule_sizes_nh, depth_nh, marker='x', linestyle='--', label='Efficuts')
    ax2.set_xlabel('Number of Rules')
    ax2.set_ylabel('Maximum Tree Depth')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout(pad=0.5)
    
    plot_filename = 'ruleset_size_trend.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to '{plot_filename}'")


if __name__ == '__main__':
    main()