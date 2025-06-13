import argparse
from cramcuts import *


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CramCuts: A decision tree building algorithm for packet classification.")

    # Efficuts Parameters
    parser.add_argument('rules_file', help='Path to the input ruleset file.')
    parser.add_argument('-b', '--bucket-size', type=int, default=16,
                        help='Max rules in a leaf node before splitting.')
    parser.add_argument('-s', '--space-factor', type=float,
                        default=0.8, help='A float to control rule replication.')
    parser.add_argument('--no-compression', action='store_true',
                        help='Disable node compression heuristics (currently a placeholder).')
    parser.add_argument('-g', '--binning-strategy', type=int, default=2, choices=[
                        0, 1, 2], help='Rule binning strategy (0: None, 1: Separable Trees, 2: Static).')
    parser.add_argument('--no-tree-merging', action='store_true',
                        help='Disable selective tree merging.')
    parser.add_argument('--no-equi-dense', action='store_true',
                        help='Disable equi-dense cuts in favor of equi-spaced cuts (currently a placeholder).')
    parser.add_argument('-f', '--max-cuts', type=int, default=16,
                        help='Max number of cuts in a standard equi-dense node.')
    parser.add_argument('-n', '--largeness-threshold', type=float,
                        default=0.9, help='"Large rule" threshold in non-IP dimensions.')
    parser.add_argument('-i', '--ip-largeness-threshold', type=float,
                        default=0.9, help='"Large rule" threshold in IP dimensions.')

    # CramCuts Parameters
    parser.add_argument('--no-cramcuts', action='store_true',
                        help='Disable the CramCuts heuristic (run in pure Efficuts mode).')
    parser.add_argument('--tcam-max-cuts', type=int, default=256,
                        help='Max number of cuts allowed in a TCAMNode.')
    parser.add_argument('--tcam-wide-rule-threshold', type=float, default=20.0,
                        help='Percentage of "wide" rules below which a TCAM node will be created.')

    # Output file
    parser.add_argument('--simulator-file', type=str, default='cramcuts-tree.json',
                        help='Path to save the simulator JSON file.')

    return parser.parse_args()


def main(args):
    """Main execution logic."""
    # 1. Load rules
    print(f"Loading rules from {args.rules_file}...")
    rules = load_rules(args.rules_file)
    print(f"Loaded {len(rules)} rules.")

    # 2. Process rules (binning and merging)
    final_rule_sets = [rules]
    if args.binning_strategy > 0:
        print("Binning rules...")
        binned_rules = bin_rules(rules)
        if not args.no_tree_merging:
            print("Merging rule bins...")
            final_rule_sets = merge_trees(binned_rules)
        else:
            final_rule_sets = binned_rules
    print(f"Created {len(final_rule_sets)} rule sets for tree generation.")

    # 3. Build trees
    use_cramcuts_heuristic = not args.no_cramcuts
    mode_string = "CramCuts" if use_cramcuts_heuristic else "Efficuts"

    print(f"\n--- Running in {mode_string} Mode ---")
    print(f"Building {mode_string}-style trees...")

    all_trees = []
    max_depth = 0
    for i, rule_set in enumerate(final_rule_sets):
        print(
            f"  Building tree {i+1}/{len(final_rule_sets)} with {len(rule_set)} rules...")
        tree, depth = create_cramcuts_tree(
            rules=rule_set,
            bucket_size=args.bucket_size,
            use_tcam_heuristic=use_cramcuts_heuristic,
            spfac=args.space_factor,
            max_cuts=args.max_cuts,
            ip_largeness_threshold=args.ip_largeness_threshold,
            largeness_threshold=args.largeness_threshold,
            tcam_max_cuts=args.tcam_max_cuts,
            tcam_wide_rule_threshold=args.tcam_wide_rule_threshold,
            no_compression=args.no_compression,
            no_equi_dense=args.no_equi_dense
        )
        all_trees.append(tree)
        if depth > max_depth:
            max_depth = depth

    print(f"{mode_string} tree construction complete.")
    print(f"\nFinal Statistics ({mode_string} Mode):")
    print_stats(all_trees, max_depth)

    if use_cramcuts_heuristic:
        print(
            f"\nGenerating JSON representation of the CramCuts tree to {args.simulator_file}...")
        generate_json_representation(all_trees, args.simulator_file)
        print(f"JSON file '{args.simulator_file}' created.")

    check_tree_correctness(all_trees, rules)


if __name__ == '__main__':
    args = parse_args()
    main(args)
