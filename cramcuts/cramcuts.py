import argparse

def parse_args():
    """
    Parses command-line arguments for the cramcuts script.
    """
    parser = argparse.ArgumentParser(description="CramCuts: Python argument parser mirroring EffiCuts.")

    # -b <int>: bucketSize
    parser.add_argument(
        '-b', '--bucketSize',
        type=int,
        default=16,
        help='Sets the maximum number of rules allowed in a leaf node. Corresponds to binth in the paper. (Default: 16)'
    )

    # -s <float>: spfac
    parser.add_argument(
        '-s', '--spfac',
        type=float,
        default=8.0,
        help='Sets the space factor for the HiCuts/HyperCuts cutting heuristic. (Default: 8.0)'
    )

    # -r <filename>: ruleset file
    parser.add_argument(
        '-r', '--ruleset',
        type=str,
        required=True,
        help='Specifies the path to the input file containing the classifier ruleset. (Mandatory)'
    )

    # -m <0|1>: hypercuts
    parser.add_argument(
        '-m', '--hypercuts',
        type=int,
        default=1,
        choices=[0, 1],
        help='Selects the base cutting algorithm. 0 for HiCuts, 1 for HyperCuts. (Default: 1)'
    )

    # -u <int>: Num_Rules_Moved_Up
    parser.add_argument(
        '-u', '--pushup',
        type=int,
        default=0,
        help='Controls the number of rules allowed to be "moved up" to parent nodes. (Default: 0)'
    )

    # -c <0|1>: compressionON
    parser.add_argument(
        '-c', '--compressionON',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enables or disables node compression heuristics. (Default: 1)'
    )

    # -g <0|1|2>: binningON
    parser.add_argument(
        '-g', '--binningON',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Controls the rule binning strategy. 0: No binning, 1: EffiCuts binning, 2: Static binning. (Default: 1)'
    )

    # -z <0|1>: mergingON
    parser.add_argument(
        '-z', '--mergingON',
        type=int,
        default=1,
        choices=[0, 1],
        help='Enables or disables the selective merging of created trees. (Default: 1)'
    )

    # -F <0|1>: fineOn
    parser.add_argument(
        '-F', '--fineOn',
        type=int,
        default=1,
        choices=[0, 1],
        help='Controls the type of cuts used. 0: equi-sized cuts, 1: equi-dense cuts. (Default: 1)'
    )

    # -f <int>: num_intervals
    parser.add_argument(
        '-f', '--num_intervals',
        type=int,
        default=7,
        help='Sets the maximum number of intervals for equi-dense cuts. (Default: 7)'
    )

    # -n <float>: bin
    parser.add_argument(
        '-n', '--bin',
        type=float,
        default=0.5,
        help='Sets the threshold for determining if a rule is "large" in non-IP dimensions. (Default: 0.5)'
    )

    # -i <float>: IPbin
    parser.add_argument(
        '-i', '--IPbin',
        type=float,
        default=0.05,
        help='Sets the threshold for determining if a rule is "large" in IP dimensions. (Default: 0.05)'
    )

    # -t <0|1>: thirtyone
    parser.add_argument(
        '-t', '--thirtyone',
        type=int,
        default=0,
        choices=[0, 1],
        help='A flag related to handling of Category 4 rules during separability binning. (Default: 0)'
    )

    # -R <int>: numReps
    parser.add_argument(
        '-R', '--numReps',
        type=int,
        default=1,
        help='Sets the number of times the 5 standard fields are repeated. (Default: 1)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")