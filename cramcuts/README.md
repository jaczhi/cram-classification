# CramCuts

Efficuts with TCAM nodes.

CramCuts preserves concepts from EffiCuts, but adds an additional type of node in addition to equi-spaced and equi-dense nodes. 
CramCuts follows the "CRAM" model of blending TCAM and SRAM through each stage, thereby using the memory resources of both. 
(See roo/efficuts-paper.txt for the original efficuts paper.)

This type of node, known as a TCAM node, has the following differences from a equi-dense node:

1. More cuts (i.e., hundreds) -- it is used to seperate dimensions where neither equi-spaced or equi-dense cuts work well. 
2. Cuts along values that minimize the number of TCAM rules. For example, it is neater to cut along 2-7 (001\*, 01\*\*) rather than 2-8 (001\*, 01\*\*, 1111).

### The TCAM Node Heuristic

The decision to use a TCAM node is made when a rule set is too dense for a standard equi-dense cut, but does not contain many "wide" rules that would cause a rule replication explosion.

A TCAM node is essentially a specialized equi-dense node that is allowed to have a much larger number of cuts (`max_cuts` can be in the hundreds). This is beneficial for densely clustered rules that require many fine-grained partitions. However, making many cuts is only feasible if it doesn't significantly replicate wide-spanning rules across the resulting children.

The heuristic is as follows:

1.  At a given node, the algorithm first attempts to find a suitable partition using the standard **equi-dense** method, where the number of cuts is limited (e.g., to 8 or 16).
2.  If no effective equi-dense partition can be found within this limit, the node is considered "problematic."
3.  For a problematic node, the algorithm analyzes the rules to determine if a TCAM node is appropriate. It checks the percentage of rules that are "wide" (i.e., span a large portion of the address space) in the dimension(s) being considered for cutting.
4.  **Decision Point:**
    *   If the percentage of wide rules is **low**, rule replication is not a major concern. The algorithm proceeds with creating a **TCAM node**, which can make the many cuts necessary to resolve the dense cluster of specific rules.
    *   If the percentage of wide rules is **high**, creating a TCAM node would cause massive rule replication. In this case, the algorithm falls back to creating a standard **equi-spaced node**, accepting its costs to avoid the even greater costs of replicating the wide rules.

This approach ensures that TCAM nodes are used surgically: only on dense clusters of specific (non-wide) rules where their powerful, multi-cut capability can be leveraged without the major drawback of rule replication.

The CramCuts executable works similarly to the EffiCuts executable, building a "tree" and then reporting metrics.
Note that since we don't have actual TCAM hardware, the TCAM datastructure is simulated using a Python class.

## Running

Show the help:

```bash
# cwd should be "cram-classification" i.e., repo root
python3 -m cramcuts.cramcuts --help
# or
python -m cramcuts.cramcuts --help
```

Example run:

```bash
python3 -m cramcuts.cramcuts reproducibility/big-acl-list

python3 -m cramcuts.cramcuts reproducibility/big-acl-list --no-cramcuts
```

## Command-Line Options

The following command-line options are available for the `cramcuts.py` script:

### Efficuts Parameters

*   **`rules_file`**
    *   **Description:** Path to the input ruleset file. (Mandatory)

*   **`-b, --bucket-size <int>`**
    *   **Default:** 16
    *   **Description:** Max rules in a leaf node before splitting.

*   **`-s, --space-factor <float>`**
    *   **Default:** 0.8
    *   **Description:** A float to control rule replication.

*   **`--no-compression`**
    *   **Default:** False
    *   **Description:** Disable node compression heuristics (currently a placeholder).

*   **`-g, --binning-strategy <0|1|2>`**
    *   **Default:** 2
    *   **Description:** Rule binning strategy (0: None, 1: Separable Trees, 2: Static).

*   **`--no-tree-merging`**
    *   **Default:** False
    *   **Description:** Disable selective tree merging.

*   **`--no-equi-dense`**
    *   **Default:** False
    *   **Description:** Disable equi-dense cuts in favor of equi-spaced cuts (currently a placeholder).

*   **`-f, --max-cuts <int>`**
    *   **Default:** 16
    *   **Description:** Max number of cuts in a standard equi-dense node.

*   **`-n, --largeness-threshold <float>`**
    *   **Default:** 0.9
    *   **Description:** "Large rule" threshold in non-IP dimensions.

*   **`-i, --ip-largeness-threshold <float>`**
    *   **Default:** 0.9
    *   **Description:** "Large rule" threshold in IP dimensions.

### CramCuts Parameters

*   **`--no-cramcuts`**
    *   **Default:** False
    *   **Description:** Disable the CramCuts heuristic (run in pure Efficuts mode).

*   **`--tcam-max-cuts <int>`**
    *   **Default:** 256
    *   **Description:** Max number of cuts allowed in a TCAMNode.

*   **`--tcam-wide-rule-threshold <float>`**
    *   **Default:** 20.0
    *   **Description:** Percentage of "wide" rules below which a TCAM node will be created.

### Output file

*   **`--simulator-file <str>`**
    *   **Default:** `cramcuts-tree.json`
    *   **Description:** Path to save the simulator JSON file.
