# EffiCuts / CompressedCuts

## Introduction

The `compressedcuts` executable is a C++ program designed to implement and evaluate various decision-tree-based packet classification algorithms. It is written by Balajee Vamanan, Gwendolyn Voskuilen and T. N. Vijaykumar, with some improvements by James Daly. Based on the provided source code and the associated research paper ("EffiCuts: Optimizing Packet Classification for Memory and Throughput"), the executable primarily implements:

1.  **HiCuts:** An earlier decision-tree algorithm (controlled by the `-m 0` flag).
2.  **HyperCuts:** An enhancement over HiCuts (controlled by the `-m 1` flag, which is the default).
3.  **EffiCuts:** The algorithm proposed in the paper, which introduces optimizations like Separable Trees, Selective Tree Merging, Equi-dense Cuts, and Node Co-location. These features are enabled by default or controlled by specific flags (`-g 1`, `-z 1`, `-F 1`).

The program takes a classifier ruleset file as input, builds the corresponding decision tree(s) using the selected algorithm and parameters, and outputs performance statistics like memory usage, tree depth, and estimated memory accesses. It also includes functionality for correctness checks and potentially simulating rule updates (as hinted by `mainAdd`/`mainSub` functions, although `mainNormal` seems to be the primary build/test path).

## Command-Line Options

The following command-line options are available for the `compressedcuts` executable, parsed in `cutio.c`:

* **`-b <int>`**
    * **Variable:** `bucketSize`
    * **Default:** 16
    * **Description:** Sets the maximum number of rules allowed in a leaf node before the node must be split.
    * **Paper Link:** Corresponds to the **`binth`** threshold mentioned in Sections 2.1 and 4 of the EffiCuts paper.

* **`-s <float>`**
    * **Variable:** `spfac`
    * **Default:** 8.0
    * **Description:** Sets the space factor used in the HiCuts/HyperCuts cutting heuristic. This factor limits the allowed increase in the total number of rules stored in child nodes compared to the parent node, aiming to control rule replication.
    * **Paper Link:** Corresponds to the **`space_factor`** parameter (Section 2.1, Heuristic 2). The paper notes `space_factor=4` was used for HyperCuts and `space_factor=8` for EffiCuts experiments (Section 4).

* **`-r <filename>`**
    * **Variable:** `fpr` (FILE pointer)
    * **Default:** None (Mandatory)
    * **Description:** Specifies the path to the input file containing the classifier ruleset.
    * **Paper Link:** N/A (Input data specification).

* **`-m <0|1>`**
    * **Variable:** `hypercuts`
    * **Default:** 1
    * **Description:** Selects the base cutting algorithm logic. `0` enables HiCuts mode, `1` enables HyperCuts mode.
    * **Paper Link:** EffiCuts builds upon HyperCuts (Section 3). This switch allows comparing against baseline HiCuts or HyperCuts.

* **`-u <int>`**
    * **Variable:** `Num_Rules_Moved_Up`
    * **Default:** 0
    * **Description:** Controls the number of rules allowed to be "moved up" to parent nodes, a heuristic from HyperCuts to reduce replication.
    * **Paper Link:** Controls the HyperCuts "rule moving-up" heuristic (Section 2.2). EffiCuts explicitly *disables* this (`Num_Rules_Moved_Up = 0`), as stated in Sections 3.4 and 4.

* **`-c <0|1>`**
    * **Variable:** `compressionON`
    * **Default:** 1
    * **Description:** Enables or disables node compression heuristics (`NodeCompress`, `LogicalMerge`). This seems related to optimizations like merging identical siblings (HiCuts heuristic 3, Section 2.1) or potentially region compaction aspects rather than EffiCuts' equi-dense cuts.
    * **Paper Link:** Likely relates to HiCuts/HyperCuts optimizations retained by EffiCuts (Section 3 states EffiCuts keeps HyperCuts heuristics except rule moving-up).

* **`-g <0|1|2>`**
    * **Variable:** `binningON`
    * **Default:** 1
    * **Description:** Controls the rule binning strategy before building trees.
        * `0`: No binning (standard HiCuts/HyperCuts single tree).
        * `1`: EffiCuts' binning based on rule separability.
        * `2`: Static binning based only on rule size (used for comparison in the paper).
    * **Paper Link:** Directly enables/disables the **Separable Trees** concept (Section 3.1). Option `2` relates to the comparison in Section 5.4.1.

* **`-z <0|1>`**
    * **Variable:** `mergingON`
    * **Default:** 1
    * **Description:** Enables or disables the selective merging of the created trees (`MergeTrees` function).
    * **Paper Link:** Directly enables/disables the **Selective Tree Merging** optimization (Section 3.2).

* **`-F <0|1>`**
    * **Variable:** `fineOn`
    * **Default:** 1
    * **Description:** Controls the type of cuts used within nodes.
        * `0`: Uses equi-sized cuts (powers of two, like HiCuts/HyperCuts).
        * `1`: Uses equi-dense cuts (unequal cuts, implemented in `fine.c`).
    * **Paper Link:** Directly enables/disables **Equi-dense Cuts** (Section 3.3).

* **`-f <int>`**
    * **Variable:** `num_intervals`
    * **Default:** 7
    * **Description:** Sets the maximum number of intervals (and thus boundaries) allowed when using equi-dense cuts (`-F 1`). The help string confusingly mentions `comfac`.
    * **Paper Link:** Corresponds to the **`max_cuts`** threshold for equi-dense cuts (Section 3.3.2). The paper used `max_cuts = 8`, which requires storing 7 boundaries (intervals). This matches the default `num_intervals = 7`.

* **`-n <float>`**
    * **Variable:** `bin`
    * **Default:** 0.5
    * **Description:** Sets the threshold for determining if a rule is "large" in non-IP dimensions (ports, protocol) based on the fraction of the dimension it covers.
    * **Paper Link:** Corresponds to **`largeness_fraction`** for non-IP fields (Section 3.1.1). The paper used 0.5 in experiments (Section 4).

* **`-i <float>`**
    * **Variable:** `IPbin`
    * **Default:** 0.05
    * **Description:** Sets the threshold for determining if a rule is "large" in IP dimensions (source/destination IP) based on the fraction of the dimension it covers.
    * **Paper Link:** Corresponds to **`largeness_fraction`** for IP fields (Section 3.1.1). The paper used 0.05 in experiments (Section 4).

* **`-t <0|1>`**
    * **Variable:** `thirtyone`
    * **Default:** 0
    * **Description:** Appears to be a flag related to the specific handling of Category 4 rules (0 or 1 wildcard field) during the separability binning (`-g 1`).
    * **Paper Link:** Likely relates to the implementation detail for Category 4 rules mentioned in Section 3.1.1, where further sub-categorization was deemed unnecessary.

* **`-R <int>`**
    * **Variable:** `numReps`
    * **Default:** 1
    * **Description:** Sets the number of times the 5 standard fields (SIP, DIP, SP, DP, Proto) are repeated to form the rule structure. Used for testing with wider rule keys.
    * **Paper Link:** Not a conceptual parameter in the paper, likely for experimental flexibility.

* **`-h`**
    * **Description:** Displays a help message summarizing the options (though the help string for `-f` seems incorrect).

* **Node Co-location:** This EffiCuts optimization (Section 3.4) doesn't appear to have a dedicated command-line switch. Its use is likely tied to enabling equi-dense cuts (`-F 1`), as the paper suggests it's most beneficial and practical in that scenario.

## Example Usage

1.  **Run EffiCuts with default parameters on a ruleset:**
    ```bash
    ./compressedcuts -r my_ruleset.rules
    ```
    *(This uses defaults: `-b 16 -s 8.0 -m 1 -u 0 -c 1 -g 1 -z 1 -F 1 -f 7 -n 0.5 -i 0.05 -t 0`)*

2.  **Run HyperCuts (for comparison) on the same ruleset:**
    ```bash
    # Disable EffiCuts specific features: binning, merging, fine cuts
    ./compressedcuts -r my_ruleset.rules -g 0 -z 0 -F 0 -m 1 -u 1 -s 4.0
    ```
    *(This disables separable trees (`-g 0`), tree merging (`-z 0`), equi-dense cuts (`-F 0`), enables HyperCuts rule moving-up (`-u 1`), and uses the space factor mentioned for HyperCuts in the paper (`-s 4.0`)).*

3.  **Run EffiCuts with a smaller bucket size and disabled tree merging:**
    ```bash
    ./compressedcuts -r my_ruleset.rules -b 8 -z 0
    ```
## Code Structure

The provided source code implements the packet classification algorithms in a modular C++ structure. Here's a breakdown of the roles of the key files:

* **`compressedcuts.c`**: This file contains the main program logic and the core decision tree construction algorithm.
  * It includes the `main` function, which orchestrates the overall process.
  * Functions like `create_tree`, `CutNode`, `CutRecursive` manage the tree building process, likely using a recursive or worklist-based approach.
  * It contains the logic (`calc_cuts`, `calc_dimensions_to_cut`, `calc_num_cuts_1D`, `calc_num_cuts_2D`) for deciding *how* to cut a node based on HiCuts/HyperCuts heuristics (e.g., selecting dimensions, determining number of equi-sized cuts).
  * It defines the `node` class constructor and destructor.

* **`cutio.c`**: Handles various input, output, and utility tasks.
  * `parseargs`: Parses command-line arguments to configure the algorithm.
  * `loadrule`, `IP2Range`: Reads and parses the ruleset file into the internal `pc_rule` format.
  * `NodeStats`, `RecordTreeStats`, `PrintStats`: Collects and prints performance statistics about the generated trees.
  * Contains helper functions like `cp_node` (copy node) and `ClearMem`.

* **`merging.c`**: Implements various optimization heuristics, primarily those related to merging, redundancy removal, and rule binning.
  * `remove_redund`, `nodeMerging`: Implement rule redundancy checks and merging of identical sibling nodes (HiCuts Heuristic 3).
  * `moveRulesUp`: Implements the HyperCuts rule moving-up heuristic.
  * `regionCompaction`: Implements the HyperCuts region compaction heuristic.
  * `NodeCompress`, `LogicalMerge`: Implement node compression/fusion logic, potentially related to HiCuts optimizations or equi-dense cut implementation details.
  * `binRules`: Implements the rule categorization logic for **EffiCuts' Separable Trees** based on `largeness_fraction` (`bin`, `IPbin` variables).
  * `MergeTrees`: Implements the logic for **EffiCuts' Selective Tree Merging**.

* **`fine.c`**: Contains the implementation specific to **EffiCuts' Equi-dense Cuts**.
  * `CalcMultiEquiCuts1D`, `CalcEquiCuts1D`, `CalcEquiCuts2D`: Generate the unequal cuts based on rule distribution.
  * `BestSplitPoint`: Determines the optimal point to make an unequal split in a given dimension.
  * `SpawnChild`: Helper function to create child nodes during cutting.

* **`checks.c`**: Provides utility functions for validation, comparison, and sorting.
  * Includes functions to check bounds (`CheckIPBounds`, `CheckPortBounds`), rule intersections (`DoRulesIntersect`), rule containment (`is_present`), and tree correctness (`CheckTrees`, `ColorOfTree`).
  * Contains comparators for rules (`mycomparison`, `myequal`) and statistics structures (`mystatsort`, `mymemsort`) used for sorting and merging.

* **`shared.c` / `shared.h`**: Define and declare global variables (configuration parameters set by `parseargs`, statistics counters) and shared data structures (`pc_rule`, `node`, `TreeStat`, `TreeDetails`) used across multiple files.

* **`compressedcuts.h`**: Header file primarily declaring the data structures (`pc_rule`, `node`, `TreeStat`, `TreeDetails`) and function prototypes defined in the `.c` files.

**Overall Flow:**
The program starts in `main` (`compressedcuts.c`), parses arguments (`cutio.c`), loads rules (`cutio.c`), optionally bins rules (`merging.c`) and merges trees (`merging.c`) if EffiCuts mode is fully enabled, then builds the tree(s) via `create_tree` (`compressedcuts.c`). Tree construction involves recursively calling `CutNode` (`compressedcuts.c`), which decides cutting strategy (`compressedcuts.c` or `fine.c`) and applies various heuristics (`merging.c`). Finally, statistics are calculated and printed (`cutio.c`).
