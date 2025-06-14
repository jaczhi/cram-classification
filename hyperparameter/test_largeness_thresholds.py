#!/usr/bin/env python3

import subprocess
import re

def run_main_with_thresholds(largeness_threshold, ip_largeness_threshold):
    """Run main.py with given thresholds and extract depth and memory."""
    cmd = [
        "python3", "../main.py", 
        "../reproducibility/big-acl-list", 
        "--largeness-threshold", str(largeness_threshold),
        "--ip-largeness-threshold", str(ip_largeness_threshold)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Find the line with "Overall Depth: <depth>"
        depth_match = re.search(r'Overall Depth: (\d+)', output)
        # Find the LAST line with "Total Memory: <memory>" (handles decimal + KB)
        memory_matches = re.findall(r'Total Memory:\s+(\d+\.?\d*)', output)
        memory_match = memory_matches[-1] if memory_matches else None
        
        if depth_match and memory_match:
            return int(depth_match.group(1)), float(memory_match)
        else:
            print(f"Warning: Could not find depth/memory for thresholds {largeness_threshold}/{ip_largeness_threshold}")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"Timeout for thresholds {largeness_threshold}/{ip_largeness_threshold}")
        return None, None
    except Exception as e:
        print(f"Error for thresholds {largeness_threshold}/{ip_largeness_threshold}: {e}")
        return None, None

def main():
    results = []
    min_depth = float('inf')
    min_memory = float('inf')
    best_depth_combo = None
    best_memory_combo = None
    
    print("Testing largeness-threshold and ip-largeness-threshold combinations (0.0 to 1.0 in 0.1 increments)...")
    print("Largeness\tIP-Largeness\tDepth\tMemory")
    print("-" * 50)
    
    # Generate combinations: 0.0, 0.1, 0.2, ..., 1.0 (11 values each = 121 combinations)
    thresholds = [round(i * 0.1, 1) for i in range(11)]
    
    total_tests = len(thresholds) * len(thresholds)
    current_test = 0
    
    for largeness_threshold in thresholds:
        for ip_largeness_threshold in thresholds:
            current_test += 1
            print(f"Running test {current_test}/{total_tests}: {largeness_threshold}, {ip_largeness_threshold}", end=" ... ", flush=True)
            
            depth, memory = run_main_with_thresholds(largeness_threshold, ip_largeness_threshold)
            
            if depth is not None and memory is not None:
                print(f"Depth: {depth}, Memory: {memory}")
                results.append((largeness_threshold, ip_largeness_threshold, depth, memory))
                
                if depth < min_depth:
                    min_depth = depth
                    best_depth_combo = (largeness_threshold, ip_largeness_threshold)
                    
                if memory < min_memory:
                    min_memory = memory
                    best_memory_combo = (largeness_threshold, ip_largeness_threshold)
            else:
                print("FAILED")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if best_depth_combo is not None:
        print(f"Optimal combination for depth: largeness={best_depth_combo[0]}, ip-largeness={best_depth_combo[1]}")
        print(f"Minimum depth: {min_depth}")
        print()
        print(f"Optimal combination for memory: largeness={best_memory_combo[0]}, ip-largeness={best_memory_combo[1]}")
        print(f"Minimum memory: {min_memory}")
    else:
        print("No successful runs found!")
    
    # Show all results for reference
    print(f"\nAll successful results ({len(results)} total):")
    print("\nSorted by depth (top 10):")
    sorted_by_depth = sorted(results, key=lambda x: x[2])
    for i, (l_thresh, ip_thresh, depth, memory) in enumerate(sorted_by_depth[:10]):
        print(f"{i+1}. Largeness: {l_thresh}, IP-Largeness: {ip_thresh} -> Depth: {depth}, Memory: {memory}")
    
    print("\nSorted by memory (top 10):")
    sorted_by_memory = sorted(results, key=lambda x: x[3])
    for i, (l_thresh, ip_thresh, depth, memory) in enumerate(sorted_by_memory[:10]):
        print(f"{i+1}. Largeness: {l_thresh}, IP-Largeness: {ip_thresh} -> Depth: {depth}, Memory: {memory}")

if __name__ == "__main__":
    main()
