#!/usr/bin/env python3

import subprocess
import re
import sys

def run_main_with_threshold(threshold):
    """Run main.py with given threshold and extract depth and memory."""
    cmd = [
        "python3", "main.py", 
        "reproducibility/big-acl-list", 
        "--tcam-wide-rule-threshold", str(threshold)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Find the line with "Overall Depth: <depth>"
        depth_match = re.search(r'Overall Depth: (\d+)', output)
        # Find the line with "Total Memory: <memory>" (handles decimal + KB)
        memory_match = re.search(r'Total Memory:\s+(\d+\.?\d*)', output)
        
        if depth_match and memory_match:
            return int(depth_match.group(1)), float(memory_match.group(1))
        else:
            print(f"Warning: Could not find depth/memory for threshold {threshold}")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"Timeout for threshold {threshold}")
        return None, None
    except Exception as e:
        print(f"Error for threshold {threshold}: {e}")
        return None, None

def main():
    results = []
    min_depth = float('inf')
    min_memory = float('inf')
    best_depth_threshold = None
    best_memory_threshold = None
    
    print("Testing thresholds 0-100...")
    print("Threshold\tDepth\tMemory")
    print("-" * 30)
    
    for threshold in range(101):  # 0 to 100 inclusive
        depth, memory = run_main_with_threshold(threshold)
        
        if depth is not None and memory is not None:
            print(f"{threshold}\t\t{depth}\t{memory}")
            results.append((threshold, depth, memory))
            
            if depth < min_depth:
                min_depth = depth
                best_depth_threshold = threshold
                
            if memory < min_memory:
                min_memory = memory
                best_memory_threshold = threshold
        else:
            print(f"{threshold}\t\tFAILED")
    
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    
    if best_depth_threshold is not None:
        print(f"Optimal threshold for depth: {best_depth_threshold}")
        print(f"Minimum depth: {min_depth}")
        print()
        print(f"Optimal threshold for memory: {best_memory_threshold}")
        print(f"Minimum memory: {min_memory}")
    else:
        print("No successful runs found!")
    
    # Show all results for reference
    print(f"\nAll successful results ({len(results)} total):")
    print("Sorted by depth:")
    for threshold, depth, memory in sorted(results, key=lambda x: x[1]):
        print(f"Threshold {threshold}: Depth {depth}, Memory {memory}")
    
    print("\nSorted by memory:")
    for threshold, depth, memory in sorted(results, key=lambda x: x[2]):
        print(f"Threshold {threshold}: Depth {depth}, Memory {memory}")

if __name__ == "__main__":
    main()