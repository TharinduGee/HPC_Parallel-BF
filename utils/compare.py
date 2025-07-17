import sys
import numpy as np

INT_MAX = 2147483647
PENALTY = 1e6 ** 2

def load_distances(filename):
    try:
        with open(filename) as f:
            lines = f.readlines()
            return [int(line.strip()) for line in lines if line.strip().isdigit()]
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

def compute_rmse(a, b):
    if len(a) != len(b):
        print("Error: Distance arrays are of different lengths.")
        sys.exit(1)

    squared_errors = []
    for x, y in zip(a, b):
        if x == INT_MAX and y == INT_MAX:
            continue 
        elif x == INT_MAX or y == INT_MAX:
            squared_errors.append(PENALTY)
        else:
            squared_errors.append((x - y) ** 2)

    if not squared_errors:
        print("Error: No valid values to compute RMSE.")
        sys.exit(1)

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_rmse.py <file1.txt> <file2.txt>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    a = load_distances(file1)
    b = load_distances(file2)
    rmse = compute_rmse(a, b)

    print(f"RMSE between '{file1}' and '{file2}': {rmse:.6f}")
