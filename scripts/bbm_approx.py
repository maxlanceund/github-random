import numpy as np
from numba import njit
import re
import os
import json
import math
from collections import Counter
import sys

sys.path.insert(0, 'scripts')
from bbm_d_f import parse_bnet, compile_rules

@njit
def next_state(current, N, input_indices, input_K, table_offsets, tables):
    new = 0
    for i in range(N):
        idx = 0
        for k in range(input_K[i]):
            var_idx = input_indices[i, k]
            bit = (current >> var_idx) & 1
            idx = (idx << 1) | bit
        bit_i = tables[table_offsets[i] + idx]
        if bit_i:
            new |= (1 << i)
    return new

@njit
def find_attractor(initial, N, input_indices, input_K, table_offsets, tables, max_steps):
    visit = {}
    path = []
    current = initial
    for step in range(max_steps):
        if current in visit:
            return current
        visit[current] = step
        path.append(current)
        new = next_state(current, N, input_indices, input_K, table_offsets, tables)
        if new == current:
            return current
        current = new
    return current

@njit
def sample_attractors(N, input_indices, input_K, table_offsets, tables, M, max_steps, seed):
    np.random.seed(seed)
    attractors = np.zeros(M, dtype=np.int64)
    for s in range(M):
        initial = np.random.randint(0, 2**min(N, 62))
        att = find_attractor(initial, N, input_indices, input_K, table_offsets, tables, max_steps)
        attractors[s] = att
    return attractors

def compute_D_f_approx(filepath, M=100000, max_steps=500):
    variables, expressions = parse_bnet(filepath)
    N = len(variables)
    if N == 0:
        return None
    try:
        input_indices, input_K, table_offsets, tables = compile_rules(variables, expressions)
    except Exception:
        return None

    if N <= 16:
        # 精确枚举
        from bbm_d_f import compute_all_attractors
        attractors = compute_all_attractors(N, input_indices, input_K, table_offsets, tables, max_steps)
        counts = Counter(attractors.tolist())
        n_states = 1 << N
        H_init = float(N)
        H_cond = 0.0
        for count in counts.values():
            p = count / n_states
            H_cond += p * math.log2(count)
        D_f = H_cond / H_init
        return {"N": N, "D_f": D_f, "method": "exact", "n_attractors": len(counts)}
    else:
        # 蒙特卡洛采样
        try:
            attractors = sample_attractors(N, input_indices, input_K, table_offsets, tables,
                                            M, max_steps, seed=42)
        except Exception:
            return None
        counts = Counter(attractors.tolist())
        n_sampled = len(attractors)
        H_attractor = 0.0
        for count in counts.values():
            p = count / n_sampled
            H_attractor -= p * math.log2(p)
        D_f = 1.0 - H_attractor / N
        if D_f < 0:
            D_f = 0.0
        return {"N": N, "D_f": D_f, "method": "MC", "n_attractors": len(counts)}

if __name__ == "__main__":
    bbm_root = "bbm/models"
    bnet_files = []
    for root, dirs, files in os.walk(bbm_root):
        for f in files:
            if f.endswith('.bnet'):
                bnet_files.append(os.path.join(root, f))

    print(f"Total .bnet files: {len(bnet_files)}")
    print("Precompiling numba...", flush=True)
    _ = sample_attractors(3, np.zeros((3,1), dtype=np.int32),
                          np.ones(3, dtype=np.int32),
                          np.array([0,1,2,3], dtype=np.int32),
                          np.array([0,1,0,1,0,1], dtype=np.int32),
                          10, 50, 1)
    print("Compiled.")
    print()

    results = []
    for filepath in bnet_files:
        try:
            r = compute_D_f_approx(filepath, M=100000)
            if r is None:
                continue
            model_name = os.path.basename(os.path.dirname(filepath))
            r["model"] = model_name
            results.append(r)
            print(f"N={r['N']:>3} | D_f={r['D_f']:.4f} | {r['method']:>5} | attr={r['n_attractors']:>5} | {model_name[:45]}", flush=True)
        except Exception:
            pass

    print()
    print(f"Total models: {len(results)}")
    n_exact = sum(1 for r in results if r['method'] == 'exact')
    n_mc = sum(1 for r in results if r['method'] == 'MC')
    print(f"  exact (N<=16): {n_exact}")
    print(f"  MC (N>16):     {n_mc}")

    if results:
        Ns = [r['N'] for r in results]
        Dfs = [r['D_f'] for r in results]
        print(f"N range: {min(Ns)} - {max(Ns)}")
        print(f"D_f mean: {np.mean(Dfs):.4f}")
        print(f"D_f std:  {np.std(Dfs):.4f}")

    with open("bbm_approx_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to bbm_approx_result.json")
