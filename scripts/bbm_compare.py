import numpy as np
from numba import njit
import re
import os
import json
import math
from collections import Counter
import sys

sys.path.insert(0, 'scripts')
from bbm_d_f import parse_bnet, compile_rules, compute_all_attractors

def get_N_K(filepath):
    """从真实网络提取 N 和每个节点的 K"""
    variables, expressions = parse_bnet(filepath)
    N = len(variables)
    var_index = {v: i for i, v in enumerate(variables)}
    K_list = []
    for var in variables:
        if var not in expressions:
            K_list.append(1)
            continue
        expr = expressions[var]
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
        inputs = set()
        for t in tokens:
            if t in var_index:
                inputs.add(t)
        K_list.append(max(len(inputs), 1))
    return N, K_list

def make_random_network(N, K_list, seed):
    """生成随机布尔网络，K 分布匹配"""
    np.random.seed(seed)
    max_K = max(K_list) if K_list else 1
    input_indices = np.full((N, max_K), -1, dtype=np.int32)
    input_K = np.array(K_list, dtype=np.int32)
    tables_list = []
    for i in range(N):
        K = K_list[i]
        # 随机选 K 个输入
        chosen = np.random.choice(N, size=min(K, N), replace=False)
        for k, c in enumerate(chosen):
            input_indices[i, k] = c
        # 随机真值表
        table = np.random.randint(0, 2, size=2**K).astype(np.int32)
        tables_list.append(table)
    table_offsets = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        table_offsets[i+1] = table_offsets[i] + len(tables_list[i])
    tables = np.concatenate(tables_list)
    return input_indices, input_K, table_offsets, tables

def compute_D_f_from_arrays(N, input_indices, input_K, table_offsets, tables, max_steps=500):
    attractors = compute_all_attractors(N, input_indices, input_K, table_offsets, tables, max_steps)
    n_states = 1 << N
    counts = Counter(attractors.tolist())
    H_init = math.log2(n_states)
    H_cond = 0.0
    for count in counts.values():
        p = count / n_states
        H_cond += p * math.log2(count)
    return H_cond / H_init if H_init > 0 else 0.0

def compute_real_D_f(filepath):
    variables, expressions = parse_bnet(filepath)
    N = len(variables)
    if N > 16 or N == 0:
        return None
    input_indices, input_K, table_offsets, tables = compile_rules(variables, expressions)
    D_f = compute_D_f_from_arrays(N, input_indices, input_K, table_offsets, tables)
    return D_f, N, input_K.tolist()

if __name__ == "__main__":
    bbm_root = "bbm/models"
    bnet_files = []
    for root, dirs, files in os.walk(bbm_root):
        for f in files:
            if f.endswith('.bnet'):
                bnet_files.append(os.path.join(root, f))

    print(f"Total .bnet files: {len(bnet_files)}")
    print("Precompiling numba...", flush=True)
    _ = compute_all_attractors(3, np.zeros((3,1), dtype=np.int32),
                                np.ones(3, dtype=np.int32),
                                np.array([0,1,2,3], dtype=np.int32),
                                np.array([0,1,0,1,0,1], dtype=np.int32), 50)
    print("Compiled.")
    print()

    results = []
    n_random_per_real = 5

    for filepath in bnet_files:
        try:
            r = compute_real_D_f(filepath)
            if r is None:
                continue
            D_f_real, N, K_list = r
        except Exception:
            continue

        # 生成随机对照
        D_f_randoms = []
        for seed in range(n_random_per_real):
            try:
                ii, iK, to, tb = make_random_network(N, K_list, seed=seed*13+1)
                D_f_r = compute_D_f_from_arrays(N, ii, iK, to, tb)
                D_f_randoms.append(D_f_r)
            except Exception:
                pass

        if not D_f_randoms:
            continue

        D_f_rand_mean = float(np.mean(D_f_randoms))
        delta = D_f_real - D_f_rand_mean

        model_name = os.path.basename(os.path.dirname(filepath))
        print(f"N={N:>3} | real={D_f_real:.4f} | random={D_f_rand_mean:.4f} | delta={delta:+.4f} | {model_name[:50]}", flush=True)

        results.append({
            "model": model_name,
            "N": N,
            "D_f_real": float(D_f_real),
            "D_f_random_mean": D_f_rand_mean,
            "D_f_random_list": [float(x) for x in D_f_randoms],
            "delta": float(delta)
        })

    print()
    print(f"Total paired models: {len(results)}")

    if results:
        deltas = [r["delta"] for r in results]
        print(f"Delta mean:   {np.mean(deltas):+.4f}")
        print(f"Delta median: {np.median(deltas):+.4f}")
        print(f"Delta range:  {min(deltas):+.4f} to {max(deltas):+.4f}")
        n_positive = sum(1 for d in deltas if d > 0)
        print(f"Real > Random: {n_positive} / {len(deltas)}")

    with open("bbm_compare_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to bbm_compare_result.json")
