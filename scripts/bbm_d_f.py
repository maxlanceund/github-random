import numpy as np
from numba import njit
import re
import os
import json
import math
from collections import Counter

def to_python_expr(expr):
    expr = re.sub(r'!', ' not ', expr)
    expr = re.sub(r'&', ' and ', expr)
    expr = re.sub(r'\|', ' or ', expr)
    expr = re.sub(r'\s+', ' ', expr)
    return expr.strip()

def parse_bnet(filepath):
    variables = []
    expressions = {}
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    if lines and lines[0].startswith('targets'):
        lines = lines[1:]
    for line in lines:
        if ',' not in line:
            continue
        target, expr = line.split(',', 1)
        target = target.strip()
        expr = expr.strip()
        if target not in variables:
            variables.append(target)
        expressions[target] = expr
    return variables, expressions

def compile_rules(variables, expressions):
    N = len(variables)
    var_index = {v: i for i, v in enumerate(variables)}
    input_indices_list = []
    input_K_list = []
    tables_list = []
    for i, var in enumerate(variables):
        if var not in expressions:
            input_indices_list.append([i])
            input_K_list.append(1)
            tables_list.append(np.array([0, 1], dtype=np.int32))
            continue
        expr = expressions[var]
        py_expr = to_python_expr(expr)
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
        input_vars = []
        seen = set()
        for t in tokens:
            if t in var_index and t not in seen:
                seen.add(t)
                input_vars.append(t)
        K = len(input_vars)
        if K == 0:
            result = eval(py_expr, {"__builtins__": {}}, {})
            input_indices_list.append([])
            input_K_list.append(0)
            tables_list.append(np.array([int(bool(result))], dtype=np.int32))
            continue
        table = np.zeros(2**K, dtype=np.int32)
        for j in range(2**K):
            state = {}
            for k, v in enumerate(input_vars):
                state[v] = (j >> (K-1-k)) & 1
            result = eval(py_expr, {"__builtins__": {}}, state)
            table[j] = int(bool(result))
        idxs = [var_index[v] for v in input_vars]
        input_indices_list.append(idxs)
        input_K_list.append(K)
        tables_list.append(table)
    max_K = max(input_K_list) if input_K_list else 1
    max_K = max(max_K, 1)
    input_indices = np.full((N, max_K), -1, dtype=np.int32)
    for i, idxs in enumerate(input_indices_list):
        for k, idx in enumerate(idxs):
            input_indices[i, k] = idx
    input_K = np.array(input_K_list, dtype=np.int32)
    table_offsets = np.zeros(N + 1, dtype=np.int32)
    for i in range(N):
        table_offsets[i+1] = table_offsets[i] + len(tables_list[i])
    tables = np.concatenate(tables_list) if tables_list else np.array([], dtype=np.int32)
    return input_indices, input_K, table_offsets, tables

@njit
def compute_next_state(current, N, input_indices, input_K, table_offsets, tables):
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
def compute_all_attractors(N, input_indices, input_K, table_offsets, tables, max_steps):
    n_states = 1 << N
    visit_time = np.zeros(n_states, dtype=np.int32)
    attractors = np.zeros(n_states, dtype=np.int64)
    for initial in range(n_states):
        current_time = initial + 1
        current = initial
        attractor = -1
        for step in range(max_steps):
            if visit_time[current] == current_time:
                min_state = current
                nxt = compute_next_state(current, N, input_indices, input_K, table_offsets, tables)
                while nxt != current:
                    if nxt < min_state:
                        min_state = nxt
                    nxt = compute_next_state(nxt, N, input_indices, input_K, table_offsets, tables)
                attractor = min_state
                break
            visit_time[current] = current_time
            new = compute_next_state(current, N, input_indices, input_K, table_offsets, tables)
            if new == current:
                attractor = current
                break
            current = new
        if attractor < 0:
            attractor = current
        attractors[initial] = attractor
    return attractors

def compute_D_f(filepath, max_N=16, max_steps=500):
    variables, expressions = parse_bnet(filepath)
    N = len(variables)
    if N > max_N or N == 0:
        return None
    try:
        input_indices, input_K, table_offsets, tables = compile_rules(variables, expressions)
    except Exception:
        return None
    attractors = compute_all_attractors(N, input_indices, input_K, table_offsets, tables, max_steps)
    n_states = 1 << N
    counts = Counter(attractors.tolist())
    H_init = math.log2(n_states)
    H_cond = 0.0
    for count in counts.values():
        p = count / n_states
        H_cond += p * math.log2(count)
    D_f = H_cond / H_init if H_init > 0 else 0.0
    return {
        "N": N,
        "n_states": n_states,
        "n_attractors": len(counts),
        "max_basin": max(counts.values()),
        "D_f": D_f
    }

if __name__ == "__main__":
    bbm_root = "bbm/models"
    bnet_files = []
    for root, dirs, files in os.walk(bbm_root):
        for f in files:
            if f.endswith('.bnet'):
                bnet_files.append(os.path.join(root, f))
    print(f"Total .bnet files: {len(bnet_files)}")
    print()
    results = []
    for filepath in bnet_files:
        try:
            r = compute_D_f(filepath)
            if r is not None:
                model_name = os.path.basename(os.path.dirname(filepath))
                r["model"] = model_name
                results.append(r)
                print(f"N={r['N']:>3} | D_f={r['D_f']:.4f} | attractors={r['n_attractors']:>5} | {model_name[:55]}", flush=True)
        except Exception:
            pass
    print()
    print(f"Total models processed: {len(results)}")
    with open("bbm_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to bbm_result.json")
