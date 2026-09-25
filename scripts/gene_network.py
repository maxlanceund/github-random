import numpy as np
from numba import njit
import math
from collections import Counter
import json

@njit
def random_boolean_network(N, K, seed):
    np.random.seed(seed)
    inputs = np.zeros((N, K), dtype=np.int32)
    tables = np.zeros((N, 2**K), dtype=np.int32)
    for i in range(N):
        for k in range(K):
            inputs[i, k] = np.random.randint(0, N)
        for j in range(2**K):
            tables[i, j] = np.random.randint(0, 2)
    return inputs, tables

@njit
def evolve_to_steady_state(inputs, tables, N, K, init_state, max_steps):
    state = init_state.copy()
    for step in range(max_steps):
        new_state = np.zeros(N, dtype=np.int32)
        for i in range(N):
            idx = 0
            for k in range(K):
                idx = idx * 2 + state[inputs[i, k]]
            new_state[i] = tables[i, idx]
        same = True
        for i in range(N):
            if new_state[i] != state[i]:
                same = False
                break
        if same:
            return new_state
        state = new_state
    return state

@njit
def compute_phenotype(inputs, tables, N, K, max_steps, seed):
    np.random.seed(seed)
    init = np.zeros(N, dtype=np.int32)
    for i in range(N):
        init[i] = np.random.randint(0, 2)
    return evolve_to_steady_state(inputs, tables, N, K, init, max_steps)

def phenotype_to_int(phenotype):
    val = 0
    for b in phenotype:
        val = val * 2 + int(b)
    return val

if __name__ == "__main__":
    print("=" * 70)
    print("Gene network D_f: with N_net = 10 * 2^N")
    print("=" * 70)
    print()

    max_steps = 100

    results = []
    for N in [6, 8, 10, 12]:
        for K in [2, 3]:
            N_net = 10 * (2 ** N)
            phenotype_counts = Counter()
            for net_id in range(N_net):
                inputs, tables = random_boolean_network(N, K, seed=net_id)
                pheno = compute_phenotype(inputs, tables, N, K, max_steps, seed=net_id*7+3)
                pheno_int = phenotype_to_int(pheno)
                phenotype_counts[pheno_int] += 1

            H_omega = math.log2(N_net)
            H_cond = 0.0
            for count in phenotype_counts.values():
                p = count / N_net
                H_cond += p * math.log2(count)
            D_f = H_cond / H_omega if H_omega > 0 else 0.0

            n_unique = len(phenotype_counts)
            max_group = max(phenotype_counts.values())

            print(f"N={N:>2}, K={K}, N_net={N_net:>6}: D_f={D_f:.4f}, unique={n_unique}, max={max_group}", flush=True)
            results.append({"N": N, "K": K, "N_net": N_net, "D_f": D_f,
                           "n_unique": n_unique, "max_group": max_group})

    with open("gene_network_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Saved to gene_network_result.json")
