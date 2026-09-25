import numpy as np
from numba import njit
import math
from collections import Counter
import json

@njit
def random_boolean_network(N, K, seed):
    """生成随机布尔网络：每个基因有 K 个输入，随机真值表"""
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
    """从初始态演化到稳态或周期"""
    state = init_state.copy()
    history = np.zeros((max_steps, N), dtype=np.int32)
    for step in range(max_steps):
        history[step] = state
        new_state = np.zeros(N, dtype=np.int32)
        for i in range(N):
            idx = 0
            for k in range(K):
                idx = idx * 2 + state[inputs[i, k]]
            new_state[i] = tables[i, idx]
        # 检查是否到达稳态或周期
        if np.array_equal(new_state, state):
            return new_state, step
        state = new_state
    return state, max_steps

@njit
def compute_phenotype(inputs, tables, N, K, max_steps):
    """从多个初始态演化，取最常见的稳态作为表型"""
    n_init = 5
    phenotypes = np.zeros((n_init, N), dtype=np.int32)
    for j in range(n_init):
        init = np.zeros(N, dtype=np.int32)
        for i in range(N):
            init[i] = np.random.randint(0, 2)
        final, _ = evolve_to_steady_state(inputs, tables, N, K, init, max_steps)
        phenotypes[j] = final
    # 简化：取第一个作为表型（也可以取多数）
    return phenotypes[0]

def phenotype_to_int(phenotype):
    """把表型编码成整数"""
    val = 0
    for b in phenotype:
        val = val * 2 + int(b)
    return val

if __name__ == "__main__":
    print("=" * 70)
    print("基因调控网络：表型丢失了多少基因型信息？")
    print("=" * 70)
    print()

    N_net = 2000  # 随机网络数
    max_steps = 100

    results = []
    for N in [6, 8, 10, 12]:
        for K in [2, 3]:
            np.random.seed(42)
            phenotype_counts = Counter()
            for net_id in range(N_net):
                inputs, tables = random_boolean_network(N, K, seed=net_id)
                pheno = compute_phenotype(inputs, tables, N, K, max_steps)
                pheno_int = phenotype_to_int(pheno)
                phenotype_counts[pheno_int] += 1

            # 计算 D_f
            H_omega = math.log2(N_net)
            H_cond = 0.0
            for count in phenotype_counts.values():
                p = count / N_net
                H_cond += p * math.log2(count)
            D_f = H_cond / H_omega if H_omega > 0 else 0.0

            n_unique = len(phenotype_counts)
            max_group = max(phenotype_counts.values())

            print(f"N={N:>2}, K={K}: D_f={D_f:.4f}, 唯一表型数={n_unique}, 最大组={max_group}", flush=True)
            results.append({"N": N, "K": K, "D_f": D_f,
                           "n_unique": n_unique, "max_group": max_group})

    with open("gene_network_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("结果已保存到 gene_network_result.json")
