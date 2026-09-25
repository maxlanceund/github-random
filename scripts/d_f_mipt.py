import numpy as np
from numba import njit
import math
import json

@njit
def apply_cnot(psi, L, control, target):
    """施加 CNOT 门"""
    mc = 1 << (L-1-control)
    mt = 1 << (L-1-target)
    for idx in range(2**L):
        if (idx & mc) and (idx & mt) == 0:
            idx2 = idx | mt
            a = psi[idx]
            psi[idx] = psi[idx2]
            psi[idx2] = a

@njit
def apply_hadamard(psi, L, site):
    """施加单比特 H 门"""
    m = 1 << (L-1-site)
    for idx in range(2**L):
        if (idx & m) == 0:
            idx2 = idx | m
            a = psi[idx]
            b = psi[idx2]
            psi[idx] = (a + b) / 1.4142135623730951
            psi[idx2] = (a - b) / 1.4142135623730951

@njit
def measure_z(psi, L, site):
    """Z 基测量，坍缩态"""
    m = 1 << (L-1-site)
    prob0 = 0.0
    for idx in range(2**L):
        if (idx & m) == 0:
            prob0 += psi[idx].real**2 + psi[idx].imag**2
    r = np.random.random()
    if r < prob0:
        norm = math.sqrt(prob0 + 1e-30)
        for idx in range(2**L):
            if (idx & m) == 0:
                psi[idx] /= norm
            else:
                psi[idx] = 0.0
    else:
        norm = math.sqrt(1.0 - prob0 + 1e-30)
        for idx in range(2**L):
            if (idx & m) != 0:
                psi[idx] /= norm
            else:
                psi[idx] = 0.0

@njit
def evolve_step(psi, L, gamma):
    """一个时间步：随机两比特门 + 随机测量"""
    # 两比特门层
    for i in range(L-1):
        apply_cnot(psi, L, i, i+1)
    # H 层
    for i in range(L):
        if np.random.random() < 0.5:
            apply_hadamard(psi, L, i)
    # 测量层
    for i in range(L):
        if np.random.random() < gamma:
            measure_z(psi, L, i)

@njit
def compute_D_f(psi, L, block_size):
    """计算信息损失率：宏观描述 = 块磁化"""
    n_blocks = L // block_size
    # 计算每个块的总磁化 Z_i
    # 只对块之间的联合分布采样
    # 简化：用块磁化的边缘分布近似
    probs = np.zeros(2**L)
    for idx in range(2**L):
        probs[idx] = psi[idx].real**2 + psi[idx].imag**2

    # 微观熵
    H_micro = 0.0
    for idx in range(2**L):
        p = probs[idx]
        if p > 1e-15:
            H_micro -= p * math.log2(p)

    # 宏观熵：块磁化的联合分布
    macro_probs = np.zeros(2**n_blocks)
    for idx in range(2**L):
        # 每个块，统计 1 的个数的奇偶
        macro_idx = 0
        for b in range(n_blocks):
            count = 0
            for j in range(block_size):
                site = b * block_size + j
                bit = (idx >> (L-1-site)) & 1
                count += bit
            # 奇偶性作为宏观变量
            if count % 2 == 1:
                macro_idx |= (1 << (n_blocks-1-b))
        macro_probs[macro_idx] += probs[idx]

    H_macro = 0.0
    for p in macro_probs:
        if p > 1e-15:
            H_macro -= p * math.log2(p)

    if H_micro > 0:
        return (H_micro - H_macro) / H_micro
    return 0.0

def run(L, gamma, n_steps, n_runs):
    """跑一次实验，返回平均 D_f"""
    Df_list = []
    for run in range(n_runs):
        psi = np.zeros(2**L, dtype=np.complex128)
        psi[0] = 1.0  # 从 |0...0> 开始
        # 先加一点纠缠：H 所有比特
        for i in range(L):
            apply_hadamard(psi, L, i)
        # 演化
        for step in range(n_steps):
            evolve_step(psi, L, gamma)
        Df = compute_D_f(psi, L, block_size=2)
        Df_list.append(Df)
    return float(np.mean(Df_list)), float(np.std(Df_list))

if __name__ == "__main__":
    L = 10
    n_steps = 30
    n_runs = 20

    print("=" * 70)
    print(f"MIPT 探测：D_f vs 测量率 γ (L={L})")
    print("=" * 70)
    print()
    print(f"{'gamma':>8} | {'D_f 平均':>12} | {'标准差':>10}")
    print("-" * 45)

    results = []
    for gamma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        mean, std = run(L, gamma, n_steps, n_runs)
        print(f"{gamma:>8.3f} | {mean:>12.6f} | {std:>10.6f}", flush=True)
        results.append({"gamma": gamma, "D_f_mean": mean, "D_f_std": std})

    with open("d_f_mipt_result.json", "w") as f:
        json.dump({"L": L, "n_steps": n_steps, "results": results}, f, indent=2)
    print()
    print("结果已保存到 d_f_mipt_result.json")
