import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import expm_multiply
import json

# 泡利矩阵
sx = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sy = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
sz = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
I2 = identity(2, format='csr')

def op_at(op, site, L):
    """在指定位置放算符，其他地方放单位矩阵"""
    result = csr_matrix(np.array([[1.0+0j]]))
    for i in range(L):
        result = kron(result, op if i == site else I2, format='csr')
    return result

def heisenberg_hamiltonian(L, J=1.0, W=0.0, seed=42):
    """海森堡链哈密顿量"""
    np.random.seed(seed)
    dim = 2**L
    H = csr_matrix((dim, dim), dtype=complex)
    
    # 最近邻相互作用
    for i in range(L - 1):
        H += J * (op_at(sx, i, L) @ op_at(sx, i+1, L) +
                  op_at(sy, i, L) @ op_at(sy, i+1, L) +
                  op_at(sz, i, L) @ op_at(sz, i+1, L))
    
    # 随机场
    if W > 0:
        fields = np.random.uniform(-W, W, L)
        for i in range(L):
            H += fields[i] * op_at(sz, i, L)
    
    return H

def entanglement_entropy(psi, L, subsystem_size):
    """半链纠缠熵"""
    psi_matrix = psi.reshape(2**subsystem_size, 2**(L - subsystem_size))
    _, s, _ = np.linalg.svd(psi_matrix, full_matrices=False)
    p = s**2
    p = p[p > 1e-15]
    return -np.sum(p * np.log2(p))

def run_simulation(L, J, W, t_max, n_steps):
    """跑一次模拟"""
    H = heisenberg_hamiltonian(L, J=J, W=W)
    
    # 初始态：所有自旋向上
    psi0 = np.zeros(2**L, dtype=complex)
    # Néel 态：交替自旋
    neel_index = sum(2**(L-1-i) for i in range(0, L, 2))
    psi0[neel_index] = 1.0
    
    times = np.linspace(0, t_max, n_steps)
    entropies = []
    
    for t in times:
        psi_t = expm_multiply(-1j * H * t, psi0)
        S = entanglement_entropy(psi_t, L, L // 2)
        entropies.append(float(S))
    
    return times.tolist(), entropies

if __name__ == "__main__":
    L = 10
    J = 1.0
    t_max = 20.0
    n_steps = 40
    
    results = {}
    
    print("=" * 60)
    print(f"量子混沌模拟：海森堡链 L={L}")
    print("=" * 60)
    print()
    
    for W in [0.0, 0.5, 1.0, 2.0, 5.0]:
        print(f"W = {W} ...", flush=True)
        times, entropies = run_simulation(L, J, W, t_max, n_steps)
        results[f"W={W}"] = {"times": times, "entropies": entropies}
        
        # 打印几个关键点
        print(f"  S(t=5)  = {entropies[n_steps//8]:.4f}")
        print(f"  S(t=10) = {entropies[n_steps//4]:.4f}")
        print(f"  S(t=20) = {entropies[-1]:.4f}")
        print()
    
    # 保存结果
    with open("quantum_chaos_result.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("结果已保存到 quantum_chaos_result.json")
