import numpy as np, json
from scipy.linalg import expm

def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)

I = np.eye(2)
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

def H_AB(j):
    return j * kron3(Z, Z, I)

def H_AC(j):
    return j * kron3(Z, I, Z)

def H_field(h):
    return h * (kron3(X, I, I) + kron3(I, X, I) + kron3(I, I, X))

def ptrace(rho, keep, N=3):
    keep = sorted(keep)
    tr = [i for i in range(N) if i not in keep]
    rt = rho.reshape([2]*(2*N))
    for i in sorted(tr, reverse=True):
        rt = np.trace(rt, axis1=i, axis2=i+N)
        N -= 1
    return rt.reshape(2**len(keep), -1)

def purity(rho):
    return float(np.trace(rho @ rho).real)

def trace_distance(r1, r2):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(r1 - r2))))

def conditional_state_B(rho_AB, A_outcome):
    # 投影 A 到 |A_outcome>，得到 B 的条件态
    P = np.zeros((4,4))
    idx = A_outcome * 2  # |0> -> 0, |1> -> 1 (A 是第一个比特)
    P[idx, idx] = 1
    # rho_AB 是 4x4 (A,B)
    proj = np.kron(np.diag([1,0]) if A_outcome==0 else np.diag([0,1]), I)
    cond = proj @ rho_AB @ proj
    p = np.trace(cond).real
    if p < 1e-10:
        return None, 0.0
    return cond / p, p

results = []

print("=" * 70)
print("观察者相对性模拟")
print("=" * 70)
print(f"\n{'lambda':>8} | {'purity_B|A':>12} | {'purity_B|C':>12} | {'TD(B|A,B|C)':>12}")
print("-" * 70)

# 初态: A 和 B 强关联
# |psi0> = (|00>_AB + |11>_AB)/sqrt(2) ⊗ |0>_C
psi0 = np.zeros(8, dtype=complex)
psi0[0] = 1/np.sqrt(2)   # |000>
psi0[3] = 1/np.sqrt(2)   # |011>  (A=0,B=1,C=1) 不对
# 重新构造: A=0,B=0,C=0 和 A=1,B=1,C=0
psi0 = np.zeros(8, dtype=complex)
psi0[0b000] = 1/np.sqrt(2)
psi0[0b110] = 1/np.sqrt(2)

for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    H = H_AB(1.0) + H_AC(lam) + H_field(0.5)
    U = expm(-1j * H * 1.0)
    psi = U @ psi0
    rho = np.outer(psi, psi.conj())
    
    # A 和 B 的联合态（追踪 C）
    rho_AB = ptrace(rho, [0, 1])
    # C 和 B 的联合态（追踪 A）
    rho_CB = ptrace(rho, [2, 1])
    
    # A 的两个结果下的 B 条件态
    purA, purC = [], []
    condA, condC = [], []
    for outcome in [0, 1]:
        cA, pA = conditional_state_B(rho_AB, outcome)
        if cA is not None:
            purA.append(purity(cA))
            condA.append(cA)
        cC, pC = conditional_state_B(rho_CB, outcome)
        if cC is not None:
            purC.append(purity(cC))
            condC.append(cC)
    
    # 取 A=0 和 C=0 下的条件态，比较
    if len(condA) > 0 and len(condC) > 0:
        td = trace_distance(condA[0], condC[0])
    else:
        td = float('nan')
    
    pA_mean = float(np.mean(purA)) if purA else float('nan')
    pC_mean = float(np.mean(purC)) if purC else float('nan')
    
    print(f"{lam:>8.2f} | {pA_mean:>12.4f} | {pC_mean:>12.4f} | {td:>12.4f}")
    
    results.append({
        "lambda": lam,
        "purity_B_given_A": pA_mean,
        "purity_B_given_C": pC_mean,
        "trace_distance": td,
    })

json.dump({"results": results}, open("observer_relativity_result.json","w"), indent=2)

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
tds = [r["trace_distance"] for r in results if not np.isnan(r["trace_distance"])]
print(f"  λ=0:      TD = {tds[0]:.4f}（应≈0，A和C看到相同的B）")
print(f"  λ=2.0:    TD = {tds[-1]:.4f}（应>0，A和C看到不同的B）")
print(f"  TD 单调递增: {all(tds[i] <= tds[i+1] + 1e-6 for i in range(len(tds)-1))}")
print(f"\n  如果 TD 随 λ 增大: RQM 的观察者相对性成立")
print(f"  如果 TD 恒为 0:   RQM 有问题")

print("\n保存到 observer_relativity_result.json")
