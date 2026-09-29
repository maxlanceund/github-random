import numpy as np, json
from scipy.linalg import expm

I = np.eye(2)
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

def kron3(a,b,c):
    return np.kron(np.kron(a,b), c)

def H(J_A, J_C, h):
    Hm = J_A * kron3(Z, Z, I)
    Hm += J_C * kron3(I, Z, Z)
    Hm += h * (kron3(X, I, I) + kron3(I, X, I) + kron3(I, I, X))
    return Hm

def ptrace(rho, keep, N=3):
    keep = sorted(keep)
    tr = [i for i in range(N) if i not in keep]
    rt = rho.reshape([2]*(2*N))
    for i in sorted(tr, reverse=True):
        rt = np.trace(rt, axis1=i, axis2=i+N)
        N -= 1
    return rt.reshape(2**len(keep), -1)

def cond_B(rho_AB, val):
    P = np.kron(np.diag([1,0]) if val==0 else np.diag([0,1]), I)
    cond = P @ rho_AB @ P
    p = np.trace(cond).real
    if p < 1e-10:
        return None
    return cond / p

def purity(rho):
    return float(np.trace(rho @ rho).real)

def td(r1, r2):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(r1 - r2))))

# 初态：A=|+>, B=|0>, C=|+>
pA = np.array([1,1])/np.sqrt(2)
pB = np.array([1,0])
pC = np.array([1,1])/np.sqrt(2)
psi0 = np.kron(np.kron(pA, pB), pC)

results = []
print("=" * 60)
print("观察者相对性（修正版）")
print("=" * 60)
print(f"初态：A=|+>, B=|0>, C=|+>")
print(f"J_A = 1.0 固定，扫描 J_C")
print(f"\n{'J_C':>6} | {'pur(B|A=0)':>12} | {'pur(B|C=0)':>12} | {'TD':>10}")
print("-" * 55)

for J_C in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    Hm = H(1.0, J_C, 0.5)
    U = expm(-1j * Hm * 1.0)
    psi = U @ psi0
    rho = np.outer(psi, psi.conj())
    
    rho_AB = ptrace(rho, [0,1])
    rho_CB = ptrace(rho, [2,1])
    
    cA = cond_B(rho_AB, 0)
    cC = cond_B(rho_CB, 0)
    
    pA_v = purity(cA) if cA is not None else float('nan')
    pC_v = purity(cC) if cC is not None else float('nan')
    dist = td(cA, cC) if (cA is not None and cC is not None) else float('nan')
    
    print(f"{J_C:>6.2f} | {pA_v:>12.4f} | {pC_v:>12.4f} | {dist:>10.4f}")
    results.append({
        "J_C": J_C, "pur_A": pA_v, "pur_C": pC_v, "TD": dist
    })

json.dump({"results": results}, open("observer_relativity_result.json","w"), indent=2)

print("\n" + "=" * 60)
print("结论:")
print("=" * 60)
tds = [r["TD"] for r in results if not np.isnan(r["TD"])]
print(f"  TD 范围: {min(tds):.4f} 到 {max(tds):.4f}")
print(f"  TD 最小值在 J_C = {results[tds.index(min(tds))]['J_C']:.2f}")
print(f"\n  如果 TD 在 J_C = J_A 时最小 → 观察者相对性成立")
print(f"  如果 TD 单调 → 需要检查公式")
