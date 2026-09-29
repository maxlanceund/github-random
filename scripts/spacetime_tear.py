import numpy as np
import json

def kop(op, k, N):
    out = np.array([[1.0]])
    for i in range(N):
        out = np.kron(out, op if i == k else np.eye(2))
    return out

def ptrace(rho, keep, n):
    keep = sorted(keep)
    tr = [i for i in range(n) if i not in keep]
    rt = rho.reshape([2]*(2*n))
    for i in sorted(tr, reverse=True):
        rt = np.trace(rt, axis1=i, axis2=i+n)
        n -= 1
    return rt.reshape(2**len(keep), -1)

def ent(rho):
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-12]
    return -np.sum(e * np.log2(e))

def mi(rho, A, B, N):
    return ent(ptrace(rho,A,N)) + ent(ptrace(rho,B,N)) - ent(ptrace(rho,A+B,N))

N = 8
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
LEFT = [0,1,2,3]
RIGHT = [4,5,6,7]

results = []

print("=" * 70)
print("空间撕裂模拟：切断中间纠缠")
print("=" * 70)
print(f"\n{'lambda':>8} | {'S(左)':>8} | {'S(右)':>8} | {'S(左∪右)':>10} | {'MI(左:右)':>10} | {'距离':>10}")
print("-" * 70)

for lam in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001, 0.0]:
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N-1):
        J = 1.0 if i != 3 else lam
        H -= J * kop(Z,i,N) @ kop(Z,i+1,N)
    for i in range(N):
        H -= 1.0 * kop(X,i,N)
    
    evals, evecs = np.linalg.eigh(H)
    psi = evecs[:, 0]
    rho = np.outer(psi, psi.conj())
    
    S_L = ent(ptrace(rho, LEFT, N))
    S_R = ent(ptrace(rho, RIGHT, N))
    S_LR = ent(ptrace(rho, LEFT+RIGHT, N))
    MI = S_L + S_R - S_LR
    
    d = np.log(1.0 / (MI + 1e-10)) if MI > 1e-10 else 100.0
    
    results.append({
        "lambda": lam,
        "S_left": float(S_L),
        "S_right": float(S_R),
        "S_joint": float(S_LR),
        "MI_LR": float(MI),
        "distance": float(d)
    })
    
    print(f"{lam:>8.3f} | {S_L:>8.4f} | {S_R:>8.4f} | {S_LR:>10.4f} | {MI:>10.4f} | {d:>10.4f}")

# 检查单调性
mis = [r["MI_LR"] for r in results]
monotone = all(mis[i] >= mis[i+1] for i in range(len(mis)-1))

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
print(f"  1. λ 从 1 降到 0，中间耦合减弱")
print(f"  2. MI(左:右) 从 {mis[0]:.4f} 降到 {mis[-1]:.4f}")
print(f"  3. MI 单调递减: {monotone}")
print(f"  4. 距离从 {results[0]['distance']:.2f} 升到 {results[-1]['distance']:.2f}")
if mis[-1] < 1e-6:
    print(f"  5. λ=0 时 MI ≈ 0 → 左右完全断开 → 空间撕裂")
else:
    print(f"  5. λ=0 时 MI = {mis[-1]:.6f}，未完全断开")

out = {"results": results, "monotone": monotone}
json.dump(out, open("spacetime_tear_result.json","w"), indent=2)
print("\n保存到 spacetime_tear_result.json")
