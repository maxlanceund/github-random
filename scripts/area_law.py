import numpy as np, json

def kop(op,k,N):
    out = np.array([[1.0]])
    for i in range(N):
        out = np.kron(out, op if i==k else np.eye(2))
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
    e = e[e>1e-12]
    return -np.sum(e*np.log2(e))

N = 12
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])

def build_H(h):
    H = np.zeros((2**N,2**N), dtype=complex)
    for i in range(N-1):
        H -= 1.0 * kop(Z,i,N) @ kop(Z,i+1,N)
    for i in range(N):
        H -= h * kop(X,i,N)
    return H

results = {}
for h in [0.5, 1.0, 2.0]:
    H = build_H(h)
    evals, evecs = np.linalg.eigh(H)
    psi = evecs[:,0]
    rho = np.outer(psi, psi.conj())
    
    S_list = []
    for L in range(1, N//2+1):
        S = ent(ptrace(rho, list(range(L)), N))
        S_list.append(float(S))
    
    Ls = list(range(1, N//2+1))
    logLs = np.log(Ls)
    Ss = np.array(S_list)
    coef_log = np.polyfit(logLs, Ss, 1)
    coef_lin = np.polyfit(Ls, Ss, 1)
    
    pred_log = np.polyval(coef_log, logLs)
    pred_lin = np.polyval(coef_lin, Ls)
    res_log = float(np.sum((Ss - pred_log)**2))
    res_lin = float(np.sum((Ss - pred_lin)**2))
    
    results[str(h)] = {
        "L": Ls, "S": S_list,
        "log_slope": float(coef_log[0]),
        "log_intercept": float(coef_log[1]),
        "lin_slope": float(coef_lin[0]),
        "residual_log": res_log,
        "residual_lin": res_lin,
        "better_fit": "log" if res_log < res_lin else "linear"
    }
    
    print(f"\n=== h = {h} ===")
    print(f"{'L':>3} | {'S(L)':>10}")
    for L, S in zip(Ls, S_list):
        print(f"{L:>3} | {S:>10.4f}")
    print(f"  log 拟合: S = {coef_log[0]:.4f}*log(L) + {coef_log[1]:.4f}, 残差 {res_log:.6f}")
    print(f"  linear 拟合: S = {coef_lin[0]:.4f}*L + {coef_lin[1]:.4f}, 残差 {res_lin:.6f}")
    print(f"  更好拟合: {results[str(h)]['better_fit']}")

json.dump(results, open("area_law_result.json","w"), indent=2)
print("\n保存到 area_law_result.json")
