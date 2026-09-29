import numpy as np, json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def build_H_pbc(N, h):
    dim = 2**N
    states = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim)
    for i in range(N):
        j = (i+1) % N
        b_i = (states >> i) & 1
        b_j = (states >> j) & 1
        diag -= (1-2*b_i) * (1-2*b_j)
    rows = np.repeat(states, N)
    cols = np.zeros_like(rows)
    for i in range(N):
        cols[i::N] = states ^ (1 << i)
    vals = np.full(len(rows), -h, dtype=float)
    H = csr_matrix((diag, (states, states)), shape=(dim,dim))
    H += csr_matrix((vals, (rows, cols)), shape=(dim,dim))
    return H

def ent_region(psi, region, N):
    L = len(region)
    rest = [i for i in range(N) if i not in region]
    psi_t = psi.reshape([2]*N)
    perm = region + rest
    psi_t = np.transpose(psi_t, perm)
    psi_mat = psi_t.reshape(2**L, 2**(N-L))
    rho = psi_mat @ psi_mat.conj().T
    e = np.linalg.eigvalsh(rho)
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

results = {}
for N in [14, 16, 18, 20]:
    print(f"\n=== N = {N} ===")
    H = build_H_pbc(N, 1.0)
    val, vec = eigsh(H, k=1, which='SA')
    psi = vec[:, 0]
    
    lls, MIs = [], []
    for L in range(1, N//4 + 1):
        A = list(range(L))
        B = list(range(L, 2*L))
        S_A = ent_region(psi, A, N)
        S_B = ent_region(psi, B, N)
        S_AB = ent_region(psi, A+B, N)
        MI = S_A + S_B - S_AB
        lls.append(L)
        MIs.append(MI)
        print(f"  L={L:>2}  S(A)={S_A:.4f}  S(B)={S_B:.4f}  S(AB)={S_AB:.4f}  I(A:B)={MI:.4f}")
    
    lls_arr = np.array(lls, dtype=float)
    MIs_arr = np.array(MIs)
    logL = np.log(lls_arr)
    A_mat = np.vstack([logL, np.ones_like(logL)]).T
    coef, _, _, _ = np.linalg.lstsq(A_mat, MIs_arr, rcond=None)
    slope, intercept = coef
    c_fit = 3 * slope
    pred = A_mat @ coef
    resid = float(np.sum((MIs_arr - pred)**2))
    
    print(f"  拟合: I(L) = {slope:.4f} * ln(L) + {intercept:.4f}")
    print(f"  斜率 = {slope:.4f}, c = 3*斜率 = {c_fit:.4f}, 残差 = {resid:.6f}")
    
    results[str(N)] = {
        "L": lls, "MI": MIs,
        "slope": float(slope),
        "c_from_MI": float(c_fit),
        "residual": resid
    }
    json.dump(results, open("mutual_info_scan_result.json","w"), indent=2)

print(f"\n=== c 从互信息提取 ===")
for N in [14,16,18,20]:
    print(f"  N={N:>2}:  c = {results[str(N)]['c_from_MI']:.4f}")
print(f"\nCFT 预言: c → 0.5")
print("\n保存到 mutual_info_scan_result.json")
