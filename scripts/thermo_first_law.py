import numpy as np
from numpy.linalg import eigvalsh
import json

def build_xx_hamiltonian(L, J=1.0):
    h = np.zeros((L, L))
    for i in range(L-1):
        h[i, i+1] = -J/2
        h[i+1, i] = -J/2
    h[0, L-1] = -J/2
    h[L-1, 0] = -J/2
    return h

def thermal_correlation(L, T, J=1.0):
    h = build_xx_hamiltonian(L, J)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    f = 1.0 / (np.exp(eigenvalues / T) + 1.0)
    C = eigenvectors @ np.diag(f) @ eigenvectors.conj().T
    return np.real(C), h

def entanglement_entropy(C_A):
    lambdas = eigvalsh(C_A)
    lambdas = np.clip(lambdas, 1e-15, 1 - 1e-15)
    S = -np.sum(lambdas * np.log(lambdas) + (1-lambdas) * np.log(1-lambdas))
    return float(S)

def subsystem_energy(C_A, h_A):
    return float(np.real(np.trace(C_A @ h_A)))

def run(L, L_A, temperatures):
    results = []
    for T in temperatures:
        C, h = thermal_correlation(L, T)
        C_A = C[:L_A, :L_A]
        h_A = h[:L_A, :L_A]
        S_A = entanglement_entropy(C_A)
        E_A = subsystem_energy(C_A, h_A)
        results.append({"T": T, "S_A": S_A, "E_A": E_A})
    return results

def compute_dS_dE(results):
    dS_dE = []
    for i in range(len(results)):
        if i == 0:
            d = (results[1]["S_A"] - results[0]["S_A"]) / (results[1]["E_A"] - results[0]["E_A"])
        elif i == len(results) - 1:
            d = (results[-1]["S_A"] - results[-2]["S_A"]) / (results[-1]["E_A"] - results[-2]["E_A"])
        else:
            d = (results[i+1]["S_A"] - results[i-1]["S_A"]) / (results[i+1]["E_A"] - results[i-1]["E_A"])
        dS_dE.append(d)
    return dS_dE

if __name__ == "__main__":
    L = 24
    L_A = 8
    temperatures = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    print("=" * 78)
    print(f"XX 链：热力学第一定律的信息论验证")
    print(f"L = {L}, L_A = {L_A}, 周期性边界条件")
    print("=" * 78)
    print()
    print(f"{'T':>6} | {'S_A':>10} | {'E_A':>14} | {'dS/dE':>12} | {'1/T':>8} | {'dS/dE * T':>12}")
    print("-" * 78)

    results = run(L, L_A, temperatures)
    dS_dE = compute_dS_dE(results)

    for i, r in enumerate(results):
        ratio = dS_dE[i] * r["T"]
        print(f"{r['T']:>6.2f} | {r['S_A']:>10.4f} | {r['E_A']:>14.6f} | {dS_dE[i]:>12.4f} | {1/r['T']:>8.4f} | {ratio:>12.4f}")

    print()
    print("=" * 78)
    print("解释：")
    print("  如果最后一列 ≈ 1 → 第一定律 dS = dE/T 成立")
    print("  如果最后一列 ≠ 1 → 第一定律在这个系统/参数范围内不成立")
    print("=" * 78)

    output = {"L": L, "L_A": L_A, "results": results, "dS_dE": dS_dE,
              "ratios": [dS_dE[i] * results[i]["T"] for i in range(len(results))]}
    with open("thermo_result.json", "w") as f:
        json.dump(output, f, indent=2)
    print("结果已保存到 thermo_result.json")
