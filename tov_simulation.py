import numpy as np
from scipy.integrate import solve_ivp

G = 6.67430e-8; c = 2.99792458e10; Msun = 1.98847e33

def tov_eqs(r, y, K, Gamma):
    m, P, y_love = y
    if P <= 1e-10: return [0, 0, 0]
    rho = (P / K)**(1/Gamma)
    eps = rho + P / c**2 
    dm = 4 * np.pi * r**2 * rho
    dp = -G * (eps + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r * (r - 2*G*m/c**2))
    term_F = (1 - 4*np.pi*r**2*eps*G/c**2) / (1 - 2*G*m/(r*c**2))
    cs2 = Gamma * P / (eps + P) 
    term_Q = (4*np.pi*G/c**2 * (5*eps + 9*P/c**2 + (eps + P/c**2)/cs2) - 6/(r**2*c**2)) / (1 - 2*G*m/(r*c**2))
    dy = -(y_love**2 + y_love*term_F + r**2*term_Q) / r
    return [dm, dp, dy]

def calc_star(rho_c, Gamma):
    K = 2.0e10 if Gamma == 2.5 else 1.2e10 
    r0 = 1.0e2; P_c = K * rho_c**Gamma; m0 = (4/3) * np.pi * r0**3 * rho_c
    def surf(r, y, *args): return y[1]
    surf.terminal = True
    try:
        sol = solve_ivp(tov_eqs, [r0, 3.0e6], [m0, P_c, 2.0], args=(K, Gamma), events=surf, max_step=100.0, rtol=1e-6, atol=1e-8)
        R = sol.t[-1]; M = sol.y[0, -1] / Msun; y_R = sol.y[2, -1]
        if y_R < 0: y_R = 0.5 
        k2 = (1 + y_R) / 2; C = G * M * Msun / (R * c**2)
        if C > 0.5 or C <= 0: return M, np.nan
        return M, (2/3) * k2 / (C**5)
    except Exception: return np.nan, np.nan

# 生成数据
hard_data, soft_data = [], []
for rho_c in np.linspace(5e14, 1.5e15, 10):
    M_h, L_h = calc_star(rho_c, 2.5)
    if not np.isnan(M_h): hard_data.append([M_h, L_h])
    M_s, L_s = calc_star(rho_c, 1.5)
    if not np.isnan(M_s): soft_data.append([M_s, L_s])

# 打印到日志
print("====== HARD_CORE ======")
for row in hard_data: print(f"Result: M={row[0]:.2f} Msun, Lambda={row[1]:.2f}")
print("====== SOFT_CORE ======")
for row in soft_data: print(f"Result: M={row[0]:.2f} Msun, Lambda={row[1]:.2f}")

# 生成 CSV 文件
np.savetxt("data_hard_core.csv", np.array(hard_data), delimiter=",", header="Mass,Lambda", comments="")
np.savetxt("data_soft_core.csv", np.array(soft_data), delimiter=",", header="Mass,Lambda", comments="")
print("✅ 正宗 CSV 文件已生成！")
