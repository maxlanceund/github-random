import numpy as np
from scipy.integrate import solve_ivp

# 真实物理常数 (CGS)
G = 6.67430e-8       # cm^3 / g s^2
c = 2.99792458e10    # cm/s
Msun = 1.98847e33    # g

# 修正后的方程，严格核对无量纲和各项系数
def tov_eqs(r, y, K, Gamma):
    m, P, y_love = y
    if P <= 1e-10: 
        return [0, 0, 0]
    
    # 1. 密度由压强反推
    rho = (P / K)**(1/Gamma)
    eps = rho + P / c**2 
    
    # 2. TOV 方程 (G, c 均保留)
    dm = 4 * np.pi * r**2 * rho
    dp = -G * (eps + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r * (r - 2*G*m/c**2))
    
    # 3. 潮汐形变修正项 (补齐了 G/c^2，修正了 Q 中的导数项)
    term_F = (1 - 4*np.pi*r**2*eps*G/c**2) / (1 - 2*G*m/(r*c**2))
    # 声速平方 (dP/deps)
    cs2 = Gamma * P / (eps + P) 
    term_Q = (4*np.pi*G/c**2 * (5*eps + 9*P/c**2 + (eps + P/c**2)/cs2) - 6/(r**2*c**2)) / (1 - 2*G*m/(r*c**2))
    
    dy = -(y_love**2 + y_love*term_F + r**2*term_Q) / r
    
    return [dm, dp, dy]

def calc_star(rho_c, K, Gamma):
    r0 = 1.0e3 
    P_c = K * rho_c**Gamma
    m0 = (4/3) * np.pi * r0**3 * rho_c
    
    def surf(r, y): return y[1]
    surf.terminal = True
    
    try:
        sol = solve_ivp(tov_eqs, [r0, 3.0e6], [m0, P_c, 2.0], 
                        args=(K, Gamma), events=surf, max_step=500.0, rtol=1e-6, atol=1e-8)
        
        R = sol.t[-1] 
        M = sol.y[0, -1] / Msun 
        y_R = sol.y[2, -1]
        
        k2 = (1 + y_R) / 2
        C = G * M * Msun / (R * c**2)
        Lambda = (2/3) * k2 / (C**5)
        
        return M, Lambda
    except Exception:
        return np.nan, np.nan

# 中心密度扫描：10^14 - 10^15 g/cm^3
rho_c_values = np.linspace(5e14, 1.5e15, 8)

print("====== HARD_CORE (Dense) ======")
# 硬核模型：高刚度，状态方程较硬 (Gamma=2.5, K 对应真实核物质刚度)
for rho in rho_c_values:
    M, L = calc_star(rho, K=1.5e10, Gamma=2.5)
    if M > 0.5: # 过滤掉不合理的极小质量
        print(f"Mass: {M:.2f} Msun | Lambda: {L:.2f}")

print("====== SOFT_CORE (Smooth) ======")
# 平滑模型：软化相变，容易变形 (Gamma=1.5, 同中心密度)
for rho in rho_c_values:
    M, L = calc_star(rho, K=1.5e10, Gamma=1.5)
    if M > 0.5: 
        print(f"Mass: {M:.2f} Msun | Lambda: {L:.2f}")
