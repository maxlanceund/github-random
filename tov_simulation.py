import numpy as np
from scipy.integrate import solve_ivp

# 真实物理常数 (CGS)
G = 6.67430e-8       
c = 2.99792458e10    
Msun = 1.98847e33    

def tov_eqs(r, y, K, Gamma):
    m, P, y_love = y
    if P <= 1e-10: 
        return [0, 0, 0]
    
    # 密度反推
    rho = (P / K)**(1/Gamma)
    eps = rho + P / c**2 
    
    # TOV方程
    dm = 4 * np.pi * r**2 * rho
    dp = -G * (eps + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r * (r - 2*G*m/c**2))
    
    # Love Number 方程 (严格量纲)
    term_F = (1 - 4*np.pi*r**2*eps*G/c**2) / (1 - 2*G*m/(r*c**2))
    cs2 = Gamma * P / (eps + P) 
    term_Q = (4*np.pi*G/c**2 * (5*eps + 9*P/c**2 + (eps + P/c**2)/cs2) - 6/(r**2*c**2)) / (1 - 2*G*m/(r*c**2))
    
    dy = -(y_love**2 + y_love*term_F + r**2*term_Q) / r
    return [dm, dp, dy]

def calc_star(rho_c, K, Gamma):
    r0 = 1.0e2 
    P_c = K * rho_c**Gamma
    m0 = (4/3) * np.pi * r0**3 * rho_c
    
    def surf(r, y): return y[1]
    surf.terminal = True
    
    try:
        sol = solve_ivp(tov_eqs, [r0, 3.0e6], [m0, P_c, 2.0], 
                        args=(K, Gamma), events=surf, max_step=100.0, rtol=1e-6, atol=1e-8)
        
        R = sol.t[-1] 
        M = sol.y[0, -1] / Msun 
        y_R = sol.y[2, -1]
        
        k2 = (1 + y_R) / 2
        C = G * M * Msun / (R * c**2)
        Lambda = (2/3) * k2 / (C**5)
        
        return M, Lambda
    except Exception as e:
        return f"ERROR: {e}", np.nan

print("====== TEST HARD_CORE (Dense) ======")
# 调整 K 值使得压强落在 10^35 量级，撑住星体
M_h, L_h = calc_star(1e15, K=3.0e-2, Gamma=2.5)
print(f"Result: M={M_h} Msun, Lambda={L_h:.2f}")

print("====== TEST SOFT_CORE (Smooth) ======")
# Gamma = 1.5 代表较软的状态方程，更容易变形
M_s, L_s = calc_star(1e15, K=3.0e-2, Gamma=1.5)
print(f"Result: M={M_s} Msun, Lambda={L_s:.2f}")
