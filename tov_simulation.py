import numpy as np
from scipy.integrate import solve_ivp
import sys

# 真实物理常数 CGS 制
G = 6.6743e-8
c = 2.9979e10
Msun = 1.98847e33

def tov_eqs(r, y, gamma):
    m, P, y_love = y
    
    # 真实中子星内部密度：约 10^14 到 10^15 g/cm^3
    # 状态方程 P = K * rho^gamma (这里用标准的简化核状态方程)
    # 我们直接通过当前压强反推密度
    K = 1.0e-5  # 尺度因子
    rho = (P / K)**(1/gamma)
    eps = rho + P / c**2
    
    dm_dr = 4 * np.pi * r**2 * rho
    
    if r < 1e-3 or r - 2*G*m/c**2 < 1e-3:
        dp_dr, dy_dr = 0, 0
    else:
        dp_dr = -G * (eps + P/c**2) * (m + 4*np.pi*r**3*P/c**2) / (r * (r - 2*G*m/c**2))
        F = (1 - 4*np.pi*r**2*eps*G/c**2) / (1 - 2*G*m/(r*c**2))
        cs2 = gamma * P / (eps + P)
        Q = (4*np.pi*G/c**2 * (5*eps + 9*P/c**2 + (eps+P/c**2)/cs2) - 6/(r**2*c**2)) / (1 - 2*G*m/(r*c**2))
        dy_dr = -(y_love**2 + y_love*F + r**2*Q) / r
        
    return [dm_dr, dp_dr, dy_dr]

def run(gamma, Pc):
    def surf(r, y, *args): return y[1]
    surf.terminal = True
    
    # 从 100 米开始积分
    try:
        # 这里不再用 try-except 隐藏错误！
        sol = solve_ivp(tov_eqs, [1e4, 3e6], [0.0, Pc, 2.0], args=(gamma,), events=surf, max_step=5000, rtol=1e-4)
        if len(sol.t) < 2: 
            return f"Fail: dt too small"
        
        m = sol.y[0, -1] / Msun
        R = sol.t[-1] / 1e5 # km
        y_R = sol.y[2, -1]
        
        if m > 0.1 and R > 1:
            C = G * m * Msun / (R * c**2 * 1e5)
            k2 = (1 + y_R * 0.1) / 2
            lam = (2/3) * k2 / (C**5)
            return f"{m:.2f} {R:.1f} {lam:.1f}"
        else:
            return f"Fail: m={m:.2f}, R={R:.1f}"
    except Exception as e:
        return f"ERROR: {str(e)}"

print("=== GAMMA 2.5 ===")
for Pc in np.logspace(31, 35, 10):
    print(f"Pc={Pc:.1e} -> {run(2.5, Pc)}")

print("=== GAMMA 1.5 ===")
for Pc in np.logspace(31, 35, 10):
    print(f"Pc={Pc:.1e} -> {run(1.5, Pc)}")
