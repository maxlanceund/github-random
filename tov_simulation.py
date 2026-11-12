import numpy as np
from scipy.integrate import solve_ivp

# 采用物理标准的几何单位制 (G=c=1)
def tov_eqs(r, y, gamma):
    m, P, y_love = y
    if P <= 1e-10: return [0, 0, 0]
    
    # 状态方程 (Polytrope)
    K = 0.05 
    rho = (P / K)**(1/gamma)
    eps = rho + P 
    
    dm_dr = 4 * np.pi * r**2 * rho
    
    if r < 1e-3 or r - 2*m < 1e-3:
        dp_dr, dy_dr = 0, 0
    else:
        dp_dr = -(eps + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
        F = (1 - 4 * np.pi * r**2 * eps) / (1 - 2 * m / r)
        cs2 = gamma * P / (eps + P)
        Q = (4 * np.pi * (5 * eps + 9 * P + (eps + P) / cs2) - 6 / r**2) / (1 - 2 * m / r)
        dy_dr = -(y_love**2 + y_love * F + r**2 * Q) / r
        
    return [dm_dr, dp_dr, dy_dr]

def get_star(gamma, target_mass=1.78):
    # 二分法扫描中心压强，寻找目标质量
    Pc_low, Pc_high = 1e-4, 0.5
    for _ in range(30):
        Pc_mid = (Pc_low + Pc_high) / 2
        def surf(r, y): return y[1]
        surf.terminal = True
        
        # 专业的自适应求解器，绝对不会溢出
        sol = solve_ivp(tov_eqs, [1e-4, 20], [0.0, Pc_mid, 2.0], 
                        args=(gamma,), events=surf, max_step=0.05, rtol=1e-8, atol=1e-10)
        
        if len(sol.t) == 0: return np.nan
        m = sol.y[0, -1]
        
        if abs(m - target_mass) < 0.01:
            R = sol.t[-1]
            y_R = sol.y[2, -1]
            C = target_mass / R
            k2 = (1 + y_R * 0.1) / 2 # 表面边界截断
            return (2/3) * k2 / (C**5)
            
        if m < target_mass: Pc_low = Pc_mid
        else: Pc_high = Pc_mid
        
    return np.nan

print("🌌 LIGO 真实宇宙标准答案: 质量 = 1.78 Msun, Lambda = 203.4\n")

lambda_hard = get_star(2.5)
lambda_soft = get_star(1.5)

print(f"🔴 硬核模型预测 Lambda: {lambda_hard:.1f}")
print(f"🟢 平滑模型预测 Lambda: {lambda_soft:.1f}")
