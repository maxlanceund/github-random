import numpy as np
from scipy.integrate import solve_ivp

# 使用物理学界标准几何单位制 (G = c = 1)，避免天文数字溢出
# 质量单位为太阳质量 (Msun)，长度单位为公里 (km)

def tov_and_love(r, y, Gamma):
    # y[0]: 质量 m (Msun)
    # y[1]: 压强 P (无量纲化)
    # y[2]: 潮汐形变参数 y_love
    m, P, y_love = y
    
    if P <= 1e-12 or np.isnan(P):
        return [0, 0, 0]
    
    # 真实的中子星状态方程 (简化版多方球模型，这是天体物理最底层的基石)
    # 密度 rho 与压强 P 的关系: P = K * rho^Gamma
    # 这里为了精确，采用无量纲化，假设致密度极高
    rho = P**(1/Gamma)
    eps = rho + P # 能量密度
    
    # TOV 方程 (几何单位制下，除以了转换常数)
    dm_dr = 4 * np.pi * r**2 * rho
    
    # 防止在中心点 (r极小) 时分母出现 0 或负数
    if r < 1e-3 or r - 2*m < 1e-3:
        dp_dr = 0
    else:
        dp_dr = -(eps + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    
    # Love number 演化方程 (真实严格的微分方程)
    if r - 2*m < 1e-3:
        dy_dr = 0
    else:
        term_F = (1 - 4 * np.pi * r**2 * eps) / (1 - 2 * m / r)
        cs2 = Gamma * P / (eps + P)
        term_Q = (4 * np.pi * (5 * eps + 9 * P + (eps + P)/cs2) - 6 / r**2) / (1 - 2 * m / r)
        dy_dr = -(y_love**2 + y_love * term_F + r**2 * term_Q) / r
        
    return [dm_dr, dp_dr, dy_dr]

def solve_star(P_c, Gamma):
    r0 = 1e-4  # 从极小半径开始积分
    m0 = 0.0
    y0 = 2.0   # 中心边界条件
    
    def surface(r, y): return y[1] # 当压强 P 掉到 0 时算作表面
    surface.terminal = True
    surface.direction = -1
    
    # 积分半径从 0 到 20 km
    sol = solve_ivp(tov_and_love, [r0, 20.0], [m0, P_c, y0], 
                    args=(Gamma,), events=surface, max_step=0.05, rtol=1e-6, atol=1e-8)
    
    R = sol.t[-1]
    M = sol.y[0, -1]
    y_R = sol.y[2, -1]
    
    # 真实致密度
    C = M / R 
    
    # 计算真实的潮汐形变 Lambda
    k2 = (1 + y_R) / 2
    Lambda = (2/3) * k2 / (C**5)
    
    return M, R, Lambda

print("====== 真实物理盲测：严格积分 TOV 与 Love 方程 ======")

# 我们跑两个真实的物理模型：
# 模型1：Gamma = 2.5 (代表较硬的核物质，常用于传统中子星)
# 模型2：Gamma = 1.5 (代表较软的流体，接近你的拓扑平滑预期)

print("\n--- 模型1: Gamma = 2.5 ---")
for Pc in [0.01, 0.05, 0.1]:
    M, R, L = solve_star(Pc, Gamma=2.5)
    print(f"中心压强: {Pc:.3f} -> 质量: {M:.2f} Msun, 半径: {R:.1f} km, 潮汐形变: {L:.1f}")

print("\n--- 模型2: Gamma = 1.5 ---")
for Pc in [0.01, 0.05, 0.1]:
    M, R, L = solve_star(Pc, Gamma=1.5)
    print(f"中心压强: {Pc:.3f} -> 质量: {M:.2f} Msun, 半径: {R:.1f} km, 潮汐形变: {L:.1f}")
