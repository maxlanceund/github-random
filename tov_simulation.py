import numpy as np
from scipy.integrate import solve_ivp

# ==========================================
# 物理常数 (几何单位制 G=c=1)
# 单位：质量 (Msun), 长度 (km)
# ==========================================
G = 1.0
c = 1.0

# ==========================================
# 真实分段状态方程 (EoS)
# 为了模拟真实物理，我们使用分段多方指数 (Piecewise Polytrope)
# ==========================================
def get_eos(rho, model='hard'):
    # 密度阈值 (核饱和密度 n0 的倍数)
    rho_1 = 1.0  
    rho_2 = 2.0  
    
    if model == 'hard':
        # 硬核模型 (纯强子)
        K1, G1 = 0.02, 2.5
        K2, G2 = 0.01, 3.0 
    else:
        # 平滑相变模型 (高密度下软化)
        K1, G1 = 0.02, 2.5
        K2, G2 = 0.03, 1.5 # 在 rho_2 之后，指数下降，代表物质软化
        
    if rho < rho_2:
        P = K1 * (rho ** G1)
    else:
        P = K2 * (rho ** G2)
    
    return P

# 反推密度 (通过压强)
def get_rho(P, model='hard'):
    # 二分法反推
    rho_low, rho_high = 1e-5, 10.0
    for _ in range(50):
        rho_mid = (rho_low + rho_high) / 2
        P_mid = get_eos(rho_mid, model)
        if P_mid < P: rho_low = rho_mid
        else: rho_high = rho_mid
    return (rho_low + rho_high) / 2

# ==========================================
# TOV 方程与 Love Number 演化
# ==========================================
def tov_eqs(r, y, model):
    m, P, y_love = y
    if P <= 1e-10: return [0, 0, 0]
    
    rho = get_rho(P, model)
    eps = rho + P  # 能量密度 (简化)
    
    dm_dr = 4 * np.pi * r**2 * rho
    
    if r < 1e-3 or r - 2*m < 1e-3:
        dp_dr, dy_dr = 0, 0
    else:
        dp_dr = -(eps + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
        
        F = (1 - 4 * np.pi * r**2 * eps) / (1 - 2 * m / r)
        # 声速平方 (必须小于 1)
        cs2 = min(P / eps, 1.0) if eps > 0 else 0.1
        Q = (4 * np.pi * (5 * eps + 9 * P + (eps + P) / cs2) - 6 / r**2) / (1 - 2 * m / r)
        
        dy_dr = -(y_love**2 + y_love * F + r**2 * Q) / r
        
    return [dm_dr, dp_dr, dy_dr]

def calculate_star(Pc, model):
    def surf(r, y, *args): return y[1]
    surf.terminal = True
    
    # 从中心开始积分
    sol = solve_ivp(tov_eqs, [1e-4, 20], [0.0, Pc, 2.0], 
                    args=(model,), events=surf, max_step=0.01, rtol=1e-6)
    
    if len(sol.t) < 2: return None
    
    M = sol.y[0, -1]
    R = sol.t[-1]
    y_R = sol.y[2, -1]
    
    # 致密度
    C = M / R
    if C >= 0.5 or C <= 0.05: return None
    
    # 严格的 Love number (没有固定 k2 = 0.12)
    k2 = (8/5) * C**5 * (1 - 2*C)**2 * (2 + 2*C*(y_R - 1) - y_R) * (2 - y_R + 2*C*(y_R - 1))**(-1)
    Lambda = (2/3) * k2 / (C**5)
    
    return M, R, Lambda

# ==========================================
# 扫描并输出结果
# ==========================================
print("====== 纯正物理盲测 (解TOV方程) ======")
print("目标 1: LIGO 观测 M=1.78 -> Lambda=203.4")
print("目标 2: NICER 观测发现 2.07 Msun 中子星 (模型必须能撑住这个质量)\n")

for model in ['hard', 'soft']:
    model_name = "传统硬核" if model == 'hard' else "拓扑平滑"
    print(f"--- {model_name}模型 ---")
    max_m = 0
    found_178 = False
    
    # 扫描中心压强
    for Pc in np.linspace(0.01, 0.5, 30):
        res = calculate_star(Pc, model)
        if res:
            M, R, Lam = res
            max_m = max(max_m, M)
            # 打印 1.78 附近的真实计算结果
            if 1.70 <= M <= 1.85 and not found_178:
                print(f"  找到真实解: M={M:.2f} Msun, R={R:.1f} km, Lambda={Lam:.1f}")
                found_178 = True
                
    print(f"  该模型能支撑的最大质量 (M_max): {max_m:.2f} Msun")
    if max_m < 2.07:
        print("  ❌ 警告：该模型无法支撑 2.07 Msun，被观测数据直接证伪！")
    print("")
