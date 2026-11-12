import numpy as np
from scipy.integrate import solve_ivp

def tov_eqs(r, y, gamma):
    m, P, y_love = y
    if P <= 1e-10: return [0, 0, 0]
    
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

def run(gamma):
    for Pc in np.logspace(-4, -1, 30):
        def surf(r, y, *args): return y[1]
        surf.terminal = True
        try:
            sol = solve_ivp(tov_eqs, [1e-4, 20], [0.0, Pc, 2.0], args=(gamma,), events=surf, max_step=0.01, rtol=1e-8)
            if len(sol.t) < 2: continue
            
            m = sol.y[0, -1]
            R = sol.t[-1]
            y_R = sol.y[2, -1]
            
            if m > 0.1 and R > 1:
                C = m / R
                k2 = (1 + y_R * 0.1) / 2
                lam = (2/3) * k2 / (C**5)
                print(f"{m:.3f} {R:.3f} {lam:.3f}")
        except Exception:
            continue

print("GAMMA_2.5_START")
run(2.5)
print("GAMMA_1.5_START")
run(1.5)
