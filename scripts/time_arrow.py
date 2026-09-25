import numpy as np
import json

def simulate(N=2000, n_steps=500, friction=0.0, dt=0.01, omega=1.0):
    np.random.seed(42)
    x = np.random.randn(N) * 0.01
    v = np.random.randn(N) * 0.01

    def D_f(x):
        hist, _ = np.histogram(x, bins=50, range=(-3, 3))
        p = hist / hist.sum()
        p = p[p > 0]
        H = -np.sum(p * np.log2(p))
        return H / np.log2(50)

    Df_fwd = []
    x_fwd = [x.copy()]
    for _ in range(n_steps):
        Df_fwd.append(D_f(x))
        a = -omega**2 * x - friction * v
        v = v + a * dt
        x = x + v * dt
        x_fwd.append(x.copy())

    # 时间反演
    v = -v
    Df_bwd = []
    x_bwd = [x.copy()]
    for _ in range(n_steps):
        Df_bwd.append(D_f(x))
        a = -omega**2 * x - friction * v
        v = v + a * dt
        x = x + v * dt
        x_bwd.append(x.copy())

    return np.array(Df_fwd), np.array(Df_bwd)

if __name__ == "__main__":
    print("=" * 70)
    print("D_f 的时间反演对称性测试")
    print("=" * 70)
    print()

    results = {}
    for name, friction in [("无摩擦", 0.0), ("有摩擦 γ=0.1", 0.1), ("有摩擦 γ=0.5", 0.5)]:
        print(f"--- {name} ---")
        Df_f, Df_b = simulate(N=2000, n_steps=500, friction=friction)

        # 时间反演对称性：D_f_fwd(t) vs D_f_bwd(T - t)
        T = len(Df_f)
        Df_b_reversed = Df_b[::-1]

        # 对称性误差
        error = np.mean(np.abs(Df_f - Df_b_reversed))
        max_error = np.max(np.abs(Df_f - Df_b_reversed))

        print(f"  D_f 正向:   起点={Df_f[0]:.4f}, 终点={Df_f[-1]:.4f}")
        print(f"  D_f 逆向:   起点={Df_b[0]:.4f}, 终点={Df_b[-1]:.4f}")
        print(f"  |D_f(t) - D_f(T-t)| 平均误差: {error:.6f}")
        print(f"  |D_f(t) - D_f(T-t)| 最大误差: {max_error:.6f}")
        print()

        results[name] = {
            "friction": friction,
            "Df_forward": Df_f.tolist(),
            "Df_backward": Df_b.tolist(),
            "mean_error": float(error),
            "max_error": float(max_error)
        }

    print("=" * 70)
    print("结论：")
    print("=" * 70)
    for name, r in results.items():
        if r["mean_error"] < 0.01:
            print(f"  {name}: 时间对称 ✓")
        else:
            print(f"  {name}: 时间不对称 ✗（平均误差 {r['mean_error']:.4f}）")

    with open("time_arrow_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("结果已保存到 time_arrow_result.json")
