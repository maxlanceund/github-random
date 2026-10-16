"""实验一补强：确定性凝聚 (H1) - 多 seed"""
import random, sys, os, statistics, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.boolean_network import BooleanNetwork, state_entropy


def run(N, K, T, n_trials, seed):
    rng = random.Random(seed)
    net = BooleanNetwork(N, K, seed=seed)
    states = [tuple(rng.randint(0, 1) for _ in range(N)) for _ in range(n_trials)]
    H_curve = []
    for t in range(T + 1):
        H_curve.append(state_entropy(states))
        if t < T:
            states = [net.step(s) for s in states]
    return H_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=2)
    args = parser.parse_args()
    N, T, n_trials = 10, 50, 50000
    seeds = list(range(1, 4))
    print("=" * 60)
    print(f"实验一补强：确定性凝聚 (H1)")
    print(f"N={N}  样本={n_trials}  采样天花板={math.log2(n_trials):.2f}")
    print(f"K={args.K}  seeds={len(seeds)}")
    print("=" * 60)
    all_curves = []
    for s in seeds:
        c = run(N, args.K, T, n_trials, seed=s)
        all_curves.append(c)
    avg = [statistics.mean(c[t] for c in all_curves) for t in range(T + 1)]
    std = [statistics.stdev(c[t] for c in all_curves) for t in range(T + 1)]
    print(f"\nK={args.K}")
    for t in [0, 5, 10, 20, 30, 50]:
        print(f"  t={t:3d}  H={avg[t]:.4f} ± {std[t]:.4f}")
