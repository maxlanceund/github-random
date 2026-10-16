"""实验二补强：维持成本 (H2) - 多 seed"""
import random, sys, os, statistics, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.boolean_network import BooleanNetwork, state_entropy


def run(N, K, T, p, n_trials, seed):
    rng = random.Random(seed)
    net = BooleanNetwork(N, K, seed=seed)
    states = [tuple(rng.randint(0, 1) for _ in range(N)) for _ in range(n_trials)]
    for _ in range(50):
        states = [net.step(s) for s in states]
    H_curve = []
    for t in range(T + 1):
        H_curve.append(state_entropy(states))
        new_states = []
        for s in states:
            s2 = list(net.step(s))
            if rng.random() < p:
                i = rng.randint(0, N - 1)
                s2[i] ^= 1
            new_states.append(tuple(s2))
        states = new_states
    return H_curve


if __name__ == "__main__":
    N, K, T, n_trials = 10, 2, 100, 50000
    seeds = list(range(1, 4))
    print("=" * 60)
    print(f"实验二补强：维持成本 (H2)")
    print(f"N={N}  K={K}  样本={n_trials}  seeds={len(seeds)}")
    print("=" * 60)
    for p in [0.0, 0.001, 0.005, 0.01, 0.05]:
        all_curves = []
        for s in seeds:
            c = run(N, K, T, p, n_trials, seed=s)
            all_curves.append(c)
        avg = [statistics.mean(c[t] for c in all_curves) for t in range(T + 1)]
        std = [statistics.stdev(c[t] for c in all_curves) for t in range(T + 1)]
        print(f"\np={p:.4f}")
        for t in [0, 20, 50, 80, 100]:
            print(f"  t={t:3d}  H={avg[t]:.4f} ± {std[t]:.4f}")
