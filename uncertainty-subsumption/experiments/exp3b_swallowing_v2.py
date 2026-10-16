"""实验三补强：确定性吞没 (H3) - 完整衰减曲线
云端版本，支持 --K 参数
"""
import random, sys, os, statistics, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.boolean_network import BooleanNetwork, state_entropy


def run(N, K, T, n_trials, seed, p=0.05):
    rng = random.Random(seed)
    net = BooleanNetwork(N, K, seed=seed)
    states = [tuple(rng.randint(0, 1) for _ in range(N)) for _ in range(n_trials)]
    for _ in range(50):
        states = [net.step(s) for s in states]
    H0 = state_entropy(states)
    H_curve = [H0]
    for t in range(1, T + 1):
        new_states = []
        for s in states:
            s2 = tuple(b ^ (1 if rng.random() < p else 0) for b in s)
            new_states.append(s2)
        states = new_states
        H_curve.append(state_entropy(states))
    return H0, H_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=2)
    args = parser.parse_args()
    N, T, n_trials = 10, 100, 1000000
    seeds = list(range(1, 11))
    print("=" * 60)
    print(f"实验三补强：确定性吞没 (H3)")
    print(f"N={N}  样本={n_trials}  采样天花板={math.log2(n_trials):.2f}")
    print(f"K={args.K}  seeds={len(seeds)}")
    print("=" * 60)
    all_curves = []
    for s in seeds:
        _, c = run(N, args.K, T, n_trials, seed=s)
        all_curves.append(c)
    avg = [statistics.mean(c[t] for c in all_curves) for t in range(T + 1)]
    std = [statistics.stdev(c[t] for c in all_curves) for t in range(T + 1)]
    print(f"\nK={args.K}")
    for t in [0, 5, 10, 20, 30, 50, 80, 100]:
        print(f"  t={t:3d}  H={avg[t]:.4f} ± {std[t]:.4f}")
