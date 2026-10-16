import random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.boolean_network import BooleanNetwork, state_entropy


def run(N, K, T, n_trials, seed):
    rng = random.Random(seed)
    net = BooleanNetwork(N, K, seed=seed)
    states = [tuple(rng.randint(0, 1) for _ in range(N)) for _ in range(n_trials)]
    H_curve = []
    for t in range(T + 1):
        H_curve.append(state_entropy(states))
        states = [net.step(s) for s in states]
    return H_curve


if __name__ == "__main__":
    N, T, n_trials = 20, 50, 3000
    print("=" * 50)
    print("实验一：确定性凝聚 (H1)")
    print(f"N={N}  初始熵上限={N}  样本={n_trials}")
    print("=" * 50)
    for K in [1, 2, 3]:
        curve = run(N, K, T, n_trials, seed=42)
        print(f"\nK={K}")
        for t in [0, 5, 10, 20, 30, 50]:
            print(f"  t={t:3d}  H={curve[t]:.4f}")
