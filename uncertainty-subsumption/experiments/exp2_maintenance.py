import random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.boolean_network import BooleanNetwork, state_entropy


def run(N, K, T, p, n_trials, seed):
    rng = random.Random(seed)
    net = BooleanNetwork(N, K, seed=seed)
    states = [tuple(rng.randint(0, 1) for _ in range(N)) for _ in range(n_trials)]
    for _ in range(30):
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
    N, K, T, n_trials = 20, 2, 100, 2000
    print("=" * 50)
    print("实验二：维持成本 (H2)")
    print(f"N={N}  K={K}  样本={n_trials}")
    print("=" * 50)
    for p in [0.0, 0.001, 0.01, 0.05, 0.1]:
        c = run(N, K, T, p, n_trials, seed=42)
        print(f"p={p:.3f}  H(0)={c[0]:.3f}  H(50)={c[50]:.3f}  H(100)={c[100]:.3f}")
