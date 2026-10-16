import random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.boolean_network import BooleanNetwork, state_entropy


def run(N, K, T, n_trials, seed, p=0.05):
    rng = random.Random(seed)
    net = BooleanNetwork(N, K, seed=seed)
    states = [tuple(rng.randint(0, 1) for _ in range(N)) for _ in range(n_trials)]
    for _ in range(30):
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
    N, T, n_trials = 20, 200, 2000
    print("=" * 50)
    print("实验三：确定性吞没 (H3)")
    print(f"N={N}  扰动 p=0.05  样本={n_trials}")
    print("=" * 50)
    for K in [1, 2, 3]:
        H0, c = run(N, K, T, n_trials, seed=42)
        print(f"\nK={K}  H0={H0:.3f}")
        for t in [0, 20, 50, 100, 150, 200]:
            print(f"  t={t:3d}  H={c[t]:.4f}")
