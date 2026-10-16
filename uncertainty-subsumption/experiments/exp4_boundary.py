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
        if t < T:
            states = [net.step(s) for s in states]
    D_curve = [(N - h) / N for h in H_curve]
    return H_curve, D_curve


if __name__ == "__main__":
    N, K, n_trials = 20, 2, 2000
    print("=" * 50)
    print("实验四：边界加速 (H4)")
    print(f"N={N}  K={K}  样本={n_trials}")
    print("=" * 50)
    for T in [50, 100, 200]:
        H, D = run(N, K, T, n_trials, seed=42)
        mid = D[T//2-5:T//2+5]
        pre = D[T-10:T]
        print(f"\nT={T}")
        print(f"  中段 D      = {sum(mid)/len(mid):.4f}")
        print(f"  边界前 10 步 D = {sum(pre)/len(pre):.4f}")
        print(f"  D(T)         = {D[T]:.4f}")
