import random
import math
from collections import Counter


class BooleanNetwork:
    def __init__(self, N, K=2, seed=None):
        self.N = N
        self.K = K
        self.rng = random.Random(seed)
        self.inputs = []
        self.rules = []
        for i in range(N):
            choices = [j for j in range(N) if j != i]
            self.inputs.append(self.rng.sample(choices, K))
            self.rules.append([self.rng.randint(0, 1) for _ in range(2 ** K)])

    def step(self, state):
        new = []
        for i in range(self.N):
            idx = 0
            for k, j in enumerate(self.inputs[i]):
                idx |= state[j] << k
            new.append(self.rules[i][idx])
        return tuple(new)


def state_entropy(samples):
    if not samples:
        return 0.0
    counts = Counter(samples)
    total = len(samples)
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log2(p)
    return H
