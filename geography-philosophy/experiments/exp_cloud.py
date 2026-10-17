"""云端大规模探索性模拟
10000 主体，500 代，50 个 seed 平均
输出结果到 results/ 目录
"""
import random
import statistics
import os
import json


def simulate(p_disaster, p_coop, p_indep,
             n_agents=10000, n_generations=500, mutation=0.01, seed=0):
    rng = random.Random(seed)
    agents = ["coop"] * (n_agents // 2) + ["indep"] * (n_agents // 2)
    curve = []
    for gen in range(n_generations):
        survivors = []
        for s in agents:
            disaster = rng.random() < p_disaster
            p = (p_coop if s == "coop" else p_indep) if disaster else 0.95
            if rng.random() < p:
                survivors.append(s)
        if not survivors:
            return 0.0, curve
        new = []
        for _ in range(n_agents):
            parent = rng.choice(survivors)
            child = ("indep" if parent == "coop" else "coop") if rng.random() < mutation else parent
            new.append(child)
        agents = new
        curve.append(agents.count("coop") / len(agents))
    return statistics.mean(curve[-50:]), curve


def multi_seed(pd, pc, pi, seeds=50):
    vals = [simulate(pd, pc, pi, seed=s)[0] for s in range(seeds)]
    return statistics.mean(vals), statistics.stdev(vals)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    output = {}

    print("=" * 76)
    print("云端大规模探索性模拟")
    print("10000 主体，500 代，50 seed")
    print("=" * 76)

    # 四个案例
    cases = [
        ("USA", 0.02, 0.55, 0.90),
        ("China", 0.15, 0.85, 0.30),
        ("Nordic", 0.12, 0.80, 0.40),
        ("MiddleEast", 0.25, 0.90, 0.15),
    ]

    print("\n[四个案例]")
    print("%12s | %10s | %12s" % ("Case", "Mean", "StdDev"))
    print("-" * 40)
    output["cases"] = {}
    for name, pd, pc, pi in cases:
        m, sd = multi_seed(pd, pc, pi)
        output["cases"][name] = {"mean": m, "std": sd}
        print("%12s | %10.3f | %12.3f" % (name, m, sd))

    # 趋势一：灾害频率扫描
    print("\n[趋势一：灾害频率扫描]")
    print("%12s | %10s | %10s" % ("p_disaster", "Mean", "StdDev"))
    print("-" * 40)
    output["trend_disaster"] = {}
    for pd in [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]:
        m, sd = multi_seed(pd, 0.85, 0.3, seeds=20)
        output["trend_disaster"][str(pd)] = {"mean": m, "std": sd}
        print("%12.2f | %10.3f | %10.3f" % (pd, m, sd))

    # 趋势二：协作存活率扫描
    print("\n[趋势二：协作存活率扫描]")
    print("%12s | %10s | %10s" % ("p_coop", "Mean", "StdDev"))
    print("-" * 40)
    output["trend_coop"] = {}
    for pc in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        m, sd = multi_seed(0.15, pc, 0.3, seeds=20)
        output["trend_coop"][str(pc)] = {"mean": m, "std": sd}
        print("%12.2f | %10.3f | %10.3f" % (pc, m, sd))

    with open("results/cloud_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n结果已保存到 results/cloud_results.json")
