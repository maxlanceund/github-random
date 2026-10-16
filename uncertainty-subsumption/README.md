# 不确定性吞没实验

对应论文：papers/uncertainty-subsumption-cn.md

## 四个可操作假设

- H1：确定性从不确定性中通过操作产生
- H2：维持确定性需要持续消耗
- H3：停止维持后确定性必然衰减
- H4：有限窗口内信号密度升高，边界处归零

## 模型

随机布尔网络（Kauffman, 1969）。N 个节点，每个节点 K 个输入，
真值表随机。操作 = 网络更新。不确定性 = 状态空间最大熵 H = N 比特。

## 运行

    python experiments/exp1_condensation.py
    python experiments/exp2_maintenance.py
    python experiments/exp3_swallowing.py
    python experiments/exp4_boundary.py
