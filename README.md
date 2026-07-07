# K-Net: 智能不确定性原理 $C \cdot V \ge K_{\text{int}}$ 的实证检验与修正

对"智能作为一种存在，由复杂度 $C$ 与预测误差 $V$ 的乘积下界 $C \cdot V \ge K_{\text{int}}$ 所定义"这一命题的三层实证检验：**测量 → 证伪 → 修正**。

## 核心结论（一句话）
> 链接给出的**方向感对**（用复杂度感知做调控）、**不等式方向错**（维持 $C\cdot V$ 下界是反 grokking）。修正为"先拟合→压缩→**泛化即收手**"的闭环后，K-Net 在 FRSMASH v3.6 上实现 **3/3 种子 $wd=0$ 稳定 grokking**——这是标准模型与原版 K-Net 都做不到的。

## 三层实验
| 层 | 问题 | 结论 |
|----|------|------|
| **实验 1** | $K_{\text{int}}$ 可测且是常数吗？ | ✅ $K=C\cdot V$ 收敛到平台，跨 3 种子 $CV=0.23\%$ |
| **实验 2/3** | 原版 K-Net（存在性惩罚）能诱发 grokking 吗？ | ❌ 方向反了，0/3 grok；可被"蠢定点"平凡满足 |
| **实验 3 修正** | 翻向 + 闭环能 work 吗？ | ✅ 3/3 种子 $wd=0$ 稳定 grok，比标准+wd 更快 |

## 关键数字
- 跨种子平台 $K$：35.475 / 35.404 / 35.275 M，**$CV=0.23\%$**
- 原版 K-Net $\gamma=5$：钉死 $K_{\text{int}}$ 偏差 **0.11%**，但 train=0.36/test=0（满足约束 ≠ 智能）
- 闭环 K-Net $wd=0$：seed 0/1/2 全 grok（s3300/4400/3200），末段 test 0.997–0.998，稳定秩 47→2–3.4

## 完整论文
**[PAPER.md](PAPER.md)** —— 含全部实验数据、表格、裁定与讨论。

## 结构
```
src/           FRSMASH v3.6 模型 + open_ash_voc 分词器(自包含)
experiments/   exp1 K_int 测量 / exp2 原版K-Net(MLP) / exp3 FRSMASH Grokking+5种governor
data/          全部 CSV 实验日志
figures/       6 张分析图
docs/          RESULTS.md, IMPROVEMENT.md
```

## 复现
见 [PAPER.md §8](PAPER.md#8-复现)。需要：PyTorch 2.x + CUDA、`fla`、`jieba`，以及数据缓存（`*.pt`、`open_ash_voc_agent.json`，不入库）。

## 诚实的边界
- $K_{\text{int}}$ 的**架构不变性**（FRSMASH vs Transformer）未测。
- 闭环 K-Net 本质 ≈ 自适应 weight_decay 调度，物理上无新力。
- "K-Net 必胜 SOTA"类断言属纯论述，未做对照。
