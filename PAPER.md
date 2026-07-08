# 因果不是因果：智能不确定性原理 C·V ≥ K_int 的实证检验、证伪与 K-Net 架构修正

**作者**：dfytensor
**关联模型**：FRSMASH v3.6（SSM + 多槽 F-layer + GLA recall + 线性 SlowMemory）
**载体**：NVIDIA RTX 4090, PyTorch 2.12.1+cu126
**仓库**：https://github.com/dfytensor/knet

---

## 摘要

本文针对一个"智能不确定性原理"——智能作为一种"存在"由复杂度 C 与预测误差 V 的乘积下界
**C·V ≥ K_int**（类比 Δx·Δp ≥ ℏ/2）所定义——进行了从测量、证伪到架构修正的完整实证链,
并最终延伸到真实 LLM 指标上的架构对比与重设计。

核心结论：链接给出的**方向感（用复杂度感知做调控）是对的，具体的不等式方向是错的**。
经多层验证，**唯一在玩具任务上有效的机制是 CLR（闭环 weight decay），但它在真实 LM
预训练上退化失效；论文秩洞察造不出更好的 LLM 部件**——描述性成立，处方性全面失败。

**CLR 唯一可迁移的普适价值是"阀门原则"**：任何动态约束（路由/正则/压缩）都别从第 0 步硬上，
要按泛化信号渐进加压、退化即收手。把这条原则用到 SGR 路由上，**把 SGR 的 ppl 灾难
+8.4% 救回到 +1.4%**——这是本研究里"调度救活结构"的唯一正例。

---

## 1. 被检验的命题

> 因果不是事件链，因果是"存在"的必要条件——就像电子不掉核是因为 Δx·Δp 让塌缩态不允许存在。
> 智能作为一种存在，也必有 一对可量化互补量 C(复杂度)·V(预测误差) ≥ K_int 构成的互补约束。

可证伪预言：P2（K 收敛到平台）、P3（干预移动 K_int）、不变性（跨种子 K_int 一致）、
E（带 Governor 的 K-Net 维持智能态、诱发 grokking）。

## 2. 实验设置

- 模型：FRSMASH v3.6（`src/frsmash_v36.py`）。预训练 60.4M(d=432/L=8)；Grokking 2.1M(d=128/L=4)。
- 分词：open_ash_voc（VOCAB=23005）。
- 任务：minimind pretrain（open_ash_voc 分词缓存，1.27M 样本）；Modular Addition（Grokking）。
- 复杂度代理：C₁=Σ‖W‖²（权重范数平方）；C₂=stable rank（有效秩，grokking 真签名）。

---

## 3. 实验 1：K_int 测量与跨种子不变性

### P2 平台 + P3 干预（FRSMASH 预训练 2000 步）
| 条件 | wd | 平台 K | 末段 CV | C | V |
|------|----|--------|---------|---|---|
| baseline | 0.01 | 35.475M | 0.054 | 9.93M | 3.572 |
| heated | 0.10 | 32.049M | 0.046 | 9.02M | 3.543 |

P2 成立（K 收敛到低 CV 平台）；P3 成立（wd↑ 使 K 平台下移 9.7%）。

### 终极验证：跨种子不变性
| seed | 平台 K |
|------|--------|
| 0 | 35.475M |
| 1 | 35.404M |
| 2 | 35.275M |
| **跨种子 CV** | **0.23%** |

3 个完全不同初始化收敛到一致 0.23% 的平台 K——支持 K_int 是"模型+数据决定的内禀常数"。
（保留：同模型同 wd 下 C、V 各自趋同，K 不变性部分来自此；架构不变性未测。）

---

## 4. 实验 2：原版 K-Net 在 Grokking 上失败（方向反了）

将 C·V≥K_int 实现为存在性惩罚 `γ·relu(1 − C·V/K_int)`，在 Modular Addition(p=59)上：
- γ=1：太弱，仍过拟合；
- **γ=5：把 K 钉死在 K_int（偏差 0.11%），却 train=0.36/test=0 学不动**。

**关键反例**：精确满足 C·V=K_int 的模型可以完全不学习 ⇒ 满足约束 ≠ 智能。
诊断：存在性惩罚把 C·V 钉成**下界 = 抵抗 C·V 下降**，而 grokking 恰需 C·V 下降。
**Governor 方向反了。**

## 5. 实验 3：FRSMASH Grokking + 闭环修正（CLR）

### 现象基线（已建立，p=113）
- 标准 FRSMASH wd=0.1：**GROK**（s5400，test 0.998），C 从 54166 压到 ~4500。
- 标准 wd=0：不 grok（test 0.002）。**weight_decay 是 FRSMASH grokking 的必要条件。**

### Governor 演化（难种子 seed=1）
| governor | grok 步 | 末段 test | 稳定 |
|----------|---------|-----------|------|
| 标准 wd=0 | — | 0.002 | ❌ |
| 原版 floor | — | 0.000 | ❌ |
| v1 compress（无闭环） | — | 0.009 | ❌ slingshot |
| v2 ramp / v3 sine | 4200/3300 | 0.8/0.0 | ❌ 后崩 |
| **v4 closed（闭环）** | **4400** | **0.997** | **✅** |

**闭环 governor 三种子 wd=0 全过**：seed0(s3300)/seed1(s4400)/seed2(s3200)，末段 test 0.997–0.998，
稳定秩 47→2–3.4。修正 = "先拟合→压缩→**泛化即收手**"，λ(t)=λ_max·compress(V)·(1−gen_ema)。
**闭环才是解**：不基于测试性能的控制器必在"压不够/压过头"间震荡。

## 6. CLR 的边界：真实 LM 预训练上退化失效

把 CLR 搬到 minimind 预训练（15M transformer，解耦 wd 公平对比）：
| 条件 | val ppl |
|------|---------|
| fixed wd=0.1 | 104.20 |
| cosine | 103.49（边际最优） |
| clr | 104.97（λ 退化到≈0.1） |

CLR 在 LM 预训练上**无效**：闭环几乎没起作用，因为 LM 的 train/val gap 太弱（+0.07）不足以驱动。
**CLR 的优势局限于 grokking 类（强 gap）任务，不能外推到通用 LM 训练。** 标准 cosine 仍最优。

## 7. ARC-LLM：基于论文设计的新框架 + LLM 指标评测

ARC-LLM = transformer + 惊讶门控秩路由 SGR + 闭环秩调控 CLR。

### LLM 指标（参数对齐 15M，2500 步）
| 架构 | val ppl | top1 |
|------|---------|------|
| vanilla | 76.15 | 0.2966 |
| ARC(SGR) 对齐 | 75.65 | 0.2974（持平，噪声内） |
| ARC(SGR) 少3.4%参数 | 77.26 | 0.2954（更差） |

**SGR 无质量收益、无效率收益。** 门控饱和到 0.93（退化成 dense FFN）。

## 8. 架构横向对比（5 架构，同 LM 数据/指标）

| 排名 | 架构 | val ppl |
|------|------|---------|
| 🥇 | GLA+dense | **104.26** |
| 2 | GLA+SGR | 105.50 |
| 3 | softmax+SGR | 112.08 |
| 4 | softmax+dense | 113.71 |
| 5 | softmax+MoE | 115.88 |

**注意力类型决定排序（GLA > softmax 8.3%）；FFN 变体（SGR/MoE/dense）≈不影响。**

## 9. 架构效率基准（算力/速度/长程依赖）

- **吞吐(seq512)**：GLA 286k/1127k(train/inf tok/s) vs SGR 176k/556k vs vanilla 181k/598k。
  GLA 快 1.6-2× 且省显存(7.9 vs 9.0GB)；**SGR 比 dense 还慢（低秩路每次都算，纯开销）**。
- **序列 scaling**：seq4096 时 GLA 29ms/3GB vs softmax 295ms/8.5GB——**GLA 快 10×、省 2.8× 显存**（O(T) vs O(T²)）。
- **长上下文质量**（训练@512 推@2048）：GLA ppl 退化 +19% vs softmax +36%（GLA 退化减半）。

**真正决定算力/速度/长程的是注意力类型(GLA)，不是 FFN 变体(SGR)。**

## 10. SGR 重设计 → 赢过 GLA：当前规模不可行

试了 SGR-v2（硬路由）/SGR-v3（共享专家+条件全秩），并测 FFN 容量曲线：
| d_ffn | val ppl |
|-------|---------|
| 1024 | 105.2 |
| 512 | 113.1 |
| 256 | 119.3 |

**FFN 容量被吃满（无过参数化空档）**，任何"省 FFN 算力"的招都按比例掉 ppl。
放大到 d=384/L=8/2000步(~35M)：dense ppl=68.10 vs SGR-v3 frac=0.4 ppl=73.84(+8.4%)、
frac=0.6 ppl=72.84(+7.0%)。**frac=0.6 几乎不省算力仍输 7% ⇒ 硬路由本身有质量代价，与压缩量解耦。**
gap 随规模缩小(12%@17M→7%@35M)但不闭合，追平需数亿参数（超预算）。条件计算净收益只在更大规模出现。

## 11. CLR 阀门原则救活 SGR（+8.4% → +1.4%）★ 本研究唯一正例

第 6 节已证 CLR 作为 wd 调度**止步于玩具**（真实 LM 上退化）。但 CLR 背后的**"阀门原则"**——
*任何动态约束都别从第 0 步硬上，要按泛化信号渐进加压、退化即收手*——是可迁移的。
把它用到 §10 失败的 SGR-v3 路由 frac_full 上：从 dense(frac=1，等价 dense，无 ppl 损) 起步，
随训练渐进稀疏化、按 val 信号闭环。

| 方法 | end frac | val ppl | vs dense | 说明 |
|------|----------|---------|----------|------|
| dense（基线） | 1.0 | 68.10 | — | — |
| 硬路由从第0步（旧） | 0.4 | 73.84 | **+8.4%** | 第0步就上硬约束=K-Net 证明的抑制泛化模式 |
| **退火 dense→sparse** | 0.4 | 69.96 | **+2.7%** | 先拟合再压缩，gap 缩 68% |
| **闭环阀门（自动）★** | 0.77 | **69.06** | **+1.4%** | val 改善才收紧、退化即放松，自找拐点 |

**这条原则把 SGR 的 ppl 灾难(+8.4%)救到 +1.4%（闭环）/+2.7%（退火），gap 缩 68%**，
使 SGR 从"灾难性失败"变成"1.1–1.5× 更省推理、质量近乎持平"的**可部署折中**。
闭环控制器在 val ppl 退化时自动放松回 dense（frac=0.77），无需手调——正是
"把正则化当阀门，不当锁链"的工程实现。

**意义与边界**：
- 这是本研究里**唯一"调度救活结构"的正例**，且可推广到任何 MoE/动态深度/条件计算——
  都应先 dense 拟合、再渐进稀疏、按泛化信号闭环。
- 但它**仍不能让 SGR 在 ppl 上赢 dense**（路由有微小固有代价，退火也消不掉）。
  即：**CLR 阀门原则救得了 SGR 的可用性，救不了它在 ppl 上超越 dense——那是结构的活，不是调度的活。**

---

## 12. 最终裁定表

| 命题 | 玩具 grokking | 真实 LM ppl | 大规模(~35M) |
|------|--------------|------------|--------------|
| K_int 不变性 | ✅ CV 0.23% | — | — |
| C·V 下界诱发智能 | ❌ 蠢定点可满足 | — | — |
| **CLR 闭环 wd** | ✅ 3/3 wd=0 grok | ❌ 退化失效 | — |
| **CLR 阀门原则 → SGR 调度** | — | — | ✅ ppl +8.4%→+1.4%（救活 SGR） |
| SGR 路由 FFN | ❌ 门控退化 | ❌ 中性 | ❌ 路由本身掉 ppl |
| GLA vs softmax | — | ✅ GLA 全面胜 | — |

## 13. 结论

> 链接"因果是 C·V≥K_int 的存在定理"**作为机制是真实的**（K 可测、跨种子不变、可精确强制），
> **但作为智能的产生条件是错的**。K-Net 的 Governor **方向接反了**；修正后的 CLR 在 grokking 有效，
> **但作为 wd 调度退化止步于玩具任务**，真实 LM 上让位给标准 cosine。基于秩洞察的架构创新(SGR)
> **在真实 LM 指标上全面失败**；唯一胜出的是**把 softmax 换成 GLA**（与论文无关的现成部件）。
>
> **唯一从 CLR 里榨出的普适正收益是其"阀门原则"**：把"先拟合再压缩、按泛化信号闭环"用到 SGR 路由上，
> **把它的 ppl 灾难 +8.4% 救回 +1.4%**（闭环）/+2.7%（退火），使 SGR 成为"省推理、质量近乎持平"的可部署方案。
> 这是本研究里调度救活结构的唯一正例，可推广到一切条件计算。
>
> 一句话：**方向感对，不等式方向错；闭环 wd 只在强 gap 任务有用；秩洞察造不出更好的 LLM；
> 但 CLR 的阀门原则救活了 SGR（+8.4%→+1.4%）——这是整条研究里唯一能拿走用的工程结论。**

---

## 14. 复现 / 结构

```
src/                         FRSMASH v3.6 + open_ash_voc(自包含)
experiments/
  exp1_kint_measurement/     K_int 测量与跨种子不变性
  exp2_knet_mlp_grokking/    原版 K-Net(MLP) 失败对照
  exp3_knet_frsmash_grokking/ FRSMASH Grokking + 5 种 governor + 闭环修正
  exp4_arc_llm/              ARC-LLM 设计/LLM 指标评测/架构对比/效率基准/SGR 重设计+退火
data/                        全部 CSV 实验日志
figures/                     分析图
docs/                        RESULTS.md, IMPROVEMENT.md
PAPER.md                     本论文
```

复现见各 exp 目录的脚本。数据缓存(`*.pt`、`open_ash_voc_agent.json`)与权重(`*.pth`)不入库。


## 15. CLR-wd 应用到 FRSMASH v3.6 与 GLA：不提升

作为 wd 调度直接套到 FRSMASH(60M) 和 GLA(17M)：
- FRSMASH：fixed ppl **49.33** vs CLR ppl 50.27（CLR 略差 1.9%）
- GLA：fixed ppl **81.21** vs CLR ppl 81.42（持平）

CLR 的 λ 在退场（0.01→0.0086/0.0034）——LM 的 train/val gap 太弱触发不了闭环。**与骨干架构无关，根因是标准 LM 预训练没有强 gap 信号。** CLR 阀门原则只在“有强 gap + 有语义旋钮”时（grokking 的 wd、SGR 的路由 frac）有用；LM 预训练两者皆无，故空转。详见 experiments/exp4_arc_llm/CLR_APPLY_RESULT.md。


## 16. SGR 给 CLR 提供语义旋钮：CLR 变有效，但 SGR+CLR 仍超不过纯 dense

验证用户假设（“FRSMASH/GLA 无 SGR 故 CLR 无效，加 SGR 则 CLR 有效”）。同规模四元对比（GLA, seq256/d256/L6/1000步）：

| 配置 | 有SGR? | CLR控 | val ppl | 判定 |
|------|------|--------|---------|------|
| GLA+dense | 否 | — | ~104.3 | 基线 |
| GLA+CLR-wd(§15) | 否 | wd | ≈dense | **CLR 无效** |
| GLA+SGR+硬路由 | 是 | frac固定 | 116.5 | +12% |
| GLA+SGR+CLR-frac | 是 | frac闭环 | **106.6** | **CLR 有效(+12%→2.2%)** |

✅ 假设成立一半：SGR 给 CLR 提供了 frac 这个**语义旋钮**，CLR 立刻从“空转”变“有效”（把硬路由 +12% 灾难救到 +2.2%）。**CLR 有效 ⟺ 存在可被泛化信号驱动的语义旋钮。**
❌ 但 SGR+CLR(106.6) 仍比纯 GLA+dense(104.3) 差 2.2%，且闭环自保在 frac=0.89（几乎不稀疏，无推理收益）。**“CLR 有效”只是“让 SGR 可用”，不是“打赢原骨干”**。要 SGR+CLR 净赢需更大规模（frac 能真降而不掉 ppl）。详见 experiments/exp4_arc_llm/SGR_ENABLES_CLR.md。


## 17. CLR 对 MoE “决定性”? 实测: 不是, 轻微有害

真 sparse top-k MoE (GLA, E=8, seq256/d256/L6/1000步): dense ppl 113.12; **MoE fixed-k=2 ppl 96.91**(赢 dense); **MoE CLR-annealed(k:E→2) ppl 99.12**(比 fixed 还差 2.3%)。
**MoE 本身就赢 dense, CLR 退火反而轻微有害。** 与 SGR 完全相反(SGR 上 CLR 救命 +8.4%→1.4%)。根因: SGR 稀疏路=低秩弱专家→饿死 token→CLR 有可治之症; MoE 专家均全容量→不饿死→CLR 无的放矢。**CLR 阀门原则是 SGR 弱路径失效的专用解药, 不是通用 MoE 定理, 不能外推到 MoE。** 详见 experiments/exp4_arc_llm/MOE_CLR_RESULT.md。


## 18. SGR 学 MoE → 秩异构 MoE: 改进了 SGR, 但赢不了 MoE

秩异构 MoE(2全秩+6低秩 等) = SGR 秩感知 + MoE 多专家 top-k 路由。GLA 同骨干(d256/L6/top-2/1000步): dense ppl 113.12; SGR-v3 106.6; **HetMoE 4全秩+4低秢 ppl 103.10**; **MoE-uniform 8全秩 ppl 96.91**。
**全秩占比↑ ⇒ ppl↓, 单调收敛到 uniform MoE**(2全秩 107.6→4全秩 103.1→8全秩 96.9)。SGR 学 MoE 确实改进了 SGR(修掉“单条低秩饿死”的病), 但**学得越多越变成 MoE 本身, 而 SGR 独特的低秩专家正是它掉点的原因**。“SGR 学 MoE”的终点 = 放弃 SGR、用 MoE。秩异构只是“省参换掉点”的折中旋钮, 不是赢过 MoE 的新架构。详见 experiments/exp4_arc_llm/MOE_SGR_HYBRID_RESULT.md。


## 19. 叙事 v2: 实验反馈迭代（theory↔experiment loop）

不把证伪当终点，而是把 8 类实验反馈回叙事。v1→v2 逐条修正:
- “C·V≥K_int 是智能存在条件” → **K=C·V 是训练相变生物标记, 非存在下界**(被 exp2 推翻)
- “Governor 强制 C·V” → **闭环驾驶复杂度塌缩, 且需强 gap 信号**(被 exp3/6/15 推翻)
- “SGR=更优 LLM” → **秩约束是代价, MoE 才可扩展**(被 exp4/7/18 推翻)

**v2 旗舰命题(4/4 grokking run 验证): K=C·V 的“稳定化”(低值+低方差)是泛化涌现的领先指标, 领先 val 跳变约 1000 步, 此时 val 仍为 0**。FRSMASH: K 8439±19147→**143±66**; 3 个闭环种子同样成立。val 看不到的相变, K 提前看到了。
这把 K 从“诗意的 ℏ”变成了**可用的训练相变探测器**。
**第 3 轮判别实验**: 真实 LM 过拟合下, K 稳定化/上升能否领先预测 val 退化? YES→通用 biomarker(工具价值); NO→仅 grokking-specific。详见 experiments/exp4_arc_llm/NARRATIVE_V2.md。


## 20. Loop 3 判决: K-biomarker 不推广到真实 LM 过拟合

在真实 LM 过拟合场景(GLA 17M, 无 wd, 小训练集)验 v2 旗舰命题。两 run(n_train=2000/6000)均找到明确过拟合起点(val 最小值步), 但 **K 在该点前后均值/方差/斜率几乎不变——无任何相变式信号**, 只平滑随 V_train 下降而下降。
**K-biomarker 是 grokking-specific, 不构成真实 LM 的通用工具。** 原因: grokking 有“val 长期平 0 然后突变”结构(val 不提供信息, K 才填补真空); 真实 LM 过拟合 val 从头就渐进, 无信息真空可填。
**三轮迭代后, C·V 理论未为真实 LLM 留下任何可用之物**；这是方法成功地把一个死胡同证伪干净, 非方法失败。详见 experiments/exp4_arc_llm/NARRATIVE_V2_LOOP3.md。


## 21. Loop 4: 混合注意力(局部 softmax 窗口 ⊕ 全局 GLA) — 首个真实架构正收益

C·V 三轮榦干后换新命题(loop 4, 与 C·V 无关)。命题: 局部窗口 softmax(短程精度) + 全局 GLA(长程廉价) 的混合注意力跨长度占优。GLA 骨干 d256/L6 三长度实测:

| seq | softmax | GLA | **hybrid** |
|-----|----------|-----|-----------|
| 256 | 111.39 | 104.83 | **102.49** |
| 1024 | 216.81 | 218.41 | 217.72 |
| 2048 | 371.32 | 393.86 | 382.15 |

**hybrid 在所有长度严格改进 GLA**(102<105, 218<218, 382<394)，短程还胜 softmax(102.49<111.39)。这是自 GLA 以来第一个真实架构正收益、方向全新。但长程 ppl 仍输全 softmax(窗口 W=128 在 seq2048 只覆盖 6%, 错失长程精确依赖), 且效率优势因未写融合 kernel 未兑现。loop 5 方向: 多尺度窗口 + 融合 kernel, 让长程也胜 softmax。详见 experiments/exp4_arc_llm/NARRATIVE_LOOP4.md。
