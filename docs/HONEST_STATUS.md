# 诚实现状：我们做成了什么、还缺什么

> 面向社区讨论的自我评估（2026-07）。  
> 指标来自固定金标准页集 + Windows 笔记本级 GPU（约 8GB VRAM）上的墙钟测时。  
> **欢迎指出测法漏洞、复现差异、以及训练/系统上的不同思路。**

仓库：https://github.com/ZhiYiTree/mini-uflash-ocr-windows  
测法与分档：`train/GOLD_BASELINE.md` · `train/STATUS.md` · `webapp/dflash_tiers.py`

---

## 一句话总结

我们在 **Windows + 8GB 级显存** 上，做出了一套 **可运行的本地 OCR 网页**，以及在 **官方 Unlimited-OCR 之上** 的 **可验证投机解码（Stable DFlash）**。

- **有墙钟加速**时，主要靠 **调度与长度策略（含 soft 截断）**，不是「无损、仅靠高接受率」的经典 speculative decoding 叙事。  
- **关掉 soft 截断** 之后，当前 drafter 的 live 接受率 **还不够** 让加速路径稳定赢过官方 decode。  
- 同分布再训（R6）**没有**带来可晋升的权重；训练与推理分布错位是核心疑点。

若你只关心能不能用：可以。  
若你关心「是否已经实现 Hunyuan/DFlash 式无损端到端加速」：**还没有。**

---

## 1. 我们做成了什么（成果）

### 1.1 可交付的本地产品形态

| 能力 | 状态 |
| --- | --- |
| 本机 Gradio 网页，图片 / PDF 逐页 | 可用 |
| 普通模式：官方 Unlimited-OCR `infer` 路径 | 可用 |
| 加速模式：目标模型 **验证前缀** 后提交（非盲写 Direct Block） | 可用 |
| 失败页回退普通模式 | 可用 |
| 三档：`fast` / `balanced` / `lossless` | 已接线到引擎与 UI |
| Windows 安装脚本、模型自动下载（HF OCR + Release 权重） | 可用（OCR 因体积不进 Git） |
| 8GB 可续训脚手架（teacher 抽取、TV/Markov/pos0 等） | 可用 |

### 1.2 工程上站得住的加速路径（Stable DFlash）

相对「直接 commit 草稿」的实验路径，我们刻意做了更稳的一条：

- draft → **target 按序 verify** → 只提交匹配前缀  
- KV **crop** 提交，避免把错误后缀写进 cache  
- 周期 **prefill resync**、退化检测、OOM 回退 B1  
- conf 前缀截断（裁 **draft 尾**，不是永久锁死 pure B1——早期 bug 已修）  
- 成本拆分：draft / verify / resync / B1（便于看钱花在哪）

这套东西的价值是：**正确性边界清楚**，适合和社区讨论「投机 OCR 在 eager + 小显存上到底怎么调度」。

### 1.3 墙钟：在「允许变短」时，确实能快

固定页集（warmup + 5 scored 页），R5 域适配权重，测时口径 = 官方 stable 墙钟 / 加速路径墙钟（>1 表示加速更快）：

| 配置 | 均值墙钟 | 最差 | 平均长度比 (accel/stable) | 说明 |
| --- | ---: | ---: | ---: | --- |
| **sys_v2 / 类 fast（历史最好一轮）** | **~1.87×** | **~1.24×** | **~1.27** | soft 帽 + 早降级 |
| **tier=fast** | **~1.69×** | **~1.24×** | **~1.27** | 同思路，跑次略低 |
| **tier=balanced** | **~1.50×** | **~0.88×** | **~1.45** | 软帽放宽 |
| **tier=lossless**（无 soft） | **~0.95×** | **~0.90×** | **~1.80** | **更完整，但不加速** |
| 早期弱调度 gold v1 | ~1.23× | ~0.94× | ~1.68 | 对照 |

**诚实读表：**

- 1.7×～1.9× **不是**「无损大 block 白嫖」；很大一块来自 **长度 soft/hard 策略**（低 τ 提前停写、硬顶 token）。  
- 部分页长度比 **&lt;1**，即输出可能比官方 **更短**——用完整度换墙钟。  
- **lossless 一关 soft，均值掉到 0.95×**：说明当前 **live 投机收益压不过** draft+verify+膨胀成本。

### 1.4 训练侧：有可用权重，也有失败实验

| 轮次 | 结果（摘要） |
| --- | --- |
| R1–R3 | 域数据 + 基础续训，live 从「明显慢」拉到可讨论区间 |
| R4 | Markov + TV 等工程/损失改进 |
| **R5 (pos0)** | 生产权重；offline pos0 / B8 小幅升；live acc 约 **1.3～1.4** 量级 |
| **R6 同轨 100 页** | **失败停训**（无优于 baseline 的 best） |

另外：

- conf **温度校准**（DSpark 式 verify_len）已做通，但 ** alone 救不了 lossless 墙钟**。  
- 我们学会了：**offline mean accept ≠ 墙钟**；选模必须以 **同页 live 墙钟 + 长度比** 为准。

### 1.5 方法上的「可讨论贡献」（自认）

即使加速倍数一般，下列点或许对社区有用：

1. **8GB Windows eager** 下 speculative OCR 的真实成本结构（B1 常占大头）。  
2. **产品三档** 把「墙钟 / 完整 / 无损叙事」拆开，避免一个数字掩盖取舍。  
3. **金标准固定页 + 只改一个变量** 的对照习惯。  
4. 明确记录：**同分布加数据再训失败** → 指向 train/infer 分布差，而不是「再堆 step」。

---

## 2. 我们还缺什么（不足）

### 2.1 核心缺口：live 接受率不够支撑「无损加速」

- 金标准上 live **mean accepted draft** 大约 **1.1～1.4**（相对 draft 长度 3 的 B4）。  
- pos0 经验接受大约 **~0.65**（校准集量级），离「稳定多 token 投机」还远。  
- 成本上 **pure B1 占比经常 60%～80%+**：很多轮 draft 不划算。

因此：

> **当前「加速器强」的地方在调度；「加速器弱」的地方在 draft 质量 / 分布对齐。**

### 2.2 墙钟与完整度强绑定（用户已感知）

- 默认 **fast**：`soft≈280`、`hard≈384` → 截断感强。  
- 想完整 → **balanced / lossless**，墙钟立刻变差甚至 &lt;1×。  
- 我们还没有找到 **soft≈0 且稳定 ≥1.2×** 的工作点。

### 2.3 训练–推理分布错位（未解决）

| 训练常见设定 | 推理真实设定 |
| --- | --- |
| teacher / 锚定前缀 | free-run 前缀 + 状态机（降级、B1 冷却、resync） |
| 偏 offline block 指标 | 墙钟 = (draft+verify+resync)/τ 与长度共同决定 |

R6 失败强化了判断：**再采同轨 teacher 页硬训，期望值低。**  
更合理的下一枪是 **free-run hard mining**（真 reject 前缀），且以 **soft=0 墙钟** 为门禁——**尚未做成、尚未验证有效。**

### 2.4 测法与泛化局限

- 金标准 **仅 5 页 + 1 warmup**，方差大（fast 1.69× vs 历史 sys_v2 1.87× 可同属噪声/细节差）。  
- 数据偏中文复杂版式样本；跨语言、扫描噪声、表格等 **未系统报**。  
- 仅 **单卡 8GB、eager、本机路径**；未做 TensorRT / 连续 batch / 服务端吞吐。  
- Unlimited-OCR **6GB+ 权重不进 Git**（平台限制），依赖 HF 下载，网络环境差时「开箱」摩擦大。

### 2.5 产品与叙事风险（我们自己要先说清）

| 容易被误解的说法 | 更诚实的说法 |
| --- | --- |
| 「1.8× 无损加速」 | 「在可截断策略下约 1.7～1.9×；无损档约 0.95×」 |
| 「训练大幅提升墙钟」 | 「R5 后墙钟跃迁主要来自 sys_v2 调度；R6 训练失败」 |
| 「DFlash 已对齐混元」 | 「学了分档与 conf 调度思想；硬件与 τ 条件不同，结果不可直接对比」 |

### 2.6 尚未完成的工作

- 默认档是否应从 fast 改为 balanced（完整优先）——产品决策未最终冻结。  
- free-run hard-mine 数据管线与以 soft=0 为门禁的训练 **未落地验证**。  
- 更大规模公开 benchmark、与其他 speculative / draft 方案的横向对比 **没有**。  
- 严格的文本质量指标（edit distance vs 官方轨迹 / 人工抽检）报得不够系统（目前更偏墙钟与长度比）。

---

## 3. 我们自己怎么定位

| 定位 | 是否成立 |
| --- | --- |
| Windows 本地 OCR + 可选投机加速的 **可运行参考实现** | **是** |
| 8GB 下 speculative OCR **调度与测法的实验平台** | **是** |
| 已解决「无损高倍加速」 | **否** |
| 域续训已稳定抬 τ 到实用无损区 | **否** |

---

## 4. 想和社区一起讨论的问题

1. **在 τ≈1.2、eager verify 很贵时**，除了长度帽，还有哪些系统手段值得做？（更激进 skip draft、自适应停写、页级策略……）  
2. **free-run hard mining** 在 OCR 投机里，有没有比「存 features + reject 标签」更稳的标签定义？  
3. conf 校准（温度 / survival）对墙钟是否值得，还是应 **几乎全部资源砸 pos0**？  
4. 金标准页太少：怎样设计 **小而稳** 的公开页集，既可复现又不泄漏训练？  
5. 8GB 下是否应放弃「对齐大实验室无损 DFlash」，改为明确产品路线 **「完整优先的弱加速」**？

---

## 5. 复现与数据口径（讨论时请带上）

```text
# 环境
Windows + CUDA PyTorch（本机验证 2.11 + cu128 一类）
权重：Release 中 mini-uflash-win-domain-continue-best.pt（R5 系）
OCR：models/PaddlePaddle/Unlimited-OCR（HF baidu/Unlimited-OCR）

# 金标准对照
.venv\Scripts\python.exe -u train\bench_stable_vs_direct.py --manifest train\bench_manifest_fast.json
.venv\Scripts\python.exe -u train\bench_stable_vs_direct.py --manifest train\bench_manifest_lossless.json
```

请同时报告：**档位、是否 soft 截断、长度比、mean_acc、机器型号与 VRAM**，否则墙钟数字很难比。

---

## 6. 结束语

我们愿意公开说：

- **成果是「能跑、能量、在可截断设定下有墙钟收益」的完整栈**；  
- **不足是「接受率与分布对齐不足以支撑无损加速，训练尚未打通下一跳」**。

若你只看到 1.8×：请再看 lossless 的 0.95×。  
若你只看到失败训练：请再看固定页上的系统消融与成本拆分。

**欢迎 issue / discussion 拍砖；带复现条件的批评优先。**
