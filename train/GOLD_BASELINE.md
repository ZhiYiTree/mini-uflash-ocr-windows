# 金标准基线（三档 + sys_v2）

## 怎么跑

```text
cd D:\Python\OCR
# 生产默认（sys_v2 显式参数，与 tier=fast 同思路）
.venv\Scripts\python.exe -u train\bench_stable_vs_direct.py --manifest train\bench_manifest.json

# 三档产品预设
.venv\Scripts\python.exe -u train\bench_stable_vs_direct.py --manifest train\bench_manifest_fast.json
.venv\Scripts\python.exe -u train\bench_stable_vs_direct.py --manifest train\bench_manifest_balanced.json
.venv\Scripts\python.exe -u train\bench_stable_vs_direct.py --manifest train\bench_manifest_lossless.json
```

对照旧调度：`--manifest train\bench_manifest_v1_archive.json`

规则：同一页集下 **每次只改一个变量**。

## 固定页集

warmup + 5 scored 页（见各 `bench_manifest*.json`）。

## 三档定义（`webapp/dflash_tiers.py`）

| 档 | 意图 | soft 截断 | 整页降级 | verify 调度 |
| --- | --- | --- | --- | --- |
| **fast** | 墙钟优先 | hard 384 / soft 280 | 是（6 次零接受） | conf + expected skip |
| **balanced** | 速度/完整折中 | hard 480 / soft 360 | 是（10） | conf schedule |
| **lossless** | 对齐混元「无损」叙事 | **关**（仅 max_length） | **关** | 校准 conf → verify_len |

校准表：`train/conf_calibration.json`（`train/calibrate_conf.py` 拟合）。

## 三档对照（2026-07-17，R5 权重，同金标准页）

| 档 | 均值墙钟 | 中位 | 最差 | mean_acc | 长度比 | 跑次 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **fast** | **1.69×** | **1.65×** | **1.24×** | 1.22 | **1.27** | `..._145421` |
| **balanced** | **1.50×** | **1.39×** | 0.88× | 1.22 | 1.45 | `..._145655` |
| **lossless** | **0.95×** | 0.93× | 0.90× | 1.14 | **1.80** | `..._145108` |
| sys_v2 基线 | **1.87×** | **1.92×** | **1.24×** | 1.24 | **1.27** | `..._141454` |
| gold v1（旧调度） | 1.23× | 0.99× | 0.94× | 1.35 | 1.68 | `..._123649` |

### 解读（客观）

1. **fast ≈ sys_v2**：墙钟来自 soft 帽 + 早降级；本轮 fast 1.69× 略低于历史 sys_v2 1.87×（跑次波动 + conf 校准后的调度差异），仍远高于 gold v1。
2. **balanced**：软帽放宽 → 墙钟略降、长度比上升；最差页（图多 p4）可能 <1.0×。
3. **lossless 诚实结论**：无 soft 截断时 live τ≈1.1–1.4，**长度膨胀到 ~1.8×**，墙钟 **跑不赢** stable。  
   不是实现 bug，是 **接受率不够高时「无损加速」在 8GB eager verify 下不成立**。  
   要 lossless 真加速，必须抬 live τ（hard-mine 再训）或更轻的 verify 路径——**不要靠再堆 soft 帽伪装无损**。

### Conf 校准摘要

- 6 页 × 36 步，216 rounds（8GB 增量 step/crop，无每步 prefill）
- pos0 emp accept **0.65** / raw conf **0.61** → T≈1.3
- pos1 emp **0.69** / conf **0.65** → T=1.0
- pos2 emp **0.20** / conf **0.17** → T=1.0
- `pos0_skip_below≈0.23`，`conf_floor=0.20`

## 生产建议

| 产品选择 | 用哪档 |
| --- | --- |
| 要墙钟、可接受软截断 | **fast** / 默认 `bench_manifest.json`（sys_v2） |
| 要更完整一点 | **balanced**，接受均值 ~1.5×、个别页可能慢 |
| 要完整、对齐稳定输出长度 | **lossless**，但 **当前不保证加速**（约 0.95×） |

API：`run_stable_dflash_mode(..., tier="fast"|"balanced"|"lossless")`。

## 训练侧（R6）回顾

同轨 100 页 + 1k step **失败停训**。墙钟提升来自 **系统调度**，不是新权重。

## 下一步（不盲训）

1. 产品默认：**fast**；UI 暴露三档。  
2. lossless 要提速：free-run hard mining 新分支，或降 verify 成本——**禁止**用 soft 截断冒充 lossless。  
3. 扫参只动 `length_soft_cap` / tier，同页集 A/B。
