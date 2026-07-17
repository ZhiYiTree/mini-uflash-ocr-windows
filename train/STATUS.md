# 训练 / 加速状态

| 项 | 状态 |
| --- | --- |
| 脚手架 / 8GB 训练 | **就绪** |
| Teacher R3 | **完成**（547 train / 61 val） |
| Drafter R1 / R2 / R3 | **完成** |
| **P0+P1+P2 工程** | **已落地**（见下） |
| Drafter R4（Markov+TV） | **完成**（1000 steps） |
| 生产权重 | **R4 P012** |
| 墙钟阶段1 | **均值 1.12× / 中位 1.06×**（5 页，20260717_115544） |
| Drafter R5 pos0 | **完成**（2000 steps，B8 1.027 / pos0 51.7%） |
| 墙钟阶段2+R5 | **均值 1.07× / 中位 1.05×**（5 页，20260717_122633） |
| **金标准 manifest** | **已冻结** `train/bench_manifest.json` + `GOLD_BASELINE.md` |
| 金标准 v1 | 均值 1.23× / 最差 0.94×（123649） |
| **sys_v2 调度** | **均值 1.87× / 中位 1.92× / 最差 1.24×**（141454）**已晋升默认** |
| 同轨 100 页 R6 | **失败停训** |
| 生产权重 | **R5** + **sys_v2 引擎默认** |
| 三档 + conf 校准 | **已落地**（fast 1.69× / balanced 1.50× / lossless 0.95×） |
| Web UI 三档选择 | **已接入**（加速模式可见；默认 fast） |

## P0 / P1 / P2 落地清单（8GB 约束）

### P0 推理（`mini_uflash_engine.run_stable_dflash_mode`）
- 置信前缀截断（只裁 tail，避免软置信锁死 pure B1）
- 动态 γ（B4/B6/B8，结合 recent accept + gate）
- token 膨胀软停（弱 accept 长尾）
- 退化 / 周期 resync 保留
- **修复**：gate / 空闲显存不得每步强制 pure B1（曾导致 accept=0）

### P1 模型 + 训练
- **MarkovHiddenHead**（rank=64，hidden 残差，避免 129k×r 词表矩阵）
- **TV 主损失**（α_tv=0.9, α_ce=0.1）+ exp 位置衰减
- conf soft-label from TV
- 旧 checkpoint 非严格加载 + zero-init Markov

### P2 8GB
- multi-anchor 采样（`anchors_per_page=2`）
- OOM 恢复 → B1 + resync
- `empty_cuda` 容错
- 低显存时缩小 block，不永久关 draft
- bench dflash `max_length` cap 1536 减 KV 峰值

## R4 训练

- 目录：`train/runs/stage11b_win_continue_r4_p012/`
- steps：1000，micro 2 × accum 10，page_cache=3
- offline B8：`0.972 → 0.993` /7（score 1.011 → 1.033）
- 权重：`weights/mini-uflash-win-domain-continue-best.pt`  
  备份：`...-r4-p012-best.pt` / `...-r3-best.pt` / `...-r2-best.pt`

## 墙钟阶段1（系统止血，同 R4 权重）

目录：`train/runs/bench_stable_vs_direct_20260717_115544/`

策略：默认 **B4**、滚动低 τ 冷却 pure B1、**长度硬帽 512**、resync_every=192、取消 weak-accept resync。

| 页 | stable | dflash | 比值 | live acc | tok s/d |
| --- | ---: | ---: | ---: | ---: | ---: |
| p1 短文 | 9.7s | **8.8s** | **1.11×** | 1.52 | 137/213 |
| p2 长文 | 22.4s | **21.1s** | **1.06×** | 1.68 | 333/512 |
| p3 杂版 | 30.3s | **22.0s** | **1.38×** | 1.10 | 366/512 |
| p4 图多 | 11.7s | **11.1s** | **1.06×** | 1.17 | 100/263 |
| p5 正文 | 19.7s | 20.2s | 0.98× | 1.10 | 364/459 |
| **均值** | **18.8s** | **16.6s** | **1.12×** | **1.31** | |
| 中位 | | | **1.06×** | | |

历史对照：

| 版本 | 墙钟均值 | 说明 |
| --- | ---: | --- |
| R2 | ~0.73× | accept 低 |
| R3 | ~0.73× | accept 升、膨胀拖累 |
| R4 未止血 | ~0.80× | Markov+TV，p3 仍 0.52× |
| **阶段1** | **1.12×** | **4/5 页 ≥1.0×** |

说明：p3 仍可能比 stable 多 token（硬帽 512），但不再 55s 级失控；墙钟已明显好于 pure stable 均值。

## 阶段2（R5 pos0 续训）

- 目录：`train/runs/stage11b_win_continue_r5_pos0/`
- 策略：偏 B4 采样 (0.45/0.30/0.25)、`exp_gamma=3.5`、`pos0_boost=1.5`、TV+Markov、选模加 pos0 权重
- Teacher：沿用 R3（抽取已是 gundam crop=1024/640；未重抽全量以省 8GB 时间）
- offline best：B8 **1.027**/7，pos0 **51.7%**（R4 基线 pos0 50.3%、B8 0.993）
- 权重：`weights/mini-uflash-win-domain-continue-best.pt`（R5）+ `...-r5-pos0-best.pt`

### 墙钟阶段2（R5 + 阶段1 调度）

| 页 | stable | dflash | 比值 | live acc |
| --- | ---: | ---: | ---: | ---: |
| p1 | 7.1s | **6.7s** | **1.05×** | 1.50 |
| p2 | 14.9s | **14.6s** | **1.03×** | 1.68 |
| p3 | 21.2s | **17.1s** | **1.24×** | 1.10 |
| p4 | 8.4s | 8.8s | 0.95× | 1.17 |
| p5 | 15.4s | **14.3s** | **1.08×** | 1.29 |
| **均值** | **13.4s** | **12.3s** | **1.07×** | **1.35** |
| 中位 | | | **1.05×** | |

对照：阶段1 均值 1.12× / acc 1.31；阶段2 均值 1.07× / acc **1.35**。  
离线 pos0 小幅上升，live acc 略升；墙钟比值与阶段1 同量级（跑次绝对耗时有波动）。  
**结论：1.0× 已站稳；1.2～1.3× 均值仍需更高 live τ 或更轻 verify。**

## 同轨 100 页实验（R6）— 失败结案

- 抽取：`train/data/teachers/online_r6` **100/100 OK**（与 gundam/bench max_length=2048 对齐）
- 训练：`stage11b_win_continue_r6_online` 1000 steps，90/10
- 离线：best **未优于 baseline**；final 略差；无 `drafter_best.pt`
- 金标准 live（权重仍为 R5）：均值 **1.00×**，acc **1.35**（`20260717_135230`）
- **按协议停训**，不扩 hard mining / 不扩数据
- 下一步方向：**系统侧**（压长度比、降 B1 回退成本）或 **重审 teacher=free-run 闭环**，而非再盲训

## 三档产品 + DSpark conf 校准（2026-07-17）

| 组件 | 路径 |
| --- | --- |
| 档位预设 | `webapp/dflash_tiers.py` |
| 引擎 `tier=` | `run_stable_dflash_mode(..., tier=)` |
| conf 校准脚本 | `train/calibrate_conf.py`（8GB 增量 path） |
| 校准表 | `train/conf_calibration.json` |
| manifest | `bench_manifest_{fast,balanced,lossless}.json` |

| 档 | 均值墙钟 | 最差 | 长度比 | 说明 |
| --- | ---: | ---: | ---: | --- |
| **fast** | **1.69×** | **1.24×** | 1.27 | soft 截断，墙钟优先 |
| **balanced** | **1.50×** | 0.88× | 1.45 | 中等 soft 帽 |
| **lossless** | **0.95×** | 0.90× | **1.80** | 无 soft；当前 **不加速**（诚实） |
| sys_v2 历史 | **1.87×** | 1.24× | 1.27 | 生产默认仍可用 |

**结论**：无损档在 live τ≈1.1–1.4 下靠 conf 调度 **抬不了墙钟**；要混元式无损加速必须先抬 accept，禁止用 soft 帽伪装。详见 `GOLD_BASELINE.md`。

## 产物路径

```
webapp/dflash_tiers.py                    # fast/balanced/lossless + schedule_verify_len
webapp/mini_uflash_engine.py              # tier= + conf schedule
train/calibrate_conf.py / conf_calibration.json
train/bench_manifest_{fast,balanced,lossless}.json
weights/mini-uflash-win-domain-continue-best.pt  # R5
```
