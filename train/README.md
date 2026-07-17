# Mini UFlash 本机训练（Windows 8GB）

在 **RTX 5060 Laptop ~8GB** 上从 Stage 11B 权重做 **域适配续训**。  
本目录是脚手架：**默认不启动任何训练**。

## 目录

```
train/
  config.py                 # 路径与 8GB 默认超参
  extract_teachers.py       # 页图 → teacher.pt（逐页、可断点）
  train_continue_8g.py      # 小 batch + 梯度累积 + 懒加载续训
  collect_pages.py          # 从 webapp/outputs 收集已有页图
  make_split.py             # 写 train/val 划分
  run_prepare.ps1           # 建目录、检查模型/权重（安全）
  run_extract.ps1           # 抽取入口
  run_train.ps1             # 训练入口
  run_full_pipeline.ps1     # 一键流水线（确认后再用）
  data/pages/pool|train|val
  data/teachers/train|val
  data/payloads/
  data/splits/
  runs/                     # checkpoint 与报告
  lib/                      # losses / metrics / teacher 缓存
```

## 设计要点（适配没租号）

| 点 | 做法 |
| --- | --- |
| 显存 | 抽 teacher 一次一页；训练只保留 emb + LM head + 8M drafter |
| 内存 | teacher **懒加载 LRU 缓存**，不全量进 16GB RAM |
| Batch | micro-batch=2，grad-accum=16 → 有效 batch≈32 |
| 起点 | 必须从 `weights/...stage11b-best.pt` **续训** |
| 验收 | 离线 B8 `mean_accepted_draft`，不是 Direct 端到端乱码 |

## 你现在可以做的（不训练）

在项目根目录：

```powershell
.\train\run_prepare.ps1
```

可选：只收集已有页图到 pool（复制文件，不跑 GPU）：

```powershell
.\.venv\Scripts\python.exe train\collect_pages.py --dry-run
.\.venv\Scripts\python.exe train\collect_pages.py
```

## 确认「开始训练」之后的顺序

```powershell
# 1) 抽 teacher（先小后大）
.\train\run_extract.ps1 -Limit 10 -DryRun
.\train\run_extract.ps1 -Limit 10

# 2) 划分
.\.venv\Scripts\python.exe train\make_split.py

# 3) 训练 dry-run → 真训
.\train\run_train.ps1 -DryRun
.\train\run_train.ps1 -Steps 3000

# 或一键（通宵）
.\train\run_full_pipeline.ps1 -ExtractLimit 120 -TrainSteps 3000
```

## 输出权重

成功后：

- `train/runs/stage11b_win_continue/drafter_best.pt`
- 可复制到 `weights/` 或设置 `MINI_UFLASH_WEIGHT` 给 webapp 使用

## 与 lab 数字的关系

本流水线目标是：**在你自己的页面分布上** 把 B8 接受率从 ~1 往 3～4 抬。  
不保证端到端 Direct 无损；在线加速路径仍建议用安全提交，而不是当前 Direct crop。

## 状态

见 `STATUS.md`。当前：**60 页 teacher 已抽取，Stage 11B 本机续训已完成**；最佳权重在  
`weights/mini-uflash-win-domain-continue-best.pt`（webapp 会优先发现该文件）。
