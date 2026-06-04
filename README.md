# 项目结构说明

## 目录总览

```text
Final/
├── src/
│   ├── main.py                  # 训练入口脚本
│   ├── eval.py                  # 独立评估脚本（MAE/F-measure/S-measure等）
│   ├── topk_predict.py          # Top-K 预测可视化（按指标排序最佳样本）
│   ├── common/                  # 通用模块
│   │   ├── __init__.py
│   │   ├── config.py            # 全局配置（路径、超参数、平台切换）
│   │   ├── data.py              # 数据集构建、数据增强、DataLoader
│   │   ├── distributed.py       # 单卡/多卡训练评估工具
│   │   ├── train.py             # Trainer 训练与验证循环
│   │   └── visualization.py     # 训练曲线和预测结果可视化
│   └── model/                   # 模型定义
│       ├── __init__.py
│       ├── resnet18.py          # ResNet18 骨干网络（含预训练版本）
│       ├── poolnet.py           # PoolNet 基线
│       ├── poolnet_cfm.py       # + Cross-level Feature Module
│       ├── poolnet_ds.py        # + Deep Supervision
│       ├── poolnet_cfm_ds.py    # + CFM + Deep Supervision
│       ├── poolnet_fbda.py      # + Feature-level Boundary-aware DA
│       ├── poolnet_cfm_fbda.py  # + CFM + FBDA
│       ├── poolnet_cfm_ds_fbda.py  # + CFM + DS + FBDA
│       ├── poolnet_rrm.py       # + Residual Refinement Module
│       ├── poolnet_cfm_rrm.py   # + CFM + RRM
│       ├── poolnet_ca.py        # + Channel Attention
│       ├── poolnet_cfm_ca_rrm.py   # + CFM + CA + RRM
│       ├── poolnet_cfm_ga.py    # + CFM + Global Attention
│       ├── cpd.py               # CPD (Cascaded Partial Decoder)
│       ├── f3net.py             # F3Net 基线
│       ├── f3net_cbam.py        # + CBAM 注意力
│       └── f3net_aspp.py        # + ASPP 空洞空间金字塔池化
├── data/                        # 数据集目录（images/masks 结构）
│   ├── ECSSD/
│   ├── DUTS-TR/
│   ├── DUTS-TE/
│   ├── DUT-OMRON/
│   ├── HKUIS/
│   ├── PASCALS/
│   └── test/
├── outputs/                     # 训练输出（模型权重、日志、可视化）
│   ├── CPDResNet/
│   ├── F3Net/
│   ├── PoolNet/
│   └── ...                      # 每个模型一个子目录
├── report/                      # 实验报告（LaTeX）
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── fig/
├── doc/                         # 补充文档（选题说明等）
├── run.sh                       # 一键训练+评估脚本
└── README.md
```

## 快速使用

```bash
# 训练并评估指定模型（默认 PoolNetCFM）
bash run.sh PoolNetCFM

# 一键执行脚本
bash run.sh PoolNetCFM multi        # 多卡
bash run.sh PoolNetCFM single       # 单卡

# 单独训练
python src/main.py --model PoolNetCFM

# 单独多卡训练
python -m torch.distributed.run --standalone --nproc_per_node=2 src/main.py --model PoolNetCFM

# 单独评估
python src/eval.py --model PoolNetCFM

# Top-K 预测可视化
python src/topk_predict.py --model PoolNetCFM --topk 5 --metric mae
```

## 可用模型

| 模型名 | 说明 |
|--------|------|
| PoolNet | 基线模型 |
| PoolNetCFM | + Cross-level Feature Module |
| CPDResNet | Cascaded Partial Decoder |
| F3Net | F3Net 基线 |
| F3NetCBAM | + CBAM 注意力模块 |
| F3NetASPP | + ASPP 空洞空间金字塔池化 |
