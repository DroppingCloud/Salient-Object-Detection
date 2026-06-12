# 项目结构说明

## 目录总览

```text
Final/
├── src/
│   ├── main.py                  # 训练入口脚本
│   ├── eval.py                  # 独立评估脚本（MAE/F-measure/S-measure等）
│   ├── topk_predict.py          # Top-K 预测可视化（按指标排序最佳样本）
│   ├── run.sh                   # 一键训练+评估脚本
│   ├── common/                  # 通用模块
│   │   ├── config.py            # 全局配置（路径、超参数、平台切换）
│   │   ├── data.py              # 数据集构建、数据增强、DataLoader
│   │   ├── distributed.py       # 单卡/多卡训练评估工具
│   │   ├── train.py             # Trainer 训练与验证循环
│   │   └── visualization.py     # 训练曲线和预测结果可视化
│   ├── model/                   # 模型定义
│   │   ├── resnet.py            # ResNet 骨干网络（18/34/50，含预训练版本）
│   │   ├── poolnet.py           # PoolNet 基线
│   │   ├── poolnet_cfm.py       # + Cross-level Feature Module
│   │   ├── poolnet_gate.py      # + Gate 门控机制
│   │   ├── poolnet_gate_cfm.py  # + Gate + CFM
│   │   ├── poolnet_cfi.py       # + Cross-level Feature Integration
│   │   ├── poolnet_cfm_cbam.py  # + CFM + CBAM 注意力
│   │   ├── poolnet_ra.py        # + Reverse Attention
│   │   ├── poolnet_aspp.py      # + ASPP 空洞空间金字塔池化
│   │   ├── f3net.py             # F3Net 基线
│   │   ├── f3net_cbam.py        # + CBAM 注意力
│   │   ├── f3net_aspp.py        # + ASPP 空洞空间金字塔池化
│   │   ├── f3net_ds.py          # + Deep Supervision
│   │   ├── f3net_cfm.py         # + Cross-level Feature Module
│   │   ├── f3net_ppm.py         # + Pyramid Pooling Module
│   │   ├── cpd.py               # CPD (Cascaded Partial Decoder)
│   │   ├── gatenet.py           # GateNet 基线
│   │   ├── gatenet_cbam.py      # + CBAM 注意力
│   │   ├── gatenet_ds.py        # + Deep Supervision
│   │   └── basnet.py            # BASNet (Boundary-Aware Segmentation)
│   └── analysis/                # 可视化分析工具
│       ├── visualize_backbone_features.py
│       ├── visualize_dual_gate.py
│       └── visualize_skip_gate_features.py
├── data/                        # 数据集目录（images/masks 结构）
│   ├── ECSSD/
│   ├── DUTS-TR/
│   ├── DUTS-TE/
│   ├── DUT-OMRON/
│   ├── HKUIS/
│   ├── PASCALS/
│   └── test/
├── outputs/                     # 训练输出（模型权重、日志、可视化）
├── checkpoints/                 # 模型检查点
├── ablation/                    # 消融实验数据
├── report/                      # 实验报告（LaTeX）
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── fig/
├── doc/                         # 补充文档（选题说明等）
└── README.md
```

## 快速使用

```bash
# 训练并评估指定模型（默认 PoolNetCFM）
bash src/run.sh PoolNetCFM

# 一键执行脚本
bash src/run.sh PoolNetCFM multi        # 多卡
bash src/run.sh PoolNetCFM single       # 单卡

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
| **PoolNet 系列** | |
| PoolNet | 基线模型 |
| PoolNetCFM | + Cross-level Feature Module |
| PoolNetGate | + Gate 门控机制 |
| PoolNetGateCFM | + Gate + CFM |
| PoolNetCFI | + Cross-level Feature Integration |
| PoolNetCFM_CBAM | + CFM + CBAM 注意力 |
| PoolNetRA | + Reverse Attention |
| **F3Net 系列** | |
| F3Net | 基线模型 |
| F3NetCBAM | + CBAM 注意力模块 |
| F3NetASPP | + ASPP 空洞空间金字塔池化 |
| F3NetDS | + Deep Supervision |
| F3NetCFM | + Cross-level Feature Module |
| F3NetPPM | + Pyramid Pooling Module |
| **GateNet 系列** | |
| GateNet | 基线模型 |
| GateNetCBAM | + CBAM 注意力 |
| GateNetDS | + Deep Supervision |
| **其他** | |
| CPDResNet | Cascaded Partial Decoder |
| BASNet | Boundary-Aware Segmentation Network |
