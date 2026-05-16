# 项目结构说明

## 目录总览

```text
Final/
├── src/
│   ├── main.py                  # 训练入口脚本
│   ├── common/                  # 训练/数据/评估/可视化通用模块
│   │   ├── __init__.py
│   │   ├── data.py              # 数据集构建、数据增强、DataLoader
│   │   ├── train.py             # Trainer 训练与验证循环
│   │   ├── eval.py              # 评估相关逻辑（如独立评估流程）
│   │   └── visualization.py     # 训练曲线和预测结果可视化
│   └── model/                   # 模型定义
│       ├── __init__.py
│       ├── resnet18.py          # ResNet18/预训练骨干网络
│       ├── poolnet.py           # PoolNet 实现
│       └── basnet.py            # BASNet 实现
├── data/                        # 数据集目录（images/masks 结构）
│   ├── ECSSD/
│   ├── DUTS-TR/
│   ├── DUTS-TE/
│   ├── DUT-OMRON/
│   ├── HKUIS/
│   └── PASCALS/
├── report/                      # 实验报告与图表
│   ├── main.tex
│   ├── references.bib
│   └── fig/
├── doc/                         # 补充文档目录
└── README.md
```

## 代码模块关系

1. `src/main.py` 负责组装训练流程：读取超参数、构建数据、实例化模型、配置损失与优化器、启动训练并输出可视化结果。  
2. `src/common/data.py` 提供数据集读取与联合变换（图像与掩码同步增强），并返回训练/验证 DataLoader。  
3. `src/common/train.py` 封装 `Trainer`，统一管理 epoch 训练、验证指标计算、最佳模型保存和日志落盘。  
4. `src/model/` 下按模型拆分实现，便于在 `main.py` 中快速替换不同网络结构进行对比实验。       
