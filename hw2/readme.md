# CIFAR 训练与迁移命令说明

本目录提供两套模型脚本：
- DenseNet: [DenseNet.py](DenseNet.py)
- ResNeXt: [ResNeXt.py](ResNeXt.py)

默认数据目录为 `./data`（若本地不存在会自动下载 CIFAR-10 或 CIFAR-100）。

## 环境准备

在 `hw2` 目录下执行命令：

```bash
cd /home/chugaev/ANN/hw2
```

如需使用虚拟环境（可选）：

```bash
source ../env1/bin/activate
```

## DenseNet

### 1) 从头训练

```bash
python DenseNet.py --dataset cifar10
```

可指定常用参数：

```bash
python DenseNet.py \
  --dataset cifar10 \
  --data-dir ./data \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --save-path ./densenet_cifar10_best.pth \
  --log-csv-path ./densenet_train_log.csv
```

### 2) 第一阶段：在 CIFAR-100 上预训练

```bash
python DenseNet.py \
  --dataset cifar100 \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --save-path ./densenet_cifar100_pretrain.pth \
  --log-csv-path ./densenet_cifar100_log.csv
```

### 3) 第二阶段：迁移到 CIFAR-10（加载本地预训练 checkpoint）

```bash
python DenseNet.py \
  --dataset cifar10 \
  --pretrained-ckpt ./densenet_cifar100_pretrain.pth \
  --lr 0.01 \
  --epochs 100 \
  --batch-size 128 \
  --save-path ./densenet_cifar10_transfer.pth \
  --log-csv-path ./densenet_cifar10_transfer_log.csv
```

## ResNeXt

### 1) 从头训练

```bash
python ResNeXt.py --dataset cifar10
```

可指定常用参数：

```bash
python ResNeXt.py \
  --dataset cifar10 \
  --data-dir ./data \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --save-path ./resnext_cifar10_best.pth \
  --log-csv-path ./resnext_train_log.csv
```

### 2) 第一阶段：在 CIFAR-100 上预训练

```bash
python ResNeXt.py \
  --dataset cifar100 \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --save-path ./resnext_cifar100_pretrain.pth \
  --log-csv-path ./resnext_cifar100_log.csv
```

### 3) 第二阶段：迁移到 CIFAR-10（加载本地预训练 checkpoint）

```bash
python ResNeXt.py \
  --dataset cifar10 \
  --pretrained-ckpt ./resnext_cifar100_pretrain.pth \
  --lr 0.01 \
  --epochs 100 \
  --batch-size 128 \
  --save-path ./resnext_cifar10_transfer.pth \
  --log-csv-path ./resnext_cifar10_transfer_log.csv
```

## 说明

- 当前脚本已不再使用任何 torchvision 官方预训练权重。
- `--dataset`：选择 `cifar10` 或 `cifar100`。
- `--num-classes`：默认按数据集自动推断（`cifar10 -> 10`，`cifar100 -> 100`），通常不必手动设置。
- `--pretrained-ckpt`：加载本地预训练模型参数做迁移学习；会自动跳过类别头形状不匹配的参数。
- `--log-csv-path`：每个 epoch 记录一次指标到 CSV，字段为 `epoch, lr, train_loss, train_acc, test_loss, test_acc, best_acc`。
