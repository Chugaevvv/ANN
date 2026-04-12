"""DenseNet in PyTorch.

该实现主要面向 CIFAR 尺寸输入 (3x32x32)。
"""
import argparse
import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        # DenseNet-B 的瓶颈结构：
        # 1x1 卷积先将通道扩展到 4*k，再用 3x3 卷积压回 k，
        # 其中 k 即 growth_rate，表示该层“新产生”的特征通道数。
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # 输入: [N, C_in, H, W]
        # bn1 + relu + conv1: [N, C_in, H, W] -> [N, 4k, H, W]
        out = self.conv1(F.relu(self.bn1(x)))
        # bn2 + relu + conv2: [N, 4k, H, W] -> [N, k, H, W]
        out = self.conv2(F.relu(self.bn2(out)))
        # Dense 连接: 将“新特征 out”与“历史特征 x”在通道维拼接
        # [N, k, H, W] cat [N, C_in, H, W] -> [N, C_in + k, H, W]
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        # Transition 层用于两个目标：
        # 1) 1x1 卷积压缩通道数 (compression)
        # 2) 2x2 平均池化将空间分辨率减半 (downsample)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        # [N, C_in, H, W] -> [N, C_out, H, W]
        out = self.conv(F.relu(self.bn(x)))
        # [N, C_out, H, W] -> [N, C_out, H/2, W/2]
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # Stem: 3x3 卷积，保持分辨率不变。
        # 对 CIFAR 输入 3x32x32，输出为 (2k)x32x32。
        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        # Dense Block 1:
        # 每个 bottleneck 额外增加 k 个通道，经过 nblocks[0] 层后：
        # C = C + nblocks[0] * k
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        # Transition 1: 通道压缩到 floor(C * reduction)，并将 H,W 减半。
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # Dense Block 2 + Transition 2
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # Dense Block 3 + Transition 3
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # Dense Block 4: 最后一个 Dense Block 后不再下采样。
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        # 分类头: BN + ReLU + 全局平均池化 + 全连接。
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            # 第 i 个 bottleneck 接收当前累积通道 in_planes，输出后通道 +growth_rate
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 3, 32, 32]
        out = self.conv1(x)
        # dense1 后通道增加，再经 trans1 下采样到 16x16
        out = self.trans1(self.dense1(out))
        # dense2 + trans2: 下采样到 8x8
        out = self.trans2(self.dense2(out))
        # dense3 + trans3: 下采样到 4x4
        out = self.trans3(self.dense3(out))
        # dense4: 仅做特征累积，不改分辨率
        out = self.dense4(out)
        # 4x4 平均池化相当于全局池化: [N, C, 4, 4] -> [N, C, 1, 1]
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        # 展平为 [N, C]
        out = out.view(out.size(0), -1)
        # 线性分类到 num_classes
        out = self.linear(out)
        return out

def DenseNet121():
    # block 配置与论文命名一致，growth_rate=32
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar(num_classes=10):
    # CIFAR 常用轻量配置
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=num_classes)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 loss 与准确率。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估，返回平均 loss 与准确率。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def build_model_and_transforms(num_classes):
    # 仅保留 CIFAR 32x32 训练/迁移流程，不再使用任何官方预训练权重。
    model = densenet_cifar(num_classes=num_classes)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, train_transform, test_transform


def load_pretrained_checkpoint(model, ckpt_path, device):
    """加载本地 checkpoint 用于迁移学习，自动跳过形状不匹配的分类头参数。"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()

    filtered_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped_keys.append(key)

    incompatible = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded pretrained checkpoint: {ckpt_path}")
    print(f"Matched params: {len(filtered_state)}")
    if skipped_keys:
        print(f"Skipped params (shape/key mismatch): {len(skipped_keys)}")
    if incompatible.missing_keys:
        print(f"Missing keys after load: {len(incompatible.missing_keys)}")


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet on CIFAR")
    parser.add_argument("--data-dir", default="./data", type=str, help="CIFAR 数据目录")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], help="选择训练数据集")
    parser.add_argument("--num-classes", default=None, type=int, help="类别数；默认按数据集自动设置")
    parser.add_argument("--pretrained-ckpt", default="", type=str, help="本地预训练 checkpoint 路径，用于迁移学习")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--batch-size", default=128, type=int, help="批大小")
    parser.add_argument("--lr", default=0.1, type=float, help="初始学习率")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="权重衰减")
    parser.add_argument("--num-workers", default=2, type=int, help="DataLoader 线程数")
    parser.add_argument("--save-path", default="./densenet_cifar10_best.pth", type=str, help="最佳模型保存路径")
    parser.add_argument("--log-csv-path", default="./densenet_train_log.csv", type=str, help="每轮训练日志输出的 CSV 路径")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    args = parser.parse_args()

    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == "cifar10" else 100

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, train_transform, test_transform = build_model_and_transforms(args.num_classes)

    dataset_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    train_set = dataset_cls(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_set = dataset_cls(root=args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    if args.pretrained_ckpt:
        # 迁移学习入口：例如先用 CIFAR-100 预训练，再在 CIFAR-10 上继续训练。
        load_pretrained_checkpoint(model, args.pretrained_ckpt, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    # 经典 CIFAR 训练策略：在中后期降低学习率。
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 80],
        gamma=0.1,
    )

    best_acc = 0.0
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 为了便于作图可视化，每次运行时重写 CSV 并写入表头。
    csv_dir = os.path.dirname(args.log_csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(args.log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "best_acc"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"lr={current_lr:.5f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
            print(f"Saved new best model to {args.save_path} (acc={best_acc:.2f}%)")

        # 每个 epoch 追加一行指标，便于后续用 pandas/matplotlib 画曲线。
        with open(args.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{current_lr:.8f}",
                f"{train_loss:.6f}",
                f"{train_acc:.4f}",
                f"{test_loss:.6f}",
                f"{test_acc:.4f}",
                f"{best_acc:.4f}",
            ])

    print(f"Training done. Best test acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()