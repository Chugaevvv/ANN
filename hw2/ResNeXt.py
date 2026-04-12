"""ResNeXt in PyTorch.

核心思想: 在残差块中使用分组卷积 (grouped convolution)，
通过 cardinality(组数) 提升表示能力，而不单纯依赖更深或更宽。

该实现主要适配 CIFAR 输入尺寸 (3x32x32)。
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Block(nn.Module):
    """ResNeXt 基本残差块 (bottleneck + grouped conv)。"""
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        # group_width = C * D
        # C: cardinality(分组数), D: 每组宽度(bottleneck_width)
        # grouped conv 的总输入/输出通道都是 group_width。
        group_width = cardinality * bottleneck_width

        # 1x1 降/升维: in_planes -> group_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        # 3x3 分组卷积:
        # groups=cardinality 表示把通道分成 C 组，每组独立卷积。
        # stride 由 stage 控制，stride=2 时做空间下采样。
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)

        # 1x1 线性投影到 expansion * group_width
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        # Shortcut 分支:
        # 若 stride!=1 或 通道不匹配，则用 1x1 卷积 + BN 对齐维度。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        # 主分支:
        # [N, C_in, H, W] -> [N, C*D, H, W]
        out = F.relu(self.bn1(self.conv1(x)))
        # [N, C*D, H, W] -> [N, C*D, H/stride, W/stride]
        out = F.relu(self.bn2(self.conv2(out)))
        # [N, C*D, ...] -> [N, 2*C*D, ...]
        out = self.bn3(self.conv3(out))

        # 与 shortcut 残差相加后再激活
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        # 当前 stage 输入通道，随着 stage 逐步更新。
        self.in_planes = 64

        # Stem: 对 CIFAR 使用轻量 1x1 卷积起始映射到 64 通道。
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 三个 stage，每个 stage 由若干 Block 组成。
        # layer1 首块 stride=1，不降采样；layer2/3 首块 stride=2，下采样。
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)

        # 线性层输入通道推导:
        # 最后一个 stage 结束后通道为 cardinality * bottleneck_width * expansion，
        # 其中 bottleneck_width 在每个 stage 结束后 *2，三段后系数为 8。
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

    def _make_layer(self, num_blocks, stride):
        # 该 stage 的第一个块使用给定 stride，后续块 stride=1。
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            # 每个 Block 输出通道固定为 expansion * cardinality * bottleneck_width
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 3, 32, 32]
        out = F.relu(self.bn1(self.conv1(x)))
        # layer1: 通常保持 32x32
        out = self.layer1(out)
        # layer2: 首块 stride=2 -> 16x16
        out = self.layer2(out)
        # layer3: 首块 stride=2 -> 8x8
        out = self.layer3(out)
        # out = self.layer4(out)
        # 8x8 平均池化，相当于全局池化到 1x1
        out = F.avg_pool2d(out, 8)
        # [N, C, 1, 1] -> [N, C]
        out = out.view(out.size(0), -1)
        # 分类 logits
        out = self.linear(out)
        return out


def ResNeXt29_2x64d(num_classes=10):
    # 总深度 29 的常见 CIFAR 版本：cardinality=2, bottleneck_width=64
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, num_classes=num_classes)

def ResNeXt29_4x64d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, num_classes=num_classes)

def ResNeXt29_8x64d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, num_classes=num_classes)

def ResNeXt29_32x4d(num_classes=10):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

def test_resnext():
    net = ResNeXt29_2x64d()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())


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
    model = ResNeXt29_2x64d(num_classes=num_classes)
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
    parser = argparse.ArgumentParser(description="Train ResNeXt on CIFAR")
    parser.add_argument("--data-dir", default="./data", type=str, help="CIFAR 数据目录")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], help="选择训练数据集")
    parser.add_argument("--num-classes", default=None, type=int, help="类别数；默认按数据集自动设置")
    parser.add_argument("--pretrained-ckpt", default="", type=str, help="本地预训练 checkpoint 路径，用于迁移学习")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--batch-size", default=128, type=int, help="批大小")
    parser.add_argument("--lr", default=0.1, type=float, help="初始学习率")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="权重衰减")
    parser.add_argument("--num-workers", default=2, type=int, help="DataLoader 线程数")
    parser.add_argument("--save-path", default="./resnext_cifar10_best.pth", type=str, help="最佳模型保存路径")
    parser.add_argument("--log-csv-path", default="./resnext_train_log.csv", type=str, help="每轮训练日志输出的 CSV 路径")
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
    # 参考 DenseNet 的训练节奏：中后期降低学习率。
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