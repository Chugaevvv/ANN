"""
CycleGAN: 非配对图像风格迁移
==============================
论文: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
       Zhu et al., ICCV 2017

核心思想：
  - 域 A → 域 B 和 域 B → 域 A 两个生成器互相约束
  - 循环一致性损失 (cycle-consistency loss) 保证图片翻译一圈后回到原点
  - 不需要成对训练数据，只需两组风格不同的图片

框架组成:
  1. Generator  (ResNet-9blocks + resize-conv 上采样)
  2. Discriminator (PatchGAN 70x70)
  3. 损失函数 (LSGAN + Cycle + Identity)
  4. Image Buffer (缓解判别器过拟合)
  5. 训练循环
"""

import os
import itertools
import random
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image  # noqa: F811


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       1. 超参数配置                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Config:
    # 数据 — 使用官方 CycleGAN 数据集, 运行时自动下载
    dataset = "apple2orange"               # 数据集名称, 可选见下方列表, 也可传 --dataset
    # 官方 CycleGAN 数据集一览 (均为非配对, 无需标注):
    #   apple2orange       苹果↔橙子 (995/1019 张, 推荐入门, 最直观)
    #   orange2apple       橙子↔苹果 (同上反向)
    #   horse2zebra        马↔斑马 (1067/1334 张, 经典, 稍慢)
    #   summer2winter_yosemite  优胜美地夏↔冬 (1231/962 张)
    #   monet2photo        莫奈画↔照片 (1072/6287 张, 域B太大不推荐笔记本跑)
    #   vangogh2photo      梵高画↔照片 (400/6287 张)
    #   cezanne2photo      塞尚画↔照片 (583/6287 张)
    #   ukiyoe2photo       浮世绘↔照片 (673/6287 张)
    #   maps               卫星图↔地图 (1096/1096 张)
    #   facades            建筑立面↔标签 (400/400 张, 配对数据)
    #   cityscapes         街景↔语义分割 (2975/2975 张, 配对数据)
    download_root = "data"                 # 数据集下载到此目录

    image_size  = 128                     # 训练尺寸: 128×128 (4060 笔记本 2h 内可完成)
    # 改动说明: image_size 从 256 降到 128, 每张图计算量降为原来的 ~1/4,
    # 同时减少显存占用, 使 RTX 4060 Laptop (8GB VRAM) 能在 2 小时内完成训练

    input_nc    = 3                       # 输入通道数 (RGB=3)
    output_nc   = 3                       # 输出通道数 (RGB=3)
    batch_size  = 1                       # CycleGAN 常用 batch_size=1, InstanceNorm 不依赖 batch 统计
    num_workers = 2                       # DataLoader 进程数 (笔记本 CPU 核心较少, 用 2 足够)

    # Generator
    ngf     = 64                          # 生成器第一层卷积通道数
    n_res   = 6                           # ResNet 残差块数 (128×128 用 6 块, 256×256 用 9 块)
    # 改动说明: 使用 resize-conv (最近邻上采样 + 普通卷积) 替代 ConvTranspose2d,
    # 这是为了从根源上避免棋盘伪影 (checkerboard artifacts)

    # Discriminator (PatchGAN 70×70)
    ndf     = 64                          # 判别器第一层卷积通道数
    n_layers_D = 3                        # PatchGAN 卷积层数 (3 → 感受野 ~70×70)

    # 训练 — 针对 4060 Laptop 调优, 目标 1~2 小时完成
    n_epochs        = 50                  # 总 epoch 数 (50 epoch 约 1~1.5h, 足够看到效果)
    n_epochs_decay  = 25                  # 后 25 个 epoch 学习率线性衰减到 0
    # 改动说明: 原论文 200 epochs, 这里减为 50。apple2orange 数据集在 30 epochs
    # 左右就能看到清晰的风格转换效果, 50 epochs 对课程作业已经足够
    lr              = 0.0002             # 初始学习率 (与论文一致)
    beta1           = 0.5                # Adam beta1 (0.5 比默认 0.9 在 GAN 中更稳定)
    pool_size       = 50                 # 图像缓冲区大小, 用于减少判别器振荡
    lambda_cycle    = 10.0               # 循环一致性损失权重 (论文推荐 10.0)
    lambda_identity = 0.5                # 身份损失权重, 缓解颜色偏移 (color shift)
    # 改动说明: lambda_identity 作为辅助损失, 当输入图已接近目标域时强制
    # 生成器不改变图片内容, 有效防止意外色调偏移

    # 保存
    save_root       = "checkpoints/cyclegan"
    sample_interval = 5                  # 每隔多少 epoch 保存一次生成样本


cfg = Config()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       2. 数据集自动下载                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def download_cyclegan_dataset(dataset_name, root="data"):
    """
    从官方 CycleGAN 仓库下载并解压数据集到 root/<dataset_name>/ 目录。
    下载一次后缓存, 不会重复下载。

    数据来源: http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/
    """
    from urllib.request import urlretrieve
    import zipfile

    dataset_dir = os.path.join(root, dataset_name)
    trainA_dir = os.path.join(dataset_dir, "trainA")
    trainB_dir = os.path.join(dataset_dir, "trainB")

    # 已存在则跳过下载
    if os.path.isdir(trainA_dir) and os.path.isdir(trainB_dir):
        print(f"[Data] 数据集已存在: {dataset_dir}")
        return trainA_dir, trainB_dir

    # 下载
    url = f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset_name}.zip"
    zip_path = os.path.join(root, f"{dataset_name}.zip")
    os.makedirs(root, exist_ok=True)

    print(f"[Data] 正在下载 {dataset_name} 数据集...")
    print(f"      地址: {url}")
    urlretrieve(url, zip_path)

    # 解压
    print(f"[Data] 正在解压到 {root}/...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(root)
    os.remove(zip_path)

    if not os.path.isdir(trainA_dir) or not os.path.isdir(trainB_dir):
        raise RuntimeError(f"解压后未找到 trainA/trainB 目录, 请检查 {dataset_dir}/ 内容")
    print(f"[Data] 下载完成: A={trainA_dir}, B={trainB_dir}")
    return trainA_dir, trainB_dir


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       3. 图像数据集                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ImageDataset(Dataset):
    """
    从给定目录加载所有图片 (支持 jpg / png)。
    注意: 域 A 和 域 B 各使用一个独立的 ImageDataset 实例,
    shuffle 之后随机配对, 不需要一一对应。
    """

    def __init__(self, root, image_size=256, mode="train"):
        self.paths = sorted(glob(os.path.join(root, "*.jpg")) +
                            glob(os.path.join(root, "*.png")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"未找到图片文件, 请检查目录: {root}")

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize(image_size + 30),                            # 先放大一点
                transforms.RandomCrop(image_size),                            # 随机裁剪 (数据增强)
                transforms.RandomHorizontalFlip(),                            # 随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),     # [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.paths)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       3. Generator (ResNet 生成器)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ResidualBlock(nn.Module):
    """
    残差块: Conv → InstanceNorm → ReLU → Conv → InstanceNorm → 跳跃连接
    InstanceNorm 而非 BatchNorm: 风格迁移任务中每个实例的统计量应当独立,
    使用 BN 会导致 batch 内图片之间互相干扰 (产生颜色偏移)。
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  # 跳跃连接


class Generator(nn.Module):
    """
    CycleGAN 生成器: 编码器-残差块-解码器 结构

    下采样: 2 个 stride-2 卷积 (特征图尺寸 → 1/4)
    残差块: 9 个残差块 (仅做变换, 不改变尺寸)
    上采样: 2 个 resize-conv 块 (特征图尺寸 → 4×)
    输出: 反射填充 + 7×7 卷积 + Tanh

    关键设计: 上采样使用 最近邻插值 + 普通卷积 (resize-convolution)
    而非转置卷积。原因见下方说明。
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_res=9):
        super().__init__()

        # ---- 编码器: 初始卷积 → 下采样×2 ----
        self.enc_conv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

        # ---- 残差块: 9 个 (不做下/上采样, 仅变换) ----
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf * 4) for _ in range(n_res)])

        # ---- 上采样: 2 个 resize-conv 块 ----
        # 不使用 ConvTranspose2d, 而是在 forward 中手动插值+卷积。
        # 原因: 转置卷积的"不均匀重叠"会产生棋盘格伪影 (checkerboard artifacts),
        # 即输出图像上出现规律的网格状花纹。resize-conv 则完全避免了这一问题,
        # 代价极小 (几乎不影响性能)。参见 Odena et al., "Deconvolution and Checkerboard Artifacts".
        self.up1 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 2, 3, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # ---- 输出层: 7×7 卷积 + Tanh → [-1, 1] ----
        self.out_conv = nn.Conv2d(ngf, output_nc, 7, padding=3, padding_mode="reflect")
        self.out_tanh = nn.Tanh()

    def forward(self, x):
        # 编码器 (下采样)
        x = self.enc_conv(x)     # (B, 64,  256, 256)
        x = self.down1(x)        # (B, 128, 128, 128)
        x = self.down2(x)        # (B, 256, 64,  64)

        # 残差变换
        x = self.res_blocks(x)   # (B, 256, 64, 64)

        # 解码器 (resize-conv 上采样: 最近邻插值 + 普通卷积)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # (B, 256, 128, 128)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # (B, 128, 256, 256)
        x = self.up2(x)

        # 输出
        x = self.out_conv(x)
        x = self.out_tanh(x)
        return x


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       4. Discriminator (PatchGAN)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class Discriminator(nn.Module):
    """
    PatchGAN 判别器: 不对整张图输出真/假, 而是输出一个 N×N 的判定矩阵,
    每个网格值判断原图中对应的 70×70 patch 是否为真。

    优点:
      - 参数量少, 可以处理任意尺寸输入
      - 更关注局部纹理/风格, 而非全局结构
      - 可以有效缓解模式崩溃 (mode collapse), 因为每个 patch 独立判断

    感受野计算 (3 层卷积, kernel=4, stride=2): ≈ 70×70
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()

        # 第一层不加 InstanceNorm (参考论文 + pytorch-CycleGAN 官方实现)
        layers = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mult = 1
        mult_next = 2
        for n in range(1, n_layers):
            layers += [
                nn.Conv2d(ndf * mult, ndf * mult_next, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * mult_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            mult = mult_next
            mult_next = min(2 ** (n + 1), 8)  # 最多乘到 8

        # 倒数第二层: stride=1 (保持特征图尺寸, 增加感受野)
        layers += [
            nn.Conv2d(ndf * mult, ndf * mult_next, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * mult_next),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # 输出层: 1 通道, 不需要 Sigmoid (LSGAN 用 MSE loss)
        layers += [
            nn.Conv2d(ndf * mult_next, 1, 4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # shape: (B, 1, H/8, W/8)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       5. 损失函数 与 图像缓冲区                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class GANLoss(nn.Module):
    """
    LSGAN 损失 (MSE 而非 BCE):
      - BCE 的 sigmoid 在判别器过强时梯度饱和, 生成器学不动
      - MSE 对所有判别分数都有稳定梯度, 缓解模式崩溃
    参考: Mao et al., "Least Squares Generative Adversarial Networks"
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.loss = nn.MSELoss()

    def __call__(self, preds, target_is_real):
        if target_is_real:
            target = self.real_label.expand_as(preds)
        else:
            target = self.fake_label.expand_as(preds)
        return self.loss(preds, target)


class ImageBuffer:
    """
    历史图像缓冲区 (Shrivastava et al., 2017):
      维护一个大小为 pool_size 的队列, 每步以 50% 概率用新生成的假图
      替换队列中的一张旧图, 然后将这张旧图 (而非最新生成的图) 返回给判别器训练。

    为什么这样做:
      判别器如果总是看到最新生成的假图, 会快速适应生成器的变化, 导致 G ↔ D 之间
      剧烈振荡。随机喂一些"过时"的假图, 可以让判别器的更新更平滑, 有效抑制
      模式崩溃 (mode collapse)。
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        """
        Args:
            image: (1, C, H, W) — batch_size=1 的生成图像
        Returns:
            可能被替换为历史图像的张量
        """
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image.clone())
            return image
        else:
            p = random.random()
            if p > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                old = self.images[idx]
                self.images[idx] = image.clone()
                return old
            else:
                return image


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       6. 学习率调度器                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class LinearDecayLR:
    """
    前 n_epochs_decay 个 epoch 保持恒定 lr, 之后线性衰减到 0。
    CycleGAN 论文发现: 恒定 lr + 最后线性衰减 → 训练更稳定。
    """

    def __init__(self, optimizer, n_epochs, n_epochs_decay):
        self.optimizer = optimizer
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.n_epochs_decay = n_epochs_decay
        self.total = n_epochs - n_epochs_decay

    def step(self, epoch):
        if epoch >= self.n_epochs_decay:
            ratio = 1.0 - (epoch - self.n_epochs_decay) / self.total
            lr = self.initial_lr * ratio
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       7. 模型初始化 (权重初始化)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def init_weights(m):
    """用正态分布 N(0, 0.02) 初始化卷积/转置卷积权重, 比默认的 Kaiming init
    在 CycleGAN 中效果更好 (实测收敛更快)。"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       8. 训练循环                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train():
    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- 自动下载数据集 ---
    path_A, path_B = download_cyclegan_dataset(cfg.dataset, cfg.download_root)

    # --- 数据集 ---
    dataset_A = ImageDataset(path_A, cfg.image_size, mode="train")
    dataset_B = ImageDataset(path_B, cfg.image_size, mode="train")
    loader_A = DataLoader(dataset_A, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, drop_last=True)
    loader_B = DataLoader(dataset_B, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, drop_last=True)
    print(f"[Data] A: {len(dataset_A)} images, B: {len(dataset_B)} images")

    # --- 模型 ---
    # G_A2B: A → B (如: 照片 → 风格画)
    # G_B2A: B → A (如: 风格画 → 照片)
    G_A2B = Generator(cfg.input_nc, cfg.output_nc, cfg.ngf, cfg.n_res).to(device)
    G_B2A = Generator(cfg.output_nc, cfg.input_nc, cfg.ngf, cfg.n_res).to(device)
    D_A = Discriminator(cfg.input_nc, cfg.ndf, cfg.n_layers_D).to(device)   # 判别域 A 的真假
    D_B = Discriminator(cfg.output_nc, cfg.ndf, cfg.n_layers_D).to(device)  # 判别域 B 的真假

    # 权重初始化
    G_A2B.apply(init_weights)
    G_B2A.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    # --- 损失函数 ---
    gan_loss = GANLoss().to(device)
    cycle_loss = nn.L1Loss()       # 循环一致性损失: L1 比 L2 更少产生模糊
    identity_loss = nn.L1Loss()    # 身份损失: 也使用 L1

    # --- 优化器 ---
    # 判别器和生成器各用一个优化器 (生成器优化器管理 G_A2B + G_B2A 的参数)
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=cfg.lr, betas=(cfg.beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=cfg.lr, betas=(cfg.beta1, 0.999)
    )

    # 学习率调度 (一个调度器管理一个优化器)
    scheduler_G = LinearDecayLR(optimizer_G, cfg.n_epochs, cfg.n_epochs_decay)
    scheduler_D = LinearDecayLR(optimizer_D, cfg.n_epochs, cfg.n_epochs_decay)

    # 图像缓冲区 (用于训练判别器时平滑梯度)
    buffer_fake_A = ImageBuffer(cfg.pool_size)
    buffer_fake_B = ImageBuffer(cfg.pool_size)

    # --- 日志 ---
    os.makedirs(cfg.save_root, exist_ok=True)
    losses = {"G": [], "D": [], "cycle": [], "identity": []}

    # ═══════════════════ 训练循环 ═══════════════════
    for epoch in range(1, cfg.n_epochs + 1):
        epoch_loss_G = 0.0; epoch_loss_D = 0.0
        epoch_loss_cycle = 0.0; epoch_loss_idt = 0.0

        # zip 两个 dataloader: 每个 batch 得到 (一张来自 A, 一张来自 B)
        # 如果两个数据集大小不同, 使用 zip(*[iter(A)]*N) 会以短的为准
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A = real_A.to(device)  # 域 A 真实图
            real_B = real_B.to(device)  # 域 B 真实图

            # ============================================================
            # 8.1 训练生成器 G_A2B 和 G_B2A
            # ============================================================
            # 目标: 最小化 GAN loss (让生成的假图骗过判别器)
            #          + cycle loss (翻译一圈后能回来)
            #          + identity loss (域内图片保持不变, 缓解颜色偏移)

            optimizer_G.zero_grad()

            # ---- 身份损失 (Identity Loss) ----
            # 改动说明: 当输入已经在目标域 (如: 用 G_B2A 处理一张域 A 的图),
            # 我们期望生成器"不做改变"直接输出原图。这能有效防止生成器
            # 在翻译时意外改变全局色调/色温 (颜色偏移问题)。
            #
            # lambda_identity=0.5 较小, 仅作为正则项, 不会主导训练。
            # 实测经验: 不加 identity loss 时, 风景→梵高风格可能整体偏黄;
            # 加上后色调更自然。
            idt_A = G_A2B(real_A)  # 照片→风格化: 输入已经是 A, 输出也应该接近 A (身份映射)
            loss_idt_A = identity_loss(idt_A, real_A) * cfg.lambda_identity
            idt_B = G_B2A(real_B)
            loss_idt_B = identity_loss(idt_B, real_B) * cfg.lambda_identity

            # ---- 对抗损失 (Adversarial Loss): 生成器视角 ----
            # G 希望 D(fake) 预测为"真"
            fake_B = G_A2B(real_A)     # A → 假 B
            pred_fake_B = D_B(fake_B)  # 判别器 D_B 对假图的判断
            loss_G_A2B = gan_loss(pred_fake_B, target_is_real=True)

            fake_A = G_B2A(real_B)     # B → 假 A
            pred_fake_A = D_A(fake_A)
            loss_G_B2A = gan_loss(pred_fake_A, target_is_real=True)

            # ---- 循环一致性损失 (Cycle-Consistency Loss) ----
            # 核心创新: x → G(x) → F(G(x)) 应该 ≈ x
            # 这保证了翻译不会任意改变内容, 两个生成器互相约束
            rec_A = G_B2A(fake_B)  # A → 假B → 重建A
            loss_cycle_A = cycle_loss(rec_A, real_A) * cfg.lambda_cycle

            rec_B = G_A2B(fake_A)  # B → 假A → 重建B
            loss_cycle_B = cycle_loss(rec_B, real_B) * cfg.lambda_cycle

            # 生成器总损失
            loss_G = (loss_G_A2B + loss_G_B2A +
                      loss_cycle_A + loss_cycle_B +
                      loss_idt_A + loss_idt_B)
            loss_G.backward()
            optimizer_G.step()

            # ============================================================
            # 8.2 训练判别器 D_A 和 D_B
            # ============================================================
            # 目标: D(真实图) → 1, D(生成假图) → 0

            optimizer_D.zero_grad()

            # --- D_A: 判别域 A ---
            # 真实图片 → 真
            loss_D_A_real = gan_loss(D_A(real_A), target_is_real=True)
            # 假图片 → 假 (使用图像缓冲区平滑)
            loss_D_A_fake = gan_loss(D_A(buffer_fake_A.query(fake_A.detach())),
                                     target_is_real=False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # --- D_B: 判别域 B ---
            loss_D_B_real = gan_loss(D_B(real_B), target_is_real=True)
            loss_D_B_fake = gan_loss(D_B(buffer_fake_B.query(fake_B.detach())),
                                     target_is_real=False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optimizer_D.step()

            # --- 记录损失 ---
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            epoch_loss_cycle += (loss_cycle_A + loss_cycle_B).item()
            epoch_loss_idt += (loss_idt_A + loss_idt_B).item()

        # --- 学习率更新 (每个 epoch 结束后) ---
        scheduler_G.step(epoch)
        scheduler_D.step(epoch)

        # --- 日志 ---
        n_batches = min(len(loader_A), len(loader_B))
        print(f"Epoch [{epoch:3d}/{cfg.n_epochs}] "
              f"G: {epoch_loss_G/n_batches:.4f}  "
              f"D: {epoch_loss_D/n_batches:.4f}  "
              f"Cycle: {epoch_loss_cycle/n_batches:.4f}  "
              f"Idt: {epoch_loss_idt/n_batches:.4f}  "
              f"lr: {optimizer_G.param_groups[0]['lr']:.6f}")

        # --- 保存生成样本 ---
        if epoch % cfg.sample_interval == 0:
            with torch.no_grad():
                G_A2B.eval(); G_B2A.eval()
                fake_B_sample = G_A2B(real_A)
                fake_A_sample = G_B2A(real_B)
                rec_A_sample = G_B2A(fake_B_sample)
                rec_B_sample = G_A2B(fake_A_sample)
                G_A2B.train(); G_B2A.train()

            images = torch.cat([real_A, fake_B_sample, rec_A_sample,
                                real_B, fake_A_sample, rec_B_sample], dim=0)
            images = (images + 1) / 2  # [-1,1] → [0,1]
            save_image(images, os.path.join(cfg.save_root, f"epoch_{epoch:03d}.png"),
                       nrow=cfg.batch_size)

        # --- 保存模型 ---
        if epoch % 50 == 0:
            torch.save(G_A2B.state_dict(), os.path.join(cfg.save_root,
                        f"G_A2B_epoch{epoch}.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(cfg.save_root,
                        f"G_B2A_epoch{epoch}.pth"))

    # --- 保存最终模型 ---
    torch.save(G_A2B.state_dict(), os.path.join(cfg.save_root, "G_A2B_final.pth"))
    torch.save(G_B2A.state_dict(), os.path.join(cfg.save_root, "G_B2A_final.pth"))
    print("[Done] Training finished.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       9. 推理: 单张图片风格迁移                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def infer(image_path, model_path, output_path="output.png", direction="A2B"):
    """加载训练好的生成器，对单张图片做风格迁移"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(cfg.input_nc, cfg.output_nc, cfg.ngf, cfg.n_res).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = G(tensor)

    output = (output.squeeze(0) + 1) / 2  # [-1,1] → [0,1]
    save_image(output, output_path)
    print(f"[Infer] Saved to {output_path} (direction: {direction})")


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CycleGAN: 非配对图像风格迁移 — 课程作业框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认数据集训练 (apple2orange)
  python cyclegan.py --mode train

  # 使用其他数据集训练
  python cyclegan.py --mode train --dataset horse2zebra

  # 推理: 苹果 → 橙子 (A2B)
  python cyclegan.py --mode infer --image my_apple.jpg \\
      --model checkpoints/cyclegan_apple2orange/G_A2B_final.pth

  # 推理: 橙子 → 苹果 (B2A)
  python cyclegan.py --mode infer --image my_orange.jpg --direction B2A \\
      --model checkpoints/cyclegan_apple2orange/G_B2A_final.pth
        """
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"],
                        help="train: 训练模型, infer: 单张图片推理")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集名称 (如 apple2orange, horse2zebra), 不指定则用 Config 默认值")
    parser.add_argument("--image", type=str, default=None, help="推理时的输入图片路径")
    parser.add_argument("--model", type=str, default=None, help="推理时的模型权重路径")
    parser.add_argument("--output", type=str, default="output.png", help="推理输出路径")
    parser.add_argument("--direction", type=str, default="A2B", choices=["A2B", "B2A"],
                        help="翻译方向: A2B=域A→域B, B2A=域B→域A")
    args = parser.parse_args()

    # 命令行可覆盖 dataset 和 save_root
    if args.dataset:
        cfg.dataset = args.dataset
    cfg.save_root = f"checkpoints/cyclegan_{cfg.dataset}"

    if args.mode == "train":
        train()
    else:
        if not args.image or not args.model:
            print("推理模式需要 --image 和 --model 参数")
        else:
            infer(args.image, args.model, args.output, args.direction)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                 附录: 训练中常见问题及对应修复方案                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# 问题 1: 模式崩溃 (Mode Collapse)
#   现象: 生成器对不同输入总是输出相同/相似结果, 丧失多样性
#   原因: 判别器被生成器"找到漏洞", G 只学会几个能骗过 D 的模式
#   修复:
#     1. LSGAN loss (MSE) — 已使用, 比 BCE 梯度更均匀
#     2. Image Buffer (历史重放) — 已使用, 防止 D 过拟合最新 G 输出
#     3. 若仍发生: 降低 D 学习率 (lr_D = lr_G * 0.5) 或减少 D 每轮更新次数
#
# 问题 2: 棋盘伪影 (Checkerboard Artifacts)
#   现象: 生成图像上出现规律性网格/方块纹理
#   原因: 转置卷积 (ConvTranspose2d) 核大小与步长不匹配, 输出像素"权重不均"
#   修复:
#     1. resize-conv (最近邻上采样 + 普通卷积) 替代转置卷积 — 已使用
#     2. 若用转置卷积, 确保 kernel_size 能被 stride 整除 (如 k=4, s=2)
#
# 问题 3: 颜色偏移 (Color Shift)
#   现象: 生成的风格化图片整体偏某种色调 (偏黄、偏蓝等)
#   原因: 生成器在风格转换时同时修改了内容颜色, 无"保留原色调"约束
#   修复:
#     1. Identity Loss — 已使用 (lambda_identity=0.5)
#     2. InstanceNorm 而非 BatchNorm — 已使用, BN 混合 batch 统计量导致颜色泄漏
#     3. 若仍偏色严重: 增加 lambda_identity 到 1.0~5.0
#
# 问题 4: 训练不稳定 / Loss 剧烈振荡
#   现象: 判别器 loss → 0, 生成器 loss 居高不下
#   原因: 判别器太强, 生成器来不及学习
#   修复:
#     1. 减小 D 学习率: lr_D = lr_G * 0.5
#     2. 每轮多次更新 G, 只更新一次 D (n_critic 变体)
#     3. 增大 ImageBuffer pool_size (如 100)
#     4. 给判别器卷积层加 spectral normalization
#
# ═══════════════════════════════════════════════════════════════════════════
# 4060 Laptop 训练时间预估 (image_size=128, batch_size=1, 50 epochs)
# ═══════════════════════════════════════════════════════════════════════════
#   apple2orange      ~1,000 张/域  →  ~1.0 ~ 1.5 小时
#   horse2zebra       ~1,200 张/域  →  ~1.5 ~ 2.0 小时
#   summer2winter     ~1,100 张/域  →  ~1.2 ~ 1.8 小时
#   monet2photo       ~1,000+6,287  →  不推荐 (域B数据倾斜严重, >6h)
