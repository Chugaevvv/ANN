# CycleGAN 图像风格迁移 — 课程作业

基于 PyTorch 实现的 CycleGAN（非配对图像风格迁移），支持一键下载数据集并训练。

**默认任务**：苹果 ↔ 橙子外观互转（保持形状，迁移颜色/纹理）

---

## 环境要求

- Python 3.8+
- PyTorch 1.10+（推荐 2.0+）
- CUDA 可选（强烈建议用 GPU 训练）

```bash
pip install torch torchvision pillow numpy
```

---

## 快速开始

### 1. 训练（自动下载数据集）

```bash
# 默认使用 apple2orange 数据集
python cyclegan.py --mode train
```

首次运行会自动将数据集下载到 `data/apple2orange/`（约 100MB），之后不再重复下载。

### 2. 换用其他数据集

```bash
python cyclegan.py --mode train --dataset horse2zebra
python cyclegan.py --mode train --dataset summer2winter_yosemite
```

所有可用数据集（无需配对标注）：

| 数据集 | 域 A | 域 B | 图片数(A/B) |
|---|---|---|---|
| `apple2orange` | 苹果 | 橙子 | 995 / 1019 |
| `horse2zebra` | 马 | 斑马 | 1067 / 1334 |
| `summer2winter_yosemite` | 夏景 | 冬景 | 1231 / 962 |
| `monet2photo` | 莫奈画 | 照片 | 1072 / 6287 |

### 3. 推理（单张图片风格迁移）

```bash
# 苹果 → 橙子方向
python cyclegan.py --mode infer \
    --image my_apple.jpg \
    --model checkpoints/cyclegan_apple2orange/G_A2B_final.pth

# 橙子 → 苹果方向（反向）
python cyclegan.py --mode infer \
    --image my_orange.jpg \
    --direction B2A \
    --model checkpoints/cyclegan_apple2orange/G_B2A_final.pth
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `train` | `train` 训练 / `infer` 推理 |
| `--dataset` | `apple2orange` | 数据集名称 |
| `--image` | - | 推理时的输入图片路径 |
| `--model` | - | 推理时的模型权重 `.pth` 路径 |
| `--output` | `output.png` | 推理输出路径 |
| `--direction` | `A2B` | 翻译方向：`A2B`（域A→域B）或 `B2A`（域B→域A） |

更多训练超参（图像尺寸、训练轮数、学习率等）见 `cyclegan.py` 顶部 `Config` 类。

---

## 训练时间预估（RTX 4060 Laptop 8GB）

| 数据集 | 图片数/域 | 预计耗时 |
|---|---|---|
| apple2orange | ~1,000 | **1.0 ~ 1.5 小时** |
| horse2zebra | ~1,200 | 1.5 ~ 2.0 小时 |
| summer2winter | ~1,100 | 1.2 ~ 1.8 小时 |

> 当前配置为 `image_size=128`, `n_epochs=50`。如需更高质量可调大到 256/100，时间会相应增加。

---

## 输出文件结构

```
checkpoints/cyclegan_apple2orange/
├── epoch_005.png          # 每 5 个 epoch 生成对比图
├── epoch_010.png          #   格式: 原A | A→假B | 重建A
├── ...                    #         原B | B→假A | 重建B
├── epoch_050.png
├── G_A2B_epoch50.pth      # 每 50 epoch 保存一次
├── G_B2A_epoch50.pth
├── G_A2B_final.pth        # 最终模型权重
└── G_B2A_final.pth
```

### 如何解读中间结果图

```
[ real_A  ][ fake_B  ][ rec_A  ]    ← 一行三列
[ real_B  ][ fake_A  ][ rec_B  ]
```

- `real_A` — 域 A 原始图片
- `fake_B` — 生成器 G_A2B 翻译到域 B
- `rec_A` — 再翻译回域 A（应与 real_A 越接近越好）
- `real_B` / `fake_A` / `rec_B` 同理（反向）

---

## 常见问题

### 生成结果模糊 / 颜色异常
- 增大 `n_epochs`（如 100）
- 增大 `image_size`（如 256），需同步调整 `n_res=9`
- 检查 `lambda_identity`（增大到 1.0~5.0 可缓解颜色偏移）

### 显存不足 (Out of Memory)
- 降低 `image_size`（如 64）
- 关闭其他占用显存的程序

### 下载数据集太慢
- 可从 [CycleGAN 官方仓库](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) 手动下载 zip 解压到 `data/` 目录下

### 论文参考
- Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
