# 九章算术语言模型训练

基于小型 Transformer（GPT 风格）在《九章算术》古典中文数学语料上训练字符级语言模型，验证模型是否学习到古代数学知识。

## 目录结构

```
hw4/
├── 九章算经.txt              # 原始语料（GB18030 编码，约72K字符）
├── jiuzhang_final.py         # 主程序：数据预处理 → Word2Vec → MiniGPT → 评估
├── report.tex                # 实验报告（LaTeX）
├── requirements.txt          # Python 依赖
├── prompt.md                 # 原始任务说明
├── checkpoints/
│   ├── best_gpt.pt           # MiniGPT 最佳模型检查点
│   └── best_lstm.pt          # LSTM 基线模型检查点
└── results/
    ├── curves.png            # 训练/验证 Loss 与 PPL 曲线
    ├── w2v_tsne.png          # Word2Vec 字符嵌入 t-SNE 可视化
    ├── similarity_map.png    # 嵌入相似度热力图（注意力代理分析）
    ├── w2v_neighbors.txt     # 关键字符语义近邻分析
    ├── completions.txt       # 题目补全生成样例
    └── summary.json          # 实验结果摘要
```

## 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖（CUDA 12.1）
pip install -r requirements.txt
```

**硬件需求**：NVIDIA GPU（≥8GB VRAM），已在 RTX 4060 Laptop (8.6GB) 上测试通过。

## 运行

```bash
cd hw4
source venv/bin/activate
python jiuzhang_final.py
```

程序自动完成以下步骤：

| 步骤 | 内容 | 耗时 |
|------|------|------|
| 1. 数据加载 | GB18030 解码、章节分割、QA 提取、滑动窗口采样 | < 1s |
| 2. Word2Vec | Skip-gram 训练 (epochs=100)、t-SNE 可视化 | ~30s |
| 3. MiniGPT 训练 | Decoder-only Transformer, 120 epochs | ~5 min |
| 4. LSTM 基线 | 2-layer LSTM, 80 epochs | ~2 min |
| 5. 评估 | 题目补全、跨章泛化准确率、温度采样对比 | ~1 min |

## 数据集划分

| 划分 | 章节 | 字符数 | 题数 |
|------|------|--------|------|
| 训练集 | 1–7（方田～盈不足） | 25,313 | 173 |
| 验证集 | 8（方程） | 3,428 | — |
| 测试集 | 9（句股） | 7,295 | 19 |

## 模型参数

### MiniGPT

| 参数 | 值 |
|------|-----|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 4 |
| d_ff | 512 |
| max_seq_len | 128 |
| dropout | 0.15 |
| 总参数量 | ~0.94M |

### LSTM 基线

| 参数 | 值 |
|------|-----|
| embed_dim | 256 |
| hidden_dim | 256 |
| n_layers | 2 |
| dropout | 0.3 |
| 总参数量 | ~1.32M |

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (weight_decay=0.05) |
| 学习率 | 3×10⁻⁴ |
| LR 调度 | Linear warmup (10%) + Cosine decay |
| Batch size | 32 |
| 精度 | FP16 混合精度 (GradScaler) |
| 损失函数 | Cross-Entropy (ignore PAD) |
| 梯度裁剪 | max_norm=1.0 |

## 实验结果

| 模型 | 测试集 PPL | 答案准确率 |
|------|-----------|-----------|
| 随机基线 | 1049 | ~0% |
| LSTM | 26.1 | 0% |
| **MiniGPT** | **24.1** | 0% |

MiniGPT PPL=24.1 远优于随机基线（1049），证明模型学到了古典数学中文的字符级语言规律。答案完全匹配率为 0% 是极端数据稀缺（~25K 训练字符）下的预期结果——纯语言模型无法仅通过文本共现学会精确数值计算。

## Word2Vec 关键发现

- `句` → `股`(0.84), `弦`(0.48)：成功捕捉勾股定理三元组关系
- `尺` → `寸`(0.64), `丈`(0.48)：度量衡单位自动聚类
- `亩` → `顷`(0.55)：面积单位语义近邻
