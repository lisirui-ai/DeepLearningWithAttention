<div align="center">

# DeepLearningWithAttention

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-Spa--Eng-FF6F00?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-WMT16%20De--En-FF6F00?style=flat-square&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 PyTorch 的注意力机制机器翻译实战系列</p>
<p>以 <b>Seq2Seq + Bahdanau Attention → Transformer + Multi-Head Attention</b> 为主线，循序渐进演示从经典注意力到自注意力机制的完整翻译流程</p>
<p>每个 Notebook 均配有详细中文注释，适合自然语言处理与注意力机制入门与进阶学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [数据集](#-数据集)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览


| #   | Notebook                                        | 核心技术                                                              | 亮点                        |
| --- | ----------------------------------------------- | ----------------------------------------------------------------- | ------------------------- |
| 1   | Seq2Seq 翻译 + Bahdanau Attention                 | Seq2Seq · GRU · Bahdanau Attention · 子词级分词                        | 经典注意力机制 · 西班牙语→英语翻译       |
| 2   | Transformer 翻译 + Multi-Head Scaled Dot-Product | Transformer · 多头缩放点积注意力 · Token 级批次 · SentencePiece BPE · WMT16 · 束搜索解码 | 完整 Transformer 架构 · 束搜索解码 · 德语→英语翻译 |


---

## 📁 目录结构

```
DeepLearningWithAttention/
├── 📓 1.seq2seq_translation_SubwordLevelTokenization_BahdanauAttention.ipynb
├── 📓 2.transformer_translation_..._MultiHeadScaledDotProductAttention.ipynb
│         (全名见下方 Notebook 介绍)
│
├── 📂 data/
│   ├── spa.txt                    # [已提供] 西班牙语-英语平行语料（Notebook 1）
│   ├── cache/                     # [自动生成] Notebook 1 预处理缓存（gitignored）
│   │   └── lang_pair.npy
│   └── wmt16/                     # WMT16 德语-英语平行语料（Notebook 2）
│       ├── train.de / train.en    # [已提供] 训练集原始文本
│       ├── val.de   / val.en      # [已提供] 验证集原始文本
│       ├── test.de  / test.en     # [已提供] 测试集原始文本
│       ├── spm_joint_bpe.model    # [自动生成] SentencePiece BPE 联合分词模型（gitignored）
│       ├── spm_joint_bpe.vocab    # [自动生成] BPE 词表（gitignored）
│       ├── train_src.bpe / train_trg.bpe   # [自动生成] 训练集 BPE 编码结果（gitignored）
│       ├── val_src.bpe   / val_trg.bpe     # [自动生成] 验证集 BPE 编码结果（gitignored）
│       ├── test_src.bpe  / test_trg.bpe    # [自动生成] 测试集 BPE 编码结果（gitignored）
│       ├── *.cut.txt              # [自动生成] Moses 分词后的文本（gitignored）
│       └── cache/                 # [自动生成] Notebook 2 预处理 NumPy 缓存（gitignored）
│           ├── de2en_train_512.npy
│           ├── de2en_val_512.npy
│           └── de2en_test_512.npy
│
├── 📂 model_checkpoints/          # 各实验最优模型权重（.ckpt）
│   ├── 1_model/
│   │   └── 1_model_best.ckpt
│   └── 2_model/
│       └── 2_model_best.ckpt
│
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

---

## 📚 Notebook 介绍

### 1.Seq2Seq 翻译 + 子词分词 + Bahdanau Attention

> `1.seq2seq_translation_SubwordLevelTokenization_BahdanauAttention.ipynb`

使用 PyTorch 搭建 **Seq2Seq（Encoder-Decoder）** 架构，结合 **GRU** 循环单元与 **Bahdanau 加性注意力机制**，在西班牙语→英语翻译任务上实现可对齐的序列生成。引入子词级分词（Subword-Level Tokenization）替代词级分词，有效缓解 OOV（未登录词）问题。


| 章节    | 内容                                                             |
| ----- | -------------------------------------------------------------- |
| 环境导入  | 依赖库导入 · 设备检测（CPU / GPU）                                        |
| 数据集准备 | `spa.txt` 加载 · 子词级分词 · 词表构建 · `lang_pair.npy` 缓存 · DataLoader  |
| 构建模型  | GRU Encoder · Bahdanau Attention · GRU Decoder · Seq2Seq 整体封装 |
| 模型训练  | 损失函数 · 优化器 · Teacher Forcing · 训练循环 · 最优模型保存                   |
| 模型评估  | BLEU 分数计算 · 注意力权重可视化 · 翻译样例展示                                  |


> **Bahdanau Attention 原理** · 在每个解码步骤，对解码器隐状态（query `sₜ₋₁`）与每个编码器隐状态（key `hᵢ`）分别经线性层投影后相加，经 tanh 非线性激活，再由可学习向量 V 压缩为对齐标量：`score(sₜ₋₁, hᵢ) = V · tanh(Wq·sₜ₋₁ + Wk·hᵢ)`，经 Softmax 归一化得到注意力权重，对编码器输出加权求和生成上下文向量，使解码器动态聚焦于源句的不同位置。
>
> 参考论文：*Neural Machine Translation by Jointly Learning to Align and Translate*（Bahdanau et al., 2015）

---

### 2.Transformer 翻译 + 子词分词 + Token 级批次 + 多头缩放点积注意力

> `2.transformer_translation_SubwordLevelTokenization_TokenLevelBatching_MultiHeadScaledDotProductAttention.ipynb`

从零手动复现完整 **Transformer** 架构，涵盖多头缩放点积自注意力（Multi-Head Scaled Dot-Product Attention）、**GEGLU 激活前馈网络**、**Pre-LayerNorm**、掩码机制、编码器/解码器堆叠，在 WMT16 德语→英语翻译任务上完成训练。引入 **Token 级动态批次（Token-Level Batching）** 策略，按 Token 总数而非句子数组批，提升 GPU 利用率。使用 **SentencePiece BPE** 进行联合子词分词。评估阶段采用 **束搜索解码（Beam Search Decoding）** 并分别统计 BLEU-1 与 BLEU-4 指标；最终通过 **端到端翻译器（Translator）** 将分词、推理、去分词封装为单一调用接口。


| 章节       | 内容                                                                                     |
| -------- | -------------------------------------------------------------------------------------- |
| 环境导入     | 依赖库导入 · 设备检测                                                                            |
| 数据集准备    | WMT16 加载 · SentencePiece BPE 分词 · Token 级批次采样 · NumPy 缓存 · DataLoader                 |
| 构建模型     | 词嵌入层（含位置编码） · 缩放点积多头注意力 · GEGLU 前馈网络 · Pre-LayerNorm · 掩码机制 · 编码器 · 解码器 · 完整 Transformer |
| 模型训练     | 标签平滑损失 · Noam Warmup 学习率调度 · AdamW 优化器 · TensorBoard 回调 · 训练与验证主循环 · 最优模型保存           |
| 模型评估     | 束搜索解码（beam_size 可调） · BLEU-1 · BLEU-4（论文标准指标） · 翻译样例展示                               |
| 端到端翻译    | Translator 封装 · 分词 → 束搜索推理 → 去分词完整管线 · 单句翻译演示                                        |


> **Scaled Dot-Product Attention 原理** · 将查询矩阵 Q 与键矩阵 K 的点积除以 `√d_k` 进行缩放，避免维度过高时点积值过大导致 Softmax 梯度消失，再对值矩阵 V 加权求和：`Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`。**多头注意力**将 Q/K/V 投影到 h 个子空间并行计算注意力，拼接后再线性变换，使模型同时关注不同位置的多种语义关系。
>
> **Beam Search 解码原理** · 推理阶段同时维护 `beam_size` 条候选序列，每步对每条路径展开整个词表并取累计对数概率最高的 Top-K 路径延续搜索；生成 EOS 的序列转入完成队列并施加长度惩罚，最终从完成序列中选取归一化得分最高者作为输出。相比贪心解码（beam=1）在标准测试集上通常可提升 **+2~4 BLEU**，且无需重新训练模型。
>
> 参考论文：*Attention Is All You Need*（Vaswani et al., 2017）

---

## 🗃 数据集

**Spa-Eng** · 西班牙语-英语平行语料

| 属性   | 详情                                  |
| ---- | ----------------------------------- |
| 语言对  | 西班牙语 → 英语                           |
| 数据规模 | 约 118,964 句对                        |
| 格式   | Tab 分隔纯文本（`spa.txt`）                |
| 分词方式 | 子词级（Subword-Level）分词                 |
| 缓存   | 首次运行后自动生成 `data/cache/lang_pair.npy`（gitignored） |

已随仓库直接提供，克隆后即可使用，无需额外下载。

---

**WMT16 De-En** · 德语-英语机器翻译标准评测数据集

| 属性   | 详情                                          |
| ---- | ------------------------------------------- |
| 语言对  | 德语 → 英语                                     |
| 训练集  | 约 4,500,000 句对                              |
| 验证集  | newstest2013（约 3,000 句对）                    |
| 测试集  | newstest2014（约 3,003 句对）                    |
| 分词方式 | SentencePiece BPE 联合词表                      |
| 最大长度 | 截断至 512 Token                               |
| 缓存   | 首次运行后自动生成至 `data/wmt16/cache/*.npy`（gitignored） |

原始语料（`*.de` / `*.en`）已随仓库直接提供，克隆后即可使用。`spm_joint_bpe.model`、`*.bpe`、`*.cut.txt` 及 NumPy 缓存均由 Notebook 运行时自动生成，无需手动准备。

---

## 🛤 学习路径

```
Notebook 1                          Notebook 2
Spa-Eng 翻译                →        WMT16 De-En 翻译
Seq2Seq + GRU                        Transformer
Bahdanau Attention                   Multi-Head Scaled Dot-Product Attention
（加性注意力 · 经典 RNN 架构）             Token-Level Batching · SentencePiece BPE
                                     Beam Search Decoding
                                     （自注意力 · 纯注意力架构）
```

建议按编号顺序学习，先掌握经典注意力机制（Notebook 1），再深入理解完整 Transformer 架构（Notebook 2）。

---

## ⚙️ 环境依赖


| 包名              | 版本           | 用途                                       |
| --------------- | ------------ | ---------------------------------------- |
| `torch`         | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2）                      |
| `tensorboard`   | 2.20.0       | 训练过程实时可视化（两个 Notebook 均使用）               |
| `matplotlib`    | 3.10.9       | 注意力权重热力图 · 训练曲线绘制                        |
| `numpy`         | 2.4.6        | 数值计算 · 数据缓存（`.npy`）                      |
| `pandas`        | 3.0.3        | 数据读取与管理（Notebook 1）                      |
| `tqdm`          | 4.68.2       | 训练进度条显示                                  |
| `nltk`          | —            | BLEU 分数计算（两个 Notebook 均使用）               |
| `subword_nmt`   | —            | BPE 子词分词器训练与应用（Notebook 1）               |
| `sentencepiece` | —            | SentencePiece BPE 子词分词（Notebook 2）       |
| `sacremoses`    | —            | Moses 分词 / 去分词（Notebook 2）               |


---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/DeepLearningWithAttention.git
cd DeepLearningWithAttention
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` 与 `torchvision` 带有 `+cu132`（CUDA 13.2）后缀，如需 CPU 版本或其他 CUDA 版本，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 替换。

**3. 准备数据集**

- **Spa-Eng**：`data/spa.txt` 已随仓库提供，无需额外下载。
- **WMT16 De-En**：`data/wmt16/*.de` 与 `*.en` 原始语料已随仓库提供。`spm_joint_bpe.model`、`*.bpe`、`*.cut.txt` 及 NumPy 缓存在首次运行 Notebook 时自动生成。

**4. 启动 Jupyter**

```bash
jupyter notebook
```

按编号顺序打开 Notebook 开始学习即可。

---

## 📄 License

[MIT](LICENSE) © 2026
