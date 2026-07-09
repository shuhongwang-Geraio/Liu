# PatchTST: A Time Series is Worth 64 Words — Long-term Forecasting with Transformers

## 论文基本信息
- **会议**: ICLR 2023
- **作者**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
- **机构**: Princeton University, IBM Research

## 研究问题
如何设计高效的Transformer模型以应对DLinear等简单模型的挑战，使Transformer能在长期时间序列预测和自监督表征学习中真正发挥优势。

## 核心思想
类比NLP中的"一段文字由词组成"：**一个时间序列由多个子序列Patch组成**。通过将时间序列分割为patch作为Transformer的输入token，结合通道独立策略，大幅提升效率和性能。

## 两大关键设计

### 1. Patching（分片）
- **操作**：将长度为L的时间序列切割为多个长度为P、步长为S的子序列（patch）
- **patch数量**：N = ⌊(L-P)/S⌋ + 2
- **三重好处**：
  - **保留局部语义**：每个patch包含一段连续的子序列，保留了局部时间模式
  - **降低计算复杂度**：token数从L降为约L/S，注意力图的计算量二次减少
  - **扩展历史视野**：在相同计算资源下可以看到更长的历史序列

### 2. Channel Independence（通道独立）
- **设计**：每个变量（通道）独立输入共享参数的Transformer
- **优势**：
  - 避免多变量间虚假相关性的干扰
  - 参数共享提升泛化能力
  - 每个通道的单变量序列作为独立样本，有效增大训练集规模
  - 对分布漂移更鲁棒

## 模型架构

```
输入序列 x(i) ∈ R^L
    ↓
Patching → x_p(i) ∈ R^(P×N)
    ↓
线性投影 W_p ∈ R^(D×P) → 潜在空间
    ↓
+ 可学习位置编码 W_pos ∈ R^(D×N)
    ↓
Vanilla Transformer Encoder (Multi-head Attention + BatchNorm + FFN)
    ↓
Flatten + Linear Head → 预测 x̂(i) ∈ R^T
```

## 自监督预训练

### Masked Patch预训练
- 随机遮蔽部分patch，训练模型重构被遮蔽的patch
- 类比BERT的Masked Language Model
- 预训练后fine-tuning可以**超越纯监督训练**
- **跨数据集迁移**：在一个数据集上预训练的表征可以迁移到其他数据集

## 实验结果

### 长期预测
- 在多个标准数据集上显著优于之前的SOTA Transformer方法
- 也优于DLinear等简单线性模型
- 使用更长look-back窗口时优势更明显

### 自监督学习
- 预训练+微调 > 纯监督训练（尤其在大数据集上）
- 证明了时间序列中自监督表征学习的有效性

## 核心贡献
1. **Patching设计**：高效、有效地将时间序列输入Transformer
2. **Channel Independence**：简洁但强大的多变量处理策略
3. **回应DLinear挑战**：证明正确设计的Transformer仍有优势
4. **自监督框架**：首次成功将masked预训练应用于时间序列Transformer

## 设计洞察
- Patch粒度的选择至关重要：P=16，S=8是常用设置
- BatchNorm优于LayerNorm（时间序列的特性）
- Vanilla Transformer encoder已足够，无需复杂变体
- 关键不在于注意力机制的改进，而在于**输入表示方式**的改进

## 与其他工作的联系
- **回应DLinear**：证明Transformer在正确设计下仍然有效
- **启发后续工作**：ModernTCN、SOFTS等都采用了类似的patching思想
- **与NLP/CV的统一**：patching类比ViT中的image patch和NLP中的word token
