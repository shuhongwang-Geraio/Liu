# PatchTST 详细技术解读（补充版）

## 说明
`patchtst.pdf` 与 `ICLR-2023-PatchTST(1).pdf` 为同一篇论文的不同版本。本文件作为补充，深入分析其技术细节和设计决策。

## 技术细节深入

### Patching的数学形式化
- 输入：单变量序列 x(i) = (x₁(i), ..., x_L(i)) ∈ R^L
- Patch参数：长度P，步长S
- 输出：patch序列 x_p(i) ∈ R^(P×N)，N = ⌊(L-P)/S⌋ + 2
- 边界处理：对序列末尾进行padding（重复最后一个值）

### 注意力计算
- Query: Q_h(i) = (x_d(i))^T · W_Q_h
- Key: K_h(i) = (x_d(i))^T · W_K_h  
- Value: V_h(i) = (x_d(i))^T · W_V_h
- 注意力输出：Attention(Q,K,V) = Softmax(QK^T / √d_k) · V

### 计算复杂度分析
| 方法 | 注意力复杂度 | Token数 |
|------|-------------|---------|
| 点级别输入 | O(L²) | L |
| PatchTST | O(N²) ≈ O((L/S)²) | L/S |
| 复杂度降低 | S²倍 | S倍 |

### 典型参数设置
- Patch长度 P = 16
- 步长 S = 8（50% overlap）
- Transformer层数 = 3
- 注意力头数 H = 4-16
- 隐藏维度 D = 128-512

## Channel Independence vs Channel Mixing
### Channel Independence的理论依据
1. **避免过拟合**：多变量间的相关性可能是虚假的或不稳定的
2. **增大有效训练样本**：M个变量各自独立使用=M倍训练数据
3. **对分布漂移鲁棒**：单变量的统计特性更稳定
4. **与RevIN兼容**：逐通道的归一化更自然

### 何时Channel Mixing更好？
- 变量间有强因果关系
- 数据量充足且分布稳定
- 变量数量较少

## 自监督预训练的深入分析

### Masked Patch预训练
- 随机选择约40%的patch进行mask
- 使用MSE损失重构被mask的patch
- 不使用[CLS] token，直接在patch级别操作

### 迁移学习实验
- 在大数据集（如Electricity）上预训练
- 迁移到小数据集（如Weather）上微调
- 跨域迁移也有效，说明学到了通用的时序模式

## 与后续工作的关系
- **iTransformer**：反转了PatchTST的设计，对变量维度做attention
- **ModernTCN**：使用卷积替代attention但保留patching
- **SOFTS**：使用MLP但也采用通道独立策略
- **TimesFM/TimeGPT**：大规模预训练模型也使用patching
