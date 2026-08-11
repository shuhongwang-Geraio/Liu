# Understanding Transformers for TSF: Moirai (ICLR 2026)

- 标题: Understanding Transformers in Time Series Forecasting: A Case Study on Moirai
- 链接: https://proceedings.iclr.cc/paper_files/paper/2026/hash/986c1ad1c8da47fffd6d64ef594bacea-Abstract-Conference.html
- PDF: paper/UnderstandingMoirai_ICLR2026.pdf
- 一句话: 理论分析：证明 Transformer 可梯度下降拟合任意单变量 AR 模型；Moirai any-variate 编码能自动适配任意协变量数的 AR；Dobrushin 条件下预训练泛化界 ~1/√(nT)。
- 相关性: 理论支撑"任意通道数下 Transformer 通道交互在容量上无障碍"——问题在交互的稳定性而非容量，间接支持我们"需要的不是更多交互而是更稳交互"的动机。
  - 注: Moirai 为跨域预训练基础模型，与单数据集场景不同。
- 详细分析: surveys/03_multiscale_causal_decoupling/paper_analysis_deep.md §5
