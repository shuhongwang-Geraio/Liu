# data_ood —— OOD 数据切分定义

存放 CausalCIT 在 OOD 评估中使用的**训练/测试时段切分**配置（时序漂移协议：
早时段训练、晚时段测试、中间留 gap）。

本目录体积小，整体进 git。若后续切分依赖的原始 csv 较大，请放在
`01_external/PatchTST/code/dataset/`（已 gitignore），本目录只保留切分索引/元数据。
