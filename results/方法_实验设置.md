# 方法部分可用的事实：bishan_tizhong

生成时间（北京时间）：2026-10-14T19:43:19+08:00。以下内容由定义文件、锁文件与产物自动汇出，只写事实；写法与措辞由作者决定。

## 数据

- **obesity_syndrome**：任务 `multi_label`；开发集 1278 行、外部测试集 308 行；特征 348 个、标签 11 个；数据文件 SHA-256 `736e0f11ac18419ba786d1277dd55980b4184f4e0e1f4a4c6d419f08f5093df1`；缺失哨兵 {"年龄": -1.0, "性别": -1.0, "病程": -1.0}；连续特征 年龄、病程。
  - 开发集：每例标签数均值 3.43、中位数 3，零标签 3 例；各标签阳性数（占比）：心 146（11.4%）、脾 614（48.0%）、肝 279（21.8%）、肾 257（20.1%）、痰 469（36.7%）、湿 794（62.1%）、热 548（42.9%）、气滞 271（21.2%）、血瘀 370（29.0%）、气虚 464（36.3%）、阳虚 169（13.2%）。
  - 外部测试集：每例标签数均值 3.63、中位数 4，零标签 0 例；各标签阳性数（占比）：心 34（11.0%）、脾 170（55.2%）、肝 71（23.1%）、肾 58（18.8%）、痰 119（38.6%）、湿 212（68.8%）、热 135（43.8%）、气滞 75（24.4%）、血瘀 79（25.6%）、气虚 121（39.3%）、阳虚 45（14.6%）。

## 验证方案

- 种子 [0, 1, 2, 3, 4]（5 个；每个种子是一次完整独立的实验：重新划分外层折、重新调参、在同一个外部测试集上评估）。
- 嵌套交叉验证：外层 5 折 × 内层 3 折；每个外层折内层调参 200 次（TPE）。
- 主指标 `macro_f1`；多标签策略 ['br']。
- 阈值：打开（逐标签按 F1 选出的阈值，网格 0.01–0.99；交叉验证嵌套选择、测试集用部署阈值）。
- 外部测试集 bootstrap 10000 次（患者级重抽样、percentile 区间）；决策曲线 开。
- 模型间比较：外部测试集上模型间的差异采用患者级成对 bootstrap 检验（重抽样 10000 次，各模型使用相同的重抽样本，重抽种子 0），同一指标内的两两比较采用 Holm 法校正多重比较，校正后 P<0.05 视为差异有统计学意义；差值的 95% 区间未作多重比较校正。每次重抽样对各种子算指标后取均值，阈值按主口径冻结。
- 交叉验证结果只作描述，不做检验（各折训练集互相重叠，不是独立样本）。
- 外部测试集只在全部学习器 × 种子完成后评估一次；报告不宣布“最终模型”。

## 学习器与数据处理

数据处理声明：以下学习器带折内数据处理步骤（每次拟合只用当次训练部分估计，再原样作用于验证与测试数据）——
  用户指定：knn —— missing_indicator: MissingIndicator(columns=cols.missing) → impute: Impute(columns=cols.missing, strategy='auto') → standardize: Standardize(columns=cols.continuous)
其余学习器（decision_tree、random_forest、extra_trees、dannv2_beta、realmlp、tabpfn_v3、tabicl、tabdpt、deepseek、glm、qwen）直接使用原始特征。

| 学习器 | 模型 | 设备 | 多标签 | 步骤 | 搜索维度 | 调参预算 |
|---|---|---|---|---|---|---|
| knn | knn | CPU | 原生 | missing_indicator: MissingIndicator(columns=cols.missing) → impute: Impute(columns=cols.missing, strategy='auto') → standardize: Standardize(columns=cols.continuous) | 3 | 200 |
| decision_tree | decision_tree | CPU | 原生 | （无） | 5 | 200 |
| random_forest | random_forest | CPU | 原生 | （无） | 7 | 200 |
| extra_trees | extra_trees | CPU | 原生 | （无） | 7 | 200 |
| dannv2_beta | dannv2_beta_ml.DANNV2BetaMLClassifier | GPU | 原生 | （无） | 0 | 0 |
| realmlp | realmlp | GPU | br | （无） | 0 | 0 |
| tabpfn_v3 | tabpfn_v3 | GPU | br | （无） | 0 | 0 |
| tabicl | tabicl | GPU | br | （无） | 0 | 0 |
| tabdpt | tabdpt | GPU | br | （无） | 0 | 0 |
| deepseek | deepseek | CPU | 原生 | （无） | 0 | 0 |
| glm | glm | CPU | 原生 | （无） | 0 | 0 |
| qwen | qwen | CPU | 原生 | （无） | 0 | 0 |

参数的搜索范围与固定值见 `learner_params.csv`。

## 指标

- 报告的指标：macro_f1、micro_f1、weighted_f1、sample_f1、macro_precision、macro_recall、subset_accuracy、hamming_loss、macro_auroc、micro_auroc、macro_auprc、micro_auprc、label_ranking_average_precision、label_ranking_loss、brier_score。定义里另声明的 coverage_error、expected_calibration_error、maximum_calibration_error 不能做外部测试集区间与成对检验，不进报告。
- 主口径：
- 依赖阈值的指标：交叉验证取**嵌套阈值**（`at_tau_nested`，每个外层折用它自己内层折外预测按逐标签 F1 选出的阈值）；外部测试集取**部署阈值**（`at_tau_per_label`，全量训练的折外预测上选出、冻结后只用一次；micro 指标用按 micro-F1 选出的全局阈值 `at_tau_micro`）。阈值一律在训练一侧的数据上选，从不在被评估的数据上选。
- 大模型学习器（deepseek、glm、qwen）零样本、不用训练数据，不做阈值选择：依赖阈值的指标取模型自己的判定（回答里选中的标签记阳性，`raw`），交叉验证与外部测试集都如此；它的概率是模型对选中标签给出的置信度，没选中的标签记 0。
- 与阈值无关的指标（AUROC、AUPRC、Brier、排序类）只有一种算法。
- 0.5 口径只作敏感性对照，归入存档。
- “±”为跨种子标准差（交叉验证先对每个种子的外层折取均值，再跨种子取样本标准差；外部测试集直接跨种子）：它只反映换种子带来的波动（分折方式、调参路径、训练随机性），不是置信区间，不含抽样不确定性，不能用来判断两个模型的差别是否显著。

## 框架与定义

- 锁文件：冻结于 2026-10-14T19:12:54+08:00；框架 1.0.0（提交 2cb76b480af270fce38e5daa4c402daa9b5bd4e4）；定义文件 `bishan_tizhong_experiment.py` SHA-256 `44931bf0c7455b595517ebf482ecc823471b5a45cc1fe15d4565d25663921e37`；修订 0 次。
- knn（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- decision_tree（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- random_forest（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- extra_trees（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- dannv2_beta（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- realmlp（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- tabpfn_v3（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- tabicl（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- tabdpt（obesity_syndrome）：框架 1.0.0，提交 efdacde48e58581eee80fa5984c6e6c503c712f0，30 个分片。
- deepseek（obesity_syndrome）：框架 1.0.0，提交 8c6e2abad84b5115b7f73d91e38db2f50a82bf2d，30 个分片。
- glm（obesity_syndrome）：框架 1.0.0，提交 8c6e2abad84b5115b7f73d91e38db2f50a82bf2d，30 个分片。
- qwen（obesity_syndrome）：框架 1.0.0，提交 8c6e2abad84b5115b7f73d91e38db2f50a82bf2d，30 个分片。

## 环境

- Python：3.12.13 | packaged by Anaconda, Inc. | (main, Jul  9 2026, 14:38:16) [GCC 14.3.0]
- 平台：Linux-5.15.0-119-generic-x86_64-with-glibc2.35
- 框架版本：1.0.0
- 依赖清单：各分片 seed_<种子>/environment_locks/（如 pip_freeze_ffa22bca.txt）
