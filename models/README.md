# 最终模型

每个学习器、每个随机种子各有一个“最终模型”：用全部训练集（1278 例）按该种子调好的参数训练，论文外部测试集（308 例）的结果就出自这些模型。公开的模型放在 [GitHub Release v1.0](https://github.com/Zshiner/TCM-Obesity-Syndrome-DANNV2/releases/tag/v1.0) 附件里，不进仓库；文件名、大小、SHA-256 与核对结论见 [`模型清单.csv`](模型清单.csv)，Release 里另附 `SHA256SUMS.txt`，下载后可用 `sha256sum -c SHA256SUMS.txt` 核对。

## 公开了哪些，没公开的为什么

| 学习器 | 最终模型 | 原因 |
|---|---|---|
| Decision Tree、Random Forest、Extra Trees、DANNV2-beta、RealMLP | **已公开**，每个学习器 5 个种子，共 25 个文件 | — |
| KNN | 不公开 | K 近邻没有可学习的权重，“训练”就是把训练病案原样存下来，预测时在其中找与新病例最相似的几例。所以它的模型文件里就是全部 1278 例训练病案的特征和证素标签，公开模型就等于公开病案数据 |
| TabPFN v3 | 不公开 | 预训练表格基础模型靠上下文学习：拟合时不更新网络权重，而是把训练病案存进模型，预测时连同待测病例一起送入预训练网络。模型文件里因此存着全部训练病案，公开它同样等于公开病案数据 |
| TabICL、TabDPT | 没有模型文件 | 与 TabPFN v3 一样靠上下文学习，模型就是“公开的预训练权重 + 全部训练病案”。实验定义里本就设定不保存（`save_model=False`），即使保存了，也会因为含训练病案而不公开 |
| DeepSeek-V4-Pro、GLM-5.3、Qwen3.8-Max | 没有模型文件 | 经厂商接口零样本调用，本研究没有训练或保存任何模型。提示词全文见仓库根目录的 [`README.md`](../README.md) |

这些学习器在外部测试集上的预测概率仍完整公开在 [`predictions/`](../predictions/)。病案数据与实验框架可向通讯作者申请。

## 文件是怎么来的

实验时每个最终模型都保存在运行实验的服务器上，SHA-256 记在 `bishan_tizhong/results/shards/runs/obesity_syndrome/<学习器>/s0<种子>final/seed_<种子>/obesity_syndrome/models/<单元>/manifest.json` 的 `final_model.sha256` 里（RealMLP 的单元名是 `br-realmlp`，其余与学习器同名）。公开文件就是从这些原件导出的：

1. 核对原件的 SHA-256 与实验记录一致。25 个全部一致，原件的 SHA-256 列在清单的“原模型文件 SHA-256”一栏。
2. 去掉对实验框架的依赖。RealMLP 的逐证素分类器原本是框架里的一个薄子类，只在入口把输入转成 float32，没有自己的状态；导出时换回 pytabkit 的原类 `RealMLP_TD_Classifier`。其余模型本来就不依赖框架。
3. DANNV2-beta、RealMLP 的张量搬到 CPU，没有显卡也能加载。
4. 用 joblib 压缩保存（`compress=3`）。

模型参数没有任何改动：导出文件与原件在 CPU 上的预测逐位相同。

## 核对了什么

核对时屏蔽实验框架（`import tcmhubd` 直接报错），DANNV2-beta 的源码用本仓库 [`dannv2_beta/`](../dannv2_beta/) 里的副本，只用公开的东西加载导出文件，对外部测试集 308 例预测，再与论文存档的预测概率比较。

- **原环境**：训练用的服务器，NVIDIA A100 80GB，Linux，torch 2.6.0+cu124，scikit-learn 1.8.0。25 个文件全部**逐位一致**。DANNV2-beta 与 RealMLP 在 GPU 上比，因为论文的预测就是训练完在 GPU 上算的；树模型在 CPU 上比。
- **其他环境**：同一台服务器的 CPU，Windows 11 的 CPU，RTX 5090 D 显卡（torch 2.11.0）。
  - 树模型仍然逐位一致。
  - DANNV2-beta 与 RealMLP 有 float32 舍入级的差异：预测概率最大绝对差分别为 2.4 × 10⁻⁷ 和 8.1 × 10⁻⁷，按部署阈值判定没有一处翻转。每个文件的数值见清单最后两栏。

导出与核对用的脚本是 [`release_models.py`](release_models.py)。它要读原模型文件和含病例特征的存档预测，在本仓库里跑不了，放在这里供查阅。

上面核对的是“用同一个模型预测”。换一张显卡按记录重新训练会得到另一个模型，指标与论文差多少，见仓库根目录 [`README.md`](../README.md) 的“复现预期”一节，以及本目录的 `换显卡重训_指标汇总.csv`、`换显卡重训_逐种子.csv`。

## 怎么用

```
python models/predict_example.py --model random_forest_seed0.joblib --cases 病例.csv --out 结果.csv
```

- 病例表每行一例，列名为 [`bishan_tizhong/data/数据字典.csv`](../bishan_tizhong/data/数据字典.csv) 中 348 个特征的名称，编码同数据字典（年龄为岁数，性别 1 男、0 女，病程 1–4，三者未载记 −1；其余 1 阳性、0 阴性）。
- 输出每例 11 个证素的概率，以及按该模型该种子的部署阈值（[`results/部署阈值.csv`](../results/部署阈值.csv)，概率 ≥ 阈值记阳性）得到的判定。论文外部测试集的判定就是这样算出来的。
- 在 Python 里直接用：`joblib.load` 后调用 `predict_proba`。
  - 树模型和 RealMLP 返回 11 个 (N, 2) 数组，各取第 2 列（正类）；DANNV2-beta 直接返回 N × 11。
  - RealMLP 的输入要转成 `float32`。
  - 加载 DANNV2-beta 前，要先把仓库的 `dannv2_beta/` 加入 `sys.path`。
  - DANNV2-beta 与 RealMLP 加载后在 CPU 上预测，不需要显卡。

依赖版本（与训练时相同）：Python 3.12，scikit-learn 1.8.0，numpy 2.4.4，pandas 3.0.2，joblib 1.5.3；DANNV2-beta、RealMLP 另需 torch（训练时为 2.6.0），RealMLP 还需 pytabkit 1.7.3。其余依赖见 [`environment/pip_freeze.txt`](../environment/pip_freeze.txt)。scikit-learn 的模型文件与版本绑定，换版本可能加载失败或结果有出入。
