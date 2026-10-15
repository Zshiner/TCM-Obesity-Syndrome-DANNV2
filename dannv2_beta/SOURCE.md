# DANNV2-beta 源码说明

论文中的分布感知网络 **DANNV2-beta** 的实现。论文把它当作固定模型使用：固定配方，不做超参数搜索，一个网络同时输出 11 个证素的概率。

## 文件与校验值

| 文件 | 作用 | SHA-256 |
|---|---|---|
| `dannv2_beta.py` | DANNV2 参考实现（定稿配方，2026-08-19 固化），单标签版。本研究中逐字节冻结、不改 | `ce8c912de29e05f8a611ab0b78c2a1e5e70af0a7d40ad8d9dee54953cbf66fc8` |
| `dannv2_beta_ml.py` | 原生多标签版（2026-10-02），论文实际使用的就是它。导入时先核对 `dannv2_beta.py` 的 SHA-256，再复用其中全部定稿常量与未改动的组件；文末是 scikit-learn 风格的估计器 `DANNV2BetaMLClassifier` | `ba52edf595e8bb19612937d8f5c99595f292b8f89cd99a4369de7d7663ff1ce2` |
| `dannv2_weights.py` | 模型自带的特征权值（2026-10-11 定稿）：只读已训练的模型，给出“特征 → 证素”的权值，不参与训练、不改预测。论文的可解释性结果（`results/可解释性/DANNV2-beta_*`）由它算出 | `cf92bc4eb181a126a5b56d87dce876fdbaf35f5b8fd0184935c958e1ab351b43` |
| `test_dannv2_beta_ml.py` | 多标签版的快速验收：折点等价、形状与范围、同种子可复现、序列化往返（需要数据文件） | `272ac1bf60171c562d061fb042f86e76b02e0d9c70c1f8dbed3e74d71ea281a8` |
| `sanity_l1_vs_original.py` | 合理性检查：只有 1 个证素时，与单标签原版比较宏 F1 与 AUROC（需要数据文件） | `44ae92a4e466fa4c560c379863458e00d4533bd536838f3b6319bd4f4c1e6dca` |

## 多标签版相对单标签原版的 5 处改写

1. 读标签：直接接收 N × L 的 0/1 矩阵。
2. 证据分段编码的折点：取各证素二分类基尼之和（只有 1 个证素时与原版逐位相同，已测）。
3. 损失与输出：BCE + sigmoid，各证素独立；早停用多标签宏 F1（阈值 0.5）；输出 N × L 概率，即各成员 sigmoid(logit/T) 的平均。
4. 类先验偏置：τ·log[(n⁺+0.5)/(n⁻+0.5)]。
5. 两段自蒸馏：段一用多标签迭代分层三折（固定种子）；软目标逐证素锐化 q²/(q²+(1−q)²) 后，与硬标签按 0.2 混合。

另有两处与配方无关的工程适配：支持 GPU（参数初始化仍在 CPU 上按原版顺序完成，再搬到设备上）；交付的模型可被 joblib 序列化，序列化时自动转回 CPU。

## 用法

```python
import sys
sys.path.insert(0, "dannv2_beta")          # 本目录
from dannv2_beta_ml import DANNV2BetaMLClassifier

clf = DANNV2BetaMLClassifier(device="cuda", random_state=0)   # device 可为 "cpu"、"cuda"、"cuda:N"、"auto"
clf.fit(X_train, Y_train)                  # X：N × 348 特征；Y：N × 11 的 0/1 矩阵
P = clf.predict_proba(X_test)              # N × 11 概率

# 论文里每个种子的 random_state 是框架按种子派生的运行种子，记在
# bishan_tizhong/results/shards/runs/obesity_syndrome/dannv2_beta/s0<种子>final/ 的 manifest.json（ctor_args）里。
# GPU 训练对显卡型号与 torch 版本敏感：换设备重训得到的模型与论文的不会逐位相同。
```

加载 Release 里的 DANNV2-beta 最终模型前，同样要先把本目录加入 `sys.path`（反序列化要用到 `dannv2_beta_ml` 里的类），见 `models/README.md`。
