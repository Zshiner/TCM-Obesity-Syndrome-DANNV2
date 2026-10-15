"""bishan_tizhong：体重管理人群中医证素多标签辨识——实验定义（TCMHUBD 脚本式接口）。

本文件就是整场实验的定义：数据、每个学习器（数据处理步骤 + 模型 + 搜索空间）、指标、
方法学设定。

用法（用框架环境的 python；相对路径都相对本文件所在目录）：

    python -m tcmhubd check  bishan_tizhong_experiment.py     # 只检查，不训练
    python -m tcmhubd status bishan_tizhong_experiment.py     # 看结果是否齐全
    python -m tcmhubd report bishan_tizhong_experiment.py     # 出汇总报告

设定：

- 零标签样本原样保留：开发集有 3 例 11 个证素全为 0。
- 病程按有序数值声明为连续特征；年龄、性别、病程均以 −1 表示缺失。
- 外 5 折 × 内 3 折；每折 TPE 200 次；5 个种子；测试集 bootstrap 10000 次；开决策曲线。
- 调参目标与主指标都是 macro-F1。做逐证素阈值选择（thresholds=True）：逐证素取 F1 最大
  的阈值，并列取中位，网格 0.01–0.99 共 99 点。调参目标按"交叉选阈值"计算（内层第 k 折
  的阈值只用其余内层折的预测选出）；交叉验证的口径是嵌套阈值（每个外层折用自己内层预测
  选出的阈值）；外部测试集的口径是部署阈值（在全量训练的折外预测上选出并冻结，再作用于
  测试集）。
- 学习器 12 个。原生多标签（一个模型同时输出 11 个证素）5 个：KNN、决策树、随机森林、
  ExtraTrees、DANNV2-beta。单输出分类器 4 个，逐证素各训练一个（Binary Relevance）：
  RealMLP、TabPFN v3、TabICL、TabDPT。通用大模型 3 个（零样本，经接口调用）：
  DeepSeek-V4-Pro、GLM-5.3、Qwen3.8-Max（快照 0902）。
- TabICL、TabDPT 不存模型文件（权重固定、结果可按种子重建）；TabPFN v3 用轻量存档；
  大模型不存模型文件（原始回答在结果目录的 llm_cache/ 与各分片的 llm_audit/）。
- 数据处理是学习器的一部分，每次拟合只用当次训练部分估计：KNN 用
  "缺失标记 → 填补 → 连续特征标准化"三步；树模型直接使用原始特征；DANNV2-beta 是冻结
  配方。不做样本增强。
- 不算 SHAP。

修订记录：
- 2026-10-12 补通用大模型基线（用户批准的方案；框架批次 H）。数据集加大模型提示词用的
  文字说明：标签类别名“证素”；性别、病程、年龄的取值文字（用户给的编码说明：“性别 1 为男、
  0 为女，病程不足 1 个月、1 个月至不足 6 个月、6 个月至不足 5 年、5 年及以上分别记 1 至 4，
  三者未载记 -1。其余特征与证素标签 1 为阳性、0 为阴性，原文未提及的表现按阴性计。”；年龄按
  岁、不足 1 岁换算成月）。这两项不进核心指纹，只进大模型学习器的指纹，原有 9 个学习器的
  分片不受影响。大模型：三家用同一个提示词（同一模板、同一病例写法，框架强制一致），
  思维模式开、采样参数用各家默认；零样本、不调参、不做阈值选择（依赖阈值的指标取模型
  自己的判定）；回答按内容缓存、各种子共用，每例只问一次；回答在 JSON 前后附了说明文字
  时取其中唯一的 JSON 对象（例数记入降级记录）。
"""
import os
import sys
from pathlib import Path

import tcmhubd as th

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 课题组模型 DANNV2-beta（多标签版）。实现在运行目录旁的 dannv2_beta/：冻结副本
# dannv2_beta.py（与 DANN2 定稿 model/ours/dannv2.py 逐字节一致，SHA-256 ce8c912d…f66fc8，
# 导入时校验）+ 多标签版 dannv2_beta_ml.py（SHA-256 ba52edf5…ff1ce2）。
# 可用环境变量 DANNV2_BETA_DIR 改指实现目录。
# ---------------------------------------------------------------------------
_DANN_DIR = Path(os.environ.get("DANNV2_BETA_DIR") or HERE.parent / "dannv2_beta")
if str(_DANN_DIR) not in sys.path:
    sys.path.insert(0, str(_DANN_DIR))
from dannv2_beta_ml import DANNV2BetaMLClassifier  # noqa: E402

# ---------------------------------------------------------------------------
# 数据：SHA-256 736E0F11AC18419BA786D1277DD55980B4184F4E0E1F4A4C6D419F08F5093DF1。
# 开发集 1278 例、测试集 308 例；特征 348 个；证素 11 个
# （心、脾、肝、肾、痰、湿、热、气滞、血瘀、气虚、阳虚）。
# 非 0/1 特征 3 个，均以 −1 表示缺失：
#   年龄（连续；开发集缺 22 例、测试集缺 1 例）
#   性别（0/1；开发集缺 6 例、测试集缺 1 例）
#   病程（1–4 的有序等级；开发集缺 249 例、测试集缺 44 例）
# ---------------------------------------------------------------------------
data = th.Dataset.from_json(
    HERE / "data" / "obesity_syndrome" / "data.json",
    missing={"年龄": -1, "性别": -1, "病程": -1},
    continuous=["年龄", "病程"],
    # 大模型提示词用的文字说明（2026-10-12 修订；只影响大模型学习器）
    label_kind="证素",
    value_text={
        "性别": {0: "女", 1: "男"},
        "病程": {1: "不足 1 个月", 2: "1 个月至不足 6 个月", 3: "6 个月至不足 5 年", 4: "5 年及以上"},
        "年龄": "age_years",
    },
)

# ---------------------------------------------------------------------------
# 数据处理步骤（显式写出，不依赖内置默认值）
# ---------------------------------------------------------------------------
# 尺度敏感的模型（近邻距离依赖特征尺度）用这三步：
#   缺失标记：年龄、性别、病程各加一列"是否缺失"；
#   填补：缺失处用训练部分的中位数（年龄、病程）、众数（性别）；
#   标准化：只对年龄、病程做 z-score，0/1 症状特征与性别保持原样（Gelman 2008）。
scale = [
    th.steps.MissingIndicator(),
    th.steps.Impute(),
    th.steps.Standardize(columns=th.cols.continuous),
]

learners = [
    # —— 经典机器学习（CPU，调参，原生多标签）——
    th.learners.get("knn", steps=scale),
    th.learners.get("decision_tree"),
    th.learners.get("random_forest"),
    th.learners.get("extra_trees"),
    # —— 课题组模型（GPU；固定配方、免调参；全部证素共享主干联合训练）——
    th.Learner("dannv2_beta", model=DANNV2BetaMLClassifier, multilabel="native", device="gpu"),
    # —— 免调参的表格基础模型与 RealMLP（GPU）。它们是单输出分类器，逐证素各训练一个
    #    （Binary Relevance）——
    th.learners.get("realmlp"),
    th.learners.get("tabpfn_v3"),
    th.learners.get("tabicl", save_model=False),
    th.learners.get("tabdpt", save_model=False),
    # —— 通用大模型（零样本，经接口调用；CPU 域；不存模型文件）——
    th.learners.get("deepseek", save_model=False),   # deepseek-v4-pro
    th.learners.get("glm", save_model=False),        # glm-5.3
    th.learners.get("qwen", save_model=False),       # qwen3.8-max-0902
]

experiment = th.Experiment(
    name="bishan_tizhong",
    data=data,
    learners=learners,
    metrics=[
        "macro_f1", "micro_f1", "weighted_f1", "sample_f1",
        "macro_precision", "macro_recall",
        "subset_accuracy", "hamming_loss",
        "macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc",
        "label_ranking_average_precision", "label_ranking_loss", "coverage_error",
        "brier_score", "expected_calibration_error", "maximum_calibration_error",
    ],
    protocol=th.Protocol(
        seeds=[0, 1, 2, 3, 4],
        outer_folds=5,
        inner_folds=3,
        n_trials=200,
        primary_metric="macro_f1",
        multilabel="br",
        thresholds=True,
        bootstrap=10000,
        decision_curve=True,
    ),
    output="results",
)

if __name__ == "__main__":
    experiment.check()
