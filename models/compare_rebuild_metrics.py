"""换一张显卡按记录重训 DANNV2-beta 与 RealMLP，外部测试集指标与论文相差多少。

1. 口径校验：用论文存档的预测（各全量分片 predictions/<单元>/final_test.npz 里的 y_prob）按同一套
   公式算 15 项指标，与论文逐种子长表（results/总体性能_长表.csv 的外部测试集行）逐项比对。
2. 重训模型：加载在另一张显卡上用实验框架 `rebuild --final` 按记录的参数与种子重训的最终模型，
   在同一张显卡上预测外部测试集，按论文的部署阈值算同样的指标，与论文逐种子比较。

用法：python compare_rebuild_metrics.py <存档的 runs/obesity_syndrome 目录> <重训模型根> <部署阈值.csv>
                                        <总体性能_长表.csv> <输出目录>
  重训模型根下为 seed_<种子>/obesity_syndrome/<单元>/final/best_model.joblib（rebuild 逐种子各给一个 --out）。

需要：存档的分片预测文件（含特征与真值，随病案数据申请）、能导入实验框架（重训的 RealMLP 模型里有
框架的适配器类）、dannv2_beta/ 源码在 sys.path 上、一张 CUDA 显卡。
输出：口径校验.csv、换显卡重训_指标汇总.csv、换显卡重训_逐种子.csv。
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn import metrics as M

runs, rebuilt, thr_csv, long_csv, out = (Path(a) for a in sys.argv[1:6])
out.mkdir(parents=True, exist_ok=True)
UNITS = {"dannv2_beta": "dannv2_beta", "realmlp": "br-realmlp"}
DEV = "cuda:0"


def metric_values(Y, P, Pm, prob):
    """论文的 15 项指标。P 为按逐证素部署阈值的判定，Pm 为按 micro 共用阈值的判定，prob 为概率。"""
    return {
        "macro_f1": M.f1_score(Y, P, average="macro", zero_division=0),
        "micro_f1": M.f1_score(Y, Pm, average="micro", zero_division=0),
        "weighted_f1": M.f1_score(Y, P, average="weighted", zero_division=0),
        "sample_f1": M.f1_score(Y, P, average="samples", zero_division=0),
        "macro_precision": M.precision_score(Y, P, average="macro", zero_division=0),
        "macro_recall": M.recall_score(Y, P, average="macro", zero_division=0),
        "subset_accuracy": M.accuracy_score(Y, P),
        "hamming_loss": M.hamming_loss(Y, P),
        "macro_auroc": M.roc_auc_score(Y, prob, average="macro"),
        "micro_auroc": M.roc_auc_score(Y, prob, average="micro"),
        "macro_auprc": M.average_precision_score(Y, prob, average="macro"),
        "micro_auprc": M.average_precision_score(Y, prob, average="micro"),
        "label_ranking_average_precision": M.label_ranking_average_precision_score(Y, prob),
        "label_ranking_loss": M.label_ranking_loss(Y, prob),
        "brier_score": float(np.mean((prob - Y) ** 2)),
    }


def pos_proba(model, X):
    p = model.predict_proba(X)
    if not isinstance(p, list):
        return np.asarray(p, dtype=float)
    return np.column_stack([np.asarray(pk)[:, 1] for pk in p]).astype(float)


def dann_to(model, dev):
    """DANNV2-beta 序列化时转回了 CPU；搬回显卡，与重训时在显卡上预测一致。"""
    m = model.model_
    m.emb.to(dev)
    for k in ("_W", "_s0", "_s1"):
        setattr(m, k, torch.nn.Parameter(getattr(m, k).detach().to(dev)))
    m._r = m._r.detach().to(dev)
    m.device = str(dev)
    return model


thr = pd.read_csv(thr_csv, encoding="utf-8-sig")
long = pd.read_csv(long_csv, encoding="utf-8-sig")
long = long[long["划分"] == "外部测试集"]
names = dict(zip(long["学习器"], long["论文用名"]))
mnames = dict(zip(long["指标"], long["指标论文用名"]))
print(f"torch {torch.__version__}；显卡 {torch.cuda.get_device_name(0)}")

check_rows, seed_rows, long_rows = [], [], []
for learner, unit in UNITS.items():
    for seed in range(5):
        z = np.load(runs / learner / f"s0{seed}final" / f"seed_{seed}" / "obesity_syndrome" / "predictions" / unit
                    / "final_test.npz", allow_pickle=True)
        labels = [str(x) for x in z["label_names"]]
        X = pd.DataFrame(np.asarray(z["X"], dtype=float), columns=[str(x) for x in z["feature_names"]])
        Y = np.asarray(z["y_true"]).astype(int)
        orig = np.asarray(z["y_prob"], dtype=float)
        t = thr[(thr["学习器"] == learner) & (thr["种子"] == seed)]
        tau = t[labels].to_numpy(dtype=float).ravel()
        tau_m = float(t["micro 指标共用阈值"].iloc[0])

        # 1. 口径校验：论文存档的预测 → 指标，对长表
        mo = metric_values(Y, (orig >= tau).astype(int), (orig >= tau_m).astype(int), orig)
        for k, v in mo.items():
            sel = long[(long["学习器"] == learner) & (long["指标"] == k)]
            paper = float(sel[f"种子{seed}"].iloc[0]) if len(sel) == 1 else np.nan
            check_rows.append([learner, seed, k, v, paper, abs(v - paper)])

        # 2. 重训模型：在显卡上预测
        model = joblib.load(rebuilt / f"seed_{seed}" / "obesity_syndrome" / unit / "final" / "best_model.joblib")
        if learner == "dannv2_beta":
            new = pos_proba(dann_to(model, DEV), X)
        else:
            new = pos_proba(model, np.asarray(X, dtype=np.float32))
        mn = metric_values(Y, (new >= tau).astype(int), (new >= tau_m).astype(int), new)
        flips = int(np.sum((orig >= tau) != (new >= tau)))
        row = {"学习器": learner, "论文用名": names[learner], "种子": seed,
               "概率平均绝对差": float(np.mean(np.abs(new - orig))), "概率最大绝对差": float(np.max(np.abs(new - orig))),
               "部署阈值下判定翻转数": flips, "判定总数": int(Y.size)}
        for k in mo:
            row[f"{mnames[k]}（论文）"] = mo[k]
            row[f"{mnames[k]}（重训）"] = mn[k]
            long_rows.append([learner, k, mo[k], mn[k]])
        seed_rows.append(row)
        print(f"{learner} 种子 {seed}：Macro-F1 {mn['macro_f1']:.4f}（论文 {mo['macro_f1']:.4f}），"
              f"Brier {mn['brier_score']:.4f}（论文 {mo['brier_score']:.4f}），翻转 {flips} / {Y.size}", flush=True)

chk = pd.DataFrame(check_rows, columns=["学习器", "种子", "指标", "本脚本", "论文长表", "差"])
chk.to_csv(out / "口径校验.csv", index=False, encoding="utf-8-sig")
print(f"口径校验：{len(chk)} 项，最大差 {chk['差'].max():.3g}，长表缺项 {int(chk['论文长表'].isna().sum())}")

pd.DataFrame(seed_rows).to_csv(out / "换显卡重训_逐种子.csv", index=False, encoding="utf-8-sig")

lr = pd.DataFrame(long_rows, columns=["学习器", "指标", "论文", "重训"])
summ = []
for (learner, k), g in lr.groupby(["学习器", "指标"], sort=False):
    sel = long[(long["学习器"] == learner) & (long["指标"] == k)]
    lo, hi, sd = (float(sel[c].iloc[0]) for c in ("合并种子95%区间下限", "合并种子95%区间上限", "标准差"))
    d = g["重训"].to_numpy() - g["论文"].to_numpy()
    summ.append({"学习器": learner, "论文用名": names[learner], "指标": k, "指标论文用名": mnames[k],
                 "论文均值": g["论文"].mean(), "论文种子标准差": sd, "论文合并种子95%区间下限": lo,
                 "论文合并种子95%区间上限": hi, "重训均值": g["重训"].mean(), "重训种子标准差": g["重训"].std(ddof=1),
                 "平均差（重训−论文）": d.mean(), "逐种子差最小": d.min(), "逐种子差最大": d.max(),
                 "平均差绝对值÷论文种子标准差": abs(d.mean()) / sd, "重训均值在论文95%区间内": "是" if lo <= g["重训"].mean() <= hi else "否"})
pd.DataFrame(summ).to_csv(out / "换显卡重训_指标汇总.csv", index=False, encoding="utf-8-sig")
print("写出", out)
