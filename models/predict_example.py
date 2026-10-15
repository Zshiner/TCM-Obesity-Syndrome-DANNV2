"""用公开的最终模型对病例做证素预测（示例）。

    python models/predict_example.py --model <模型文件.joblib> --cases <病例.csv> --out <结果.csv>

- 模型文件见 GitHub Release，文件名为 <学习器>_seed<种子>.joblib（学习器：decision_tree、
  random_forest、extra_trees、dannv2_beta、realmlp）。
- 病例 CSV：每行一例，列名为 bishan_tizhong/data/数据字典.csv 里 348 个特征的名称（列顺序不限，
  多余的列忽略）。取值编码同数据字典：年龄为岁数，性别 1 男、0 女，病程 1–4，三者未载记 −1；
  其余特征 1 阳性、0 阴性。
- 输出：每例 11 个证素的概率，以及按该模型该种子的部署阈值（results/部署阈值.csv，概率 ≥ 阈值
  记阳性）得到的判定。论文外部测试集的结果就是这样算出来的。

依赖：numpy、pandas、joblib、scikit-learn 1.8.0（模型按此版本保存，换版本可能无法加载或结果有出入）；
DANNV2-beta 另需 torch 与本仓库的 dannv2_beta/ 源码；RealMLP 另需 torch 与 pytabkit 1.7.3。
"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["心", "脾", "肝", "肾", "痰", "湿", "热", "气滞", "血瘀", "气虚", "阳虚"]


def positive_proba(model, X) -> np.ndarray:
    """各证素的正类概率，N × 11。

    scikit-learn 的多输出模型（树模型、RealMLP 的逐证素分类器组）返回 11 个 (N, 2) 数组，
    逐个取正类一列；DANNV2-beta 直接返回 (N, 11)。
    """
    p = model.predict_proba(X)
    if not isinstance(p, list):
        return np.asarray(p)
    ests = getattr(model, "estimators_", None)
    cols = []
    for k, pk in enumerate(p):
        pk = np.asarray(pk)
        if pk.shape[1] == 2:
            cols.append(pk[:, 1])
        else:  # 训练时该证素只有一类（本研究没有出现）
            classes = ests[k].classes_ if ests is not None else model.classes_[k]
            cols.append(np.full(len(pk), float(int(classes[0]) == 1)))
    return np.column_stack(cols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="模型文件，如 dannv2_beta_seed0.joblib")
    ap.add_argument("--cases", type=Path, required=True, help="病例 CSV（列名为 348 个特征名）")
    ap.add_argument("--out", type=Path, required=True, help="输出 CSV")
    args = ap.parse_args()

    m = re.fullmatch(r"(.+)_seed(\d)\.joblib", args.model.name)
    if m is None:
        raise SystemExit("模型文件名应为 <学习器>_seed<种子>.joblib")
    learner, seed = m.group(1), int(m.group(2))
    if learner == "dannv2_beta":
        sys.path.insert(0, str(ROOT / "dannv2_beta"))  # 反序列化要用到模型类定义
    model = joblib.load(args.model)

    dd = pd.read_csv(ROOT / "bishan_tizhong/data/数据字典.csv", encoding="utf-8-sig")
    features = dd.loc[dd["角色"] == "特征", "名称"].tolist()
    cases = pd.read_csv(args.cases, encoding="utf-8-sig")
    missing = [f for f in features if f not in cases.columns]
    if missing:
        raise SystemExit(f"病例表缺 {len(missing)} 个特征列，如：{missing[:5]}")
    X = cases[features].astype(float)
    if learner == "realmlp":
        X = np.asarray(X, dtype=np.float32)  # 与训练、评估时的输入精度一致
    prob = positive_proba(model, X)

    thr = pd.read_csv(ROOT / "results/部署阈值.csv", encoding="utf-8-sig")
    row = thr[(thr["学习器"] == learner) & (thr["种子"] == seed)]
    if len(row) != 1:
        raise SystemExit(f"results/部署阈值.csv 里没有 {learner} 种子 {seed} 的阈值")
    tau = row[LABELS].to_numpy(dtype=float).ravel()

    out = pd.DataFrame(prob, columns=[f"{lab}_概率" for lab in LABELS])
    for k, lab in enumerate(LABELS):
        out[f"{lab}_判定"] = (prob[:, k] >= tau[k]).astype(int)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"{learner} 种子 {seed}：{len(out)} 例 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
