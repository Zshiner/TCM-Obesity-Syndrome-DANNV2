# -*- coding: utf-8 -*-
"""DANNV2-beta 多标签版合理性检查：只有 1 个证素时，与冻结副本（原版二分类）宏 F1 是否相近。

  python sanity_l1_vs_original.py <data.json> --label <证素序号> --impl orig|ml [--device cpu|cuda]

同一证素、同一 80/20 分层划分（按该证素，random_state=0）、同一种子 0：
  orig：冻结副本 fit_dannv2（二分类，标签为 0/1 类别号，CPU）
  ml  ：多标签版 fit_dannv2_ml（L=1 的 N×1 矩阵）
两者都按"正类概率 ≥ 0.5"判定，报告二分类宏 F1（0、1 两类平均）与 AUROC。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--label", type=int, required=True)
    ap.add_argument("--impl", choices=["orig", "ml"], required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    d = json.loads(Path(a.data).read_text(encoding="utf-8"))
    X = np.array([s["features"] for s in d["train_samples"]], dtype=np.float64)
    y = np.array([s["labels"][a.label] for s in d["train_samples"]], dtype=np.int64)
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    t0 = time.time()
    if a.impl == "orig":
        import dannv2_beta as orig
        m = orig.fit_dannv2(X, y, tr, 0)
        P = m.predict_proba(X[te])
        pos = P[:, list(m.classes_).index(1)]
    else:
        import dannv2_beta_ml as ml
        m = ml.fit_dannv2_ml(X, y[:, None], tr, 0, device=a.device)
        pos = m.predict_proba(X[te])[:, 0]
    dt = time.time() - t0
    pred = (pos >= 0.5).astype(int)
    out = {
        "label": d["label_names"][a.label], "label_idx": a.label, "impl": a.impl, "device": a.device,
        "n_train": int(len(tr)), "n_test": int(len(te)), "pos_rate_test": float(y[te].mean()),
        "e_star": int(m.e_star_), "macro_f1": float(f1_score(y[te], pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y[te], pos)), "minutes": round(dt / 60, 2),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
