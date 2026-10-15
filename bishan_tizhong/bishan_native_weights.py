# -*- coding: utf-8 -*-
"""璧山 DANNV2-beta 原生权值表（定稿 2026-10-11）。

  python bishan_native_weights.py [--model <best_model.joblib>] [--data <data.json>]
                                  [--device cuda] [--steps 1024] [--out <目录>]

放在 bishan_tizhong/ 运行目录下（与 explain_common.py 同级），dannv2_weights.py 放在
同级或 ../dannv2_beta/ 下。只读模型与数据，不改 dannv2_beta_ml.py，不写回实验目录。
缺省模型取 explain/dannv2_beta/fidelity.json 里记的部署级全量模型（种子 0），
缺省产出写到 explain/dannv2_beta/native_weights/：

  native_weight_table.csv      定稿权值表：行 = 特征取值，列 = 类别、特征、取值、人数、11 证素
  native_segment_table.csv     分段权值表：年龄、病程在组件 A 里的每一段一行，列 = 特征、段下界、段上界、
                               人数、11 证素，数值是越过该段一整段的边际权值
  native_weights_percase.npz   逐例权值 W（1586×348×11，float32）、逐例逐段权值 W_seg（1586×541×11）、
                               D、D_ref、P、is_test、case_id，以及两张表的未打折平均与预计病例数
  native_weights_check.json    核对数字与全部参数

解释集 = 开发集 1278 例 + 测试集 308 例，共 1586 例（先开发集后测试集）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "dannv2_beta"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import explain_common as ec        # noqa: E402
import dannv2_beta_ml as ml        # noqa: E402
import dannv2_weights as nw        # noqa: E402

LEVEL_NAMES = {"性别": {0.0: "女", 1.0: "男"}}     # 0 为女、1 为男（妇科症状只出现在 0）
CHECK_CELLS = [("腰酸", "肾"), ("苔腻", "湿"), ("苔黄", "热"), ("舌紫", "血瘀"), ("畏寒(冷)", "阳虚"),
               ("舌有点刺", "热"), ("苔灰黑", "心"), ("畏寒(冷)", "热"), ("苔黄", "阳虚")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="部署级全量模型（种子 0）的 best_model.joblib")
    ap.add_argument("--data", default=None)
    ap.add_argument("--device", default=None, help="缺省有 GPU 用 GPU")
    ap.add_argument("--steps", type=int, default=nw.STEPS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = ec.load_data(args.data)
    fn, ln = data.feature_names, data.label_names
    model_path = Path(args.model) if args.model else Path(json.loads(
        (ec.EXPLAIN_DIR / "dannv2_beta" / "fidelity.json").read_text(encoding="utf-8"))["model_path"])
    model = joblib.load(model_path)
    out = Path(args.out) if args.out else ec.EXPLAIN_DIR / "dannv2_beta" / "native_weights"
    out.mkdir(parents=True, exist_ok=True)

    X = np.vstack([data.X_dev, data.X_test]).astype(np.float64)
    n_dev, N = len(data.X_dev), len(X)
    ref, ref_record = ec.build_reference_row(data)
    P_before = np.asarray(model.predict_proba(X), dtype=np.float64)
    Y_model = np.asarray(model.predict(X)).astype(int)

    # ---- 逐例权值 ----
    t0 = time.time()

    def progress(done, total):
        if done == 10 or done % 128 == 0:
            el = time.time() - t0
            print(f"   路径 {done}/{total} 步，已用 {el:.0f} 秒，预计共 {el * total / done / 60:.1f} 分钟", flush=True)

    res = nw.path_weights(model, X, ref, steps=args.steps, device=args.device, progress=progress)
    runtime = time.time() - t0
    W, D, D_ref = res["W"], res["D"], res["D_ref"]
    P_double = 1.0 / (1.0 + np.exp(-D))
    rebuilt = ((D_ref[None, :] + W.sum(1)) >= 0).astype(int)
    P_after = np.asarray(model.predict_proba(X), dtype=np.float64)
    print(f"逐例权值：{N} 例，{args.steps} 步，{runtime:.0f} 秒")
    print(f"  基线 + 权值之和 − 模型输出对数几率：最大 {res['err_max']:.2e}，平均 {res['err_mean']:.2e}")
    print(f"  双精度前向与 predict_proba 最大差 {np.abs(P_double - P_before).max():.2e}")
    print(f"  由权值复算的判定与模型判定不一致 {int((rebuilt != Y_model).sum())} / {Y_model.size}")
    print(f"  算完后 predict_proba 与算之前逐位相同：{bool(np.array_equal(P_before, P_after))}")
    print(f"  各段权值加回特征权值的最大差 {res['seg_err_max']:.2e}")

    # ---- 表的行 ----
    sym = [f for f, n in enumerate(fn) if n.startswith(ec.SYMPTOM_PREFIXES)]
    defs = []                                                    # (类别, 特征, 取值, 列号, 掩码)
    for f, name in enumerate(fn):
        if f in sym:
            continue
        cuts = nw.cut_order(data.X_dev[:, f], data.Y_dev.astype(np.float64), ml._best_split_ml, bins=ml.BINS)
        edges = nw.net_of(model)._enc.edges_[f]                  # 重放的折点应与模型里存的折点一致
        lo, hi = float(data.X_dev[:, f].min()), float(data.X_dev[:, f].max())
        replay = np.unique([t for t in cuts if lo < t < hi])
        stored = np.asarray([] if edges is None else edges[1:-1], dtype=np.float64)
        if len(replay) != len(stored) or (len(stored) and np.abs(replay - stored).max() > 1e-9):
            raise RuntimeError(f"{name}：重放的折点与模型里存的折点不一致")
        for label, mask in nw.value_rows(X[:, f], ref[f], cuts=cuts, missing=ec.MISSING,
                                         level_names=LEVEL_NAMES.get(name)):
            defs.append(("一般", name, label, f, mask))
    for f in sorted(sym, key=lambda f: -(X[:, f] > 0).sum()):   # 症状按出现人数从多到少
        if (X[:, f] > 0).any():
            defs.append((fn[f].split("：")[0], fn[f].split("：", 1)[1], "出现", f, X[:, f] > 0))

    # ---- 权值表 ----
    tab = nw.weight_table(model, X, W, [(d[3], d[4]) for d in defs], ref)
    val = tab["value"]
    T = pd.DataFrame([dict(类别=d[0], 特征=d[1], 取值=d[2], 人数=int(tab["n"][r]),
                           **{lab: round(float(val[r, c]) + 0.0, 3) for c, lab in enumerate(ln)})
                      for r, d in enumerate(defs)])
    T.to_csv(out / "native_weight_table.csv", index=False, encoding="utf-8-sig")
    is_sym = np.array([d[2] == "出现" for d in defs])
    names = [d[1] for d in defs]
    vs = val[is_sym]
    print(f"权值表：{len(defs)} 行 × {len(ln)} 证素，症状 {int(is_sym.sum())} 个，空格 {int(T[ln].isna().sum().sum())} 个")
    print(f"  症状格：≥ +1 的 {int((vs >= 1).sum())}，≥ +2 的 {int((vs >= 2).sum())}，≤ −1 的 {int((vs <= -1).sum())}，≤ −1.5 的 {int((vs <= -1.5).sum())}")
    top = nw.top_weights(val, names, ln, k=5, rows=np.flatnonzero(is_sym))
    for lab in ln:
        print(f"  【{lab}】绝对值前五：" + "、".join(f"{n} {v:+.2f}" for n, v in top[lab]))

    # ---- 分段权值表：连续特征在组件 A 里的每一段一行 ----
    cont = [fn.index(n) for n in ec.CONTINUOUS]
    seg = nw.segment_table(model, X, res, ref, features=cont)
    G = pd.DataFrame([dict(特征=fn[int(seg["feature"][r])], 段下界=float(seg["lo"][r]), 段上界=float(seg["hi"][r]),
                           人数=int(seg["n"][r]),
                           **{lab: round(float(seg["value"][r, c]) + 0.0, 3) for c, lab in enumerate(ln)})
                      for r in range(len(seg["n"]))])
    G.to_csv(out / "native_segment_table.csv", index=False, encoding="utf-8-sig")
    print(f"分段权值表：{len(G)} 行（{'、'.join(ec.CONTINUOUS)}），没有病人碰到的段 {int((seg['n'] == 0).sum())} 个")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(G.round(2).to_string(index=False))

    np.savez_compressed(
        out / "native_weights_percase.npz", W=W.astype(np.float32), D=D, D_ref=D_ref, P=P_before,
        W_seg=res["W_seg"].astype(np.float32), seg_feature=res["seg_feature"], seg_lo=res["seg_lo"], seg_hi=res["seg_hi"],
        segtab_feature=seg["feature"], segtab_lo=seg["lo"], segtab_hi=seg["hi"], segtab_n=seg["n"],
        segtab_amount=seg["amount"], segtab_value=seg["value"], segtab_mean=seg["mean"],
        segtab_E_with=seg["E_with"], segtab_E_ref=seg["E_ref"],
        is_test=np.arange(N) >= n_dev, case_id=np.array(data.case_ids),
        feature_names=np.array(fn), label_names=np.array(ln), reference_row=ref,
        table_value=val, table_mean=tab["mean"], table_E_with=tab["E_with"], table_E_ref=tab["E_ref"],
        table_n=tab["n"], table_rows=np.array([f"{d[0]}|{d[1]}|{d[2]}" for d in defs]),
        table_feature_index=np.array([d[3] for d in defs]))
    cells = {}
    for s, lab in CHECK_CELLS:
        hit = [r for r, d in enumerate(defs) if d[1] == s and d[2] == "出现"]
        if hit and lab in ln:
            r, c = hit[0], ln.index(lab)
            cells[f"{s}→{lab}"] = {"人数": int(tab["n"][r]), "出现时平均": round(float(tab["mean"][r, c]), 4),
                                   "预计病例数": round(float(max(tab["E_with"][r, c], tab["E_ref"][r, c])), 2),
                                   "权值": round(float(val[r, c]), 4)}
    check = {
        "method": "dannv2_native_weights",
        "definition": "逐例权值 = 模型输出概率对数几率从参照样本到病人沿组件 A 编码路径的累加；"
                      "权值表格值 = 出现时平均 × E/(E+1)，E 为这批病人里模型预计的该证病例数（带着该特征与换成参照值各算一次取大者）",
        "steps": int(args.steps), "pseudo_cases": nw.PSEUDO_CASES, "n_cuts": nw.N_CUTS, "max_levels": nw.MAX_LEVELS,
        "explain_set": {"n": int(N), "n_dev": int(n_dev), "n_test": int(N - n_dev)},
        "reference_row": ref_record,
        "model_path": str(model_path), "model_sha256": ec.sha256_file(model_path),
        "weights_module_sha256": ec.sha256_file(Path(nw.__file__)),
        "completeness_err_max": res["err_max"], "completeness_err_mean": res["err_mean"],
        "double_forward_vs_predict_proba": float(np.abs(P_double - P_before).max()),
        "decision_mismatch": int((rebuilt != Y_model).sum()), "decision_total": int(Y_model.size),
        "predict_proba_identical_after": bool(np.array_equal(P_before, P_after)),
        "segment_sum_vs_feature_weight_max": res["seg_err_max"],
        "segment_table_rows": int(len(G)), "segment_table_untouched": int((seg["n"] == 0).sum()),
        "segment_table": [{"特征": fn[int(seg["feature"][r])], "段": [float(seg["lo"][r]), float(seg["hi"][r])],
                           "人数": int(seg["n"][r]),
                           "权值": {lab: round(float(seg["value"][r, c]), 4) for c, lab in enumerate(ln)}}
                          for r in range(len(seg["n"]))],
        "table_rows": int(len(defs)), "table_symptoms": int(is_sym.sum()), "table_empty_cells": int(T[ln].isna().sum().sum()),
        "symptom_cells": {"ge_1": int((vs >= 1).sum()), "ge_2": int((vs >= 2).sum()),
                          "le_-1": int((vs <= -1).sum()), "le_-1.5": int((vs <= -1.5).sum())},
        "check_cells": cells, "top5_abs_by_label": top,
        "runtime_s": round(runtime, 1), "device": str(args.device or "auto"),
        "versions": ec.versions(), "date": ec.now_beijing(),
    }
    (out / "native_weights_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
