# -*- coding: utf-8 -*-
"""随机森林 TreeSHAP（《可解释性补实验方案（定稿）》§4）。

  python explain_rf.py --model <best_model.joblib> [--npz <final_test.npz>] [--data <data.json>]

设置：shap.TreeExplainer(rf, data=参考样本一行, feature_perturbation="interventional")；解释集 = 外部测试集 308 例；
sklearn 多输出森林返回 (308, 348, 2L) 按 (证素 j 的类 0, 类 1) 交错，取类 1；缩放系数 k 以可加性核对定
（(sv_pos.sum(1) + expected_value) × k ≈ predict_proba 正类列，误差 < 1e-6）。该学习器的自声明
（feature_importances_ 合并向量，不分证素）关闭，不作产出。只读模型；产出写到 explain/random_forest/。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import explain_common as ec

OUT = ec.EXPLAIN_DIR / "random_forest"


def positive_class_values(sv: np.ndarray, rf, n_labels: int) -> tuple[np.ndarray, np.ndarray, str]:
    """把 TreeExplainer 的输出整理成 (N, F, L) 正类值与 (L,) 正类基线；返回布局说明。"""
    import shap  # noqa: F401
    arr = np.asarray(sv)
    layout = f"raw shape {arr.shape}"
    if arr.ndim == 3 and arr.shape[2] == 2 * n_labels:
        pos = arr[:, :, 1::2]
        layout += "；第三维按 (证素 j 类 0, 类 1) 交错，取 1::2"
    elif arr.ndim == 4 and arr.shape[0] == n_labels and arr.shape[3] == 2:  # list of (N,F,2) 堆叠
        pos = np.moveaxis(arr[:, :, :, 1], 0, -1)
        layout += "；按证素堆叠的 (N,F,2)，取类 1 后移到末维"
    elif arr.ndim == 4 and arr.shape[0] == 2 * n_labels:  # list of 2L 个 (N,F)
        pos = np.moveaxis(arr[1::2], 0, -1)
        layout += "；2L 个 (N,F) 按 (类 0, 类 1) 交错，取 1::2"
    else:
        raise RuntimeError(f"无法识别 shap_values 的形状 {arr.shape}")
    return pos, layout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    import shap

    data = ec.load_data(args.data)
    ref, ref_path = ec.write_reference_row(data)
    model_path = Path(args.model).resolve()
    npz_path = Path(args.npz).resolve() if args.npz else (
        model_path.parents[3] / "predictions" / model_path.parents[1].name / "final_test.npz")
    saved = ec.load_final_npz(npz_path, data)
    rf = joblib.load(model_path)
    from sklearn.ensemble import RandomForestClassifier
    if not isinstance(rf, RandomForestClassifier):
        raise RuntimeError(f"期望 sklearn RandomForestClassifier，收到 {type(rf).__name__}")
    L = len(data.label_names)
    if rf.n_outputs_ != L:
        raise RuntimeError(f"森林有 {rf.n_outputs_} 个输出，证素 {L} 个")

    def proba(X):
        P = rf.predict_proba(X)
        return np.column_stack([p[:, 1] for p in P])

    with ec.Timer() as t_pred:
        P_before = proba(data.X_test)
    fid_before = ec.max_abs_diff(P_before, saved["y_prob"])
    print(f"载入后 predict_proba 与存档 y_prob 最大绝对差：{fid_before:.3g}（阈值 1e-6）")
    if fid_before > 1e-6:
        print("保真核对不通过，停止。")
        return 2

    ref_df = pd.DataFrame([ref], columns=data.feature_names)
    X_df = pd.DataFrame(data.X_test, columns=data.feature_names)
    with ec.Timer() as t_run:
        explainer = shap.TreeExplainer(rf, data=ref_df, feature_perturbation="interventional")
        sv = explainer.shap_values(X_df)
    pos, layout = positive_class_values(sv, rf, L)
    ev = np.asarray(explainer.expected_value, dtype=np.float64).ravel()
    if ev.size == 2 * L:
        ev_pos = ev[1::2]
    elif ev.size == L:
        ev_pos = ev
    else:
        raise RuntimeError(f"expected_value 大小 {ev.size} 无法对应 {L} 个证素")
    print(f"TreeSHAP：{t_run.seconds:.1f} 秒；{layout}；expected_value 大小 {ev.size}")

    # 缩放核对：找使可加性误差 < 1e-6 的 k
    recon = pos.sum(axis=1) + ev_pos[None, :]
    errs = {k: float(np.abs(k * recon - P_before).max()) for k in (1, 2, L, 2 * L)}
    k_ok = [k for k, e in errs.items() if e < 1e-6]
    k_ls = float((recon * P_before).sum() / (recon * recon).sum())
    print(f"可加性误差按 k：{errs}；最小二乘 k = {k_ls:.6f}")
    if not k_ok:
        print("没有 k 使可加性误差 < 1e-6，停止；请核对 shap 版本与输出布局。")
        return 3
    k = int(k_ok[0])
    shap_vals = k * pos
    base = k * ev_pos
    additivity_err = errs[k]
    # 参考样本自身：SHAP 全 0、预测 = 基线
    P_ref = proba(ref[None, :])
    base_err = ec.max_abs_diff(P_ref[0], base)

    P_after = proba(data.X_test)
    identical_after = bool(np.array_equal(P_before, P_after))
    print(f"归因后 predict_proba 与归因前逐位相同：{identical_after}；k = {k}，可加性误差 {additivity_err:.3g}，基线 vs 参考样本预测差 {base_err:.3g}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "shap_values.npz", shap_values=shap_vals.astype(np.float64), expected_value=base,
        case_id=np.array(data.case_ids), feature_names=np.array(data.feature_names), label_names=np.array(data.label_names),
        scale_factor_k=np.array(k),
    )
    signed = shap_vals.mean(axis=0)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top = ec.write_summary_tables(signed, mean_abs, data, OUT)
    print("正向前 5（带符号均值 SHAP）：")
    ec.print_top5(top)
    check = {
        "scale_factor_k": k, "additivity_max_err": additivity_err, "additivity_err_by_k": errs, "least_squares_k": k_ls,
        "expected_value_vs_reference_row_prediction_max_err": base_err,
        "shap_output_layout": layout, "expected_value_size": int(ev.size),
        "fidelity_max_abs_diff": fid_before, "predict_proba_identical_after_explain": identical_after,
        "runtime_s": round(t_run.seconds, 2), "predict_runtime_s": round(t_pred.seconds, 3),
        "n_trees": int(rf.n_estimators), "n_outputs": int(rf.n_outputs_), "date": ec.now_beijing(),
    }
    (OUT / "additivity_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    ec.update_params("random_forest", {
        "method": "tree_shap", "self_importance": False,
        "self_importance_off_reason": "多输出森林的 feature_importances_ 为合并向量，不分证素",
        "feature_perturbation": "interventional", "background": "reference_row",
        "scale_factor_k": k, "additivity_max_err": additivity_err,
        "model_path": str(model_path), "model_sha256": ec.sha256_file(model_path), "seed": 0,
        "fidelity_max_abs_diff": fid_before, "predict_proba_identical_after_explain": identical_after,
        "runtime_s": check["runtime_s"], "n_trees": check["n_trees"],
        "outputs": ["shap_values.npz", "mean_abs.csv", "mean_signed.csv", "top5_by_label.csv", "additivity_check.json"],
    }, data=data, reference_path=ref_path)
    print(f"产出：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
