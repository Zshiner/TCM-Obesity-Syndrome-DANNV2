# -*- coding: utf-8 -*-
"""DANNV2-beta 特征权值的汇总与记录（接在 bishan_native_weights.py 之后；只读，不算任何新量）。

  python make_dannv2_weights_summary.py

读 explain/dannv2_beta/native_weights/（bishan_native_weights.py 的产出，原样不动）与存档预测 final_test.npz，写：
  explain/dannv2_beta/top5_by_label.csv   每证素正向前 5：只在 345 个症状行（取值“出现”）里排，取权值表格值 > 0 的最大 5 个，
                                          归一化 = 值 / 本列最大值（与另两模型同一规则；三模型并排表与问卷用）
  explain/dannv2_beta/fidelity.json       算权值所用模型在测试集 308 例上的概率与存档预测之差、权值的核对数字、模型路径与校验值
  explain/explain_params.json             dannv2_beta 一条换成 native_weights_check.json 的内容
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import explain_common as ec        # noqa: E402

NW = ec.EXPLAIN_DIR / "dannv2_beta" / "native_weights"
OUT = ec.EXPLAIN_DIR / "dannv2_beta"
FINAL_NPZ = (HERE / "results" / "shards" / "runs" / "obesity_syndrome" / "dannv2_beta" / "s00final" / "seed_0"
             / "obesity_syndrome" / "predictions" / "dannv2_beta" / "final_test.npz")
K = 5
TOP5_RULE = "只在 345 个症状行（取值“出现”）里排；权值表格值 > 0 的最大 5 个；归一化 = 值 / 本列最大值"


def main() -> int:
    data = ec.load_data()
    check = json.loads((NW / "native_weights_check.json").read_text(encoding="utf-8"))
    T = pd.read_csv(NW / "native_weight_table.csv", encoding="utf-8-sig")
    z = np.load(NW / "native_weights_percase.npz", allow_pickle=True)
    fn, ln = [str(x) for x in z["feature_names"]], [str(x) for x in z["label_names"]]
    if fn != data.feature_names or ln != data.label_names:
        raise RuntimeError("逐例权值文件的特征 / 证素名与数据文件不一致")
    if [str(x) for x in z["table_rows"]] != [f"{a}|{b}|{c}" for a, b, c in zip(T["类别"], T["特征"], T["取值"])]:
        raise RuntimeError("权值表的行与逐例权值文件里记的行不一致")
    fidx = z["table_feature_index"].astype(int)
    mean = z["table_mean"]
    E = np.maximum(z["table_E_with"], z["table_E_ref"])

    # ---- 每证素正向前 5（由权值表取，不重算） ----
    sym = (T["取值"] == "出现").to_numpy()
    rows = []
    for c, lab in enumerate(ln):
        col = T[lab].to_numpy(dtype=float)
        top = [r for r in np.argsort(-col, kind="stable") if sym[r] and col[r] > 0][:K]
        for rank, r in enumerate(top, start=1):
            rows.append({"label": lab, "rank": rank, "feature": fn[fidx[r]], "weight": float(col[r]),
                         "normalized": float(col[r] / col[top[0]]), "n_present": int(T["人数"].iloc[r]),
                         "mean_when_present": float(mean[r, c]), "expected_cases": float(E[r, c])})
    top5 = pd.DataFrame(rows)
    top5.to_csv(OUT / "top5_by_label.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    for lab, g in top5.groupby("label", sort=False):
        print(f"  {lab} = " + "、".join(f"{r.feature.split('：')[-1]} {r.weight:+.2f}（{r.normalized:.2f}）" for r in g.itertuples()))

    # ---- 与存档预测的保真 ----
    arch = ec.load_final_npz(FINAL_NPZ, data)
    is_test = z["is_test"].astype(bool)
    if [int(x) for x in z["case_id"]] != data.case_ids or int(is_test.sum()) != len(data.X_test):
        raise RuntimeError("逐例权值文件的测试集行与数据文件不一致")
    fidelity = ec.max_abs_diff(z["P"][is_test], arch["y_prob"])
    record = {
        "method": check["method"],
        "model_path": check["model_path"], "model_sha256": check["model_sha256"],
        "npz_path": FINAL_NPZ.relative_to(HERE).as_posix(), "npz_sha256": ec.sha256_file(FINAL_NPZ),
        "fidelity_max_abs_diff": fidelity,
        "fidelity_definition": "算权值所用模型在测试集 308 例上的 predict_proba 与存档 final_test.npz 的 y_prob 的最大绝对差",
        "predict_proba_identical_after_explain": bool(check["predict_proba_identical_after"]),
        "additivity_max_err": check["completeness_err_max"], "additivity_mean_err": check["completeness_err_mean"],
        "additivity_definition": "参照样本的对数几率 + 各特征权值之和 − 模型输出的对数几率（全部病例 × 11 证素，对数几率单位）",
        "double_forward_vs_predict_proba": check["double_forward_vs_predict_proba"],
        "decision_mismatch": check["decision_mismatch"], "decision_total": check["decision_total"],
        "segment_sum_vs_feature_weight_max": check["segment_sum_vs_feature_weight_max"],
        "table_rows": check["table_rows"], "table_symptoms": check["table_symptoms"], "table_empty_cells": check["table_empty_cells"],
        "segment_table_rows": check["segment_table_rows"], "segment_table_untouched": check["segment_table_untouched"],
        "n_cases": check["explain_set"]["n"], "n_dev": check["explain_set"]["n_dev"], "n_test": check["explain_set"]["n_test"],
        "n_features": len(fn), "n_labels": len(ln), "steps": check["steps"],
        "runtime_s": check["runtime_s"], "device": check["device"], "date": check["date"],
    }
    (OUT / "fidelity.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保真（测试集 308 例，与存档预测）：{fidelity:.2e}")

    # ---- 参数记录 ----
    params = json.loads(ec.PARAMS_PATH.read_text(encoding="utf-8"))
    rec = dict(check)
    rec.update({
        "self_importance": True, "seed": 0,
        "explain_set": {**check["explain_set"], "split": "development + external_test（先开发集后测试集）"},
        "fidelity_max_abs_diff": fidelity,
        "top5_rule": TOP5_RULE,
        "code": {"module": "dannv2_beta/dannv2_weights.py", "module_sha256": check["weights_module_sha256"],
                 "script": "bishan_native_weights.py", "script_sha256": ec.sha256_file(HERE / "bishan_native_weights.py"),
                 "summary_script": "make_dannv2_weights_summary.py",
                 "model_file_unchanged": "dannv2_beta_ml.py 未改（进学习器指纹）"},
        "outputs": ["native_weights/native_weight_table.csv", "native_weights/native_segment_table.csv",
                    "native_weights/native_weights_percase.npz", "native_weights/native_weights_check.json",
                    "top5_by_label.csv", "fidelity.json"],
    })
    params["models"]["dannv2_beta"] = rec
    params["explain_set"]["applies_to"] = ("random_forest、tabicl；dannv2_beta 的特征权值用开发集加测试集，"
                                           "见 models.dannv2_beta.explain_set")
    params["aggregation"] = {
        "random_forest, tabicl": {"global": "mean over explain set (signed and abs)",
                                  "questionnaire": "signed mean, positive top-5, normalized by column max"},
        "dannv2_beta": {"global": "weight table: mean among cases having the feature value × E/(E+1)",
                        "questionnaire": "weight table cell, positive top-5 among symptoms, normalized by column max"},
    }
    params["excluded_methods_note"] = ("指外加的归因方法（用于随机森林与 TabICL）。gradient_explainer、divide_by_activation_rate_ranking "
                                       "两项不适用于 DANNV2-beta 的特征权值：那是模型自带的输出，不调用外部解释工具。")
    if "versions_server" in params:
        params["versions_server"]["note"] = ("TabICL 的重建与 KernelSHAP、dannv2_beta 的特征权值在课题组 A100 服务器上执行"
                                             "（dannv2_beta 的版本另见 models.dannv2_beta.versions）；random_forest 在本机")
    params["updated"] = ec.now_beijing()
    ec.PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产出：{OUT / 'top5_by_label.csv'}、{OUT / 'fidelity.json'}；已更新 {ec.PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
