# -*- coding: utf-8 -*-
"""汇总可解释性补实验的执行记录（随机森林、TabICL 按《可解释性补实验方案（定稿）》§6、§9 验收；
DANNV2-beta 的特征权值按《DANNV2 原生权值：定稿方案与最终代码》第 4 节验收）。

  python make_explain_record.py

读 explain/explain_params.json、explain/reference_row.json、各模型的 fidelity / additivity_check / rebuild_check / probe、
top5_by_label.csv，写 explain/可解释性补实验_执行记录.md（验收表 + 三模型每证素正向前 5 并排 + 参数记录）。只读，不算任何新量。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPLAIN = HERE / "explain"
OUT = EXPLAIN / "可解释性补实验_执行记录.md"
MODELS = [("dannv2_beta", "DANNV2-beta", "特征权值（模型自带）"), ("random_forest", "随机森林", "TreeSHAP（interventional，单行参考）"),
          ("tabicl", "TabICL", "KernelSHAP（nsamples=auto，l1_reg=False，单行参考）")]
# 定稿方案第 4 节的验收数字（DANNV2-beta 特征权值；跑完后逐项对照）
D_EXPECT_CELLS = {"腰酸→肾": (211, 5.06, 179.6, 5.03), "苔腻→湿": (924, 4.37, 822.0, 4.37), "苔黄→热": (539, 4.21, 489.9, 4.20),
                  "舌紫→血瘀": (94, 4.71, 83.9, 4.65), "畏寒(冷)→阳虚": (148, 3.62, 97.9, 3.58), "舌有点刺→热": (6, 3.85, 5.3, 3.24),
                  "苔灰黑→心": (9, 1.59, 0.5, 0.55), "畏寒(冷)→热": (148, -1.75, 26.1, -1.68), "苔黄→阳虚": (539, -1.44, 20.2, -1.37)}
D_EXPECT_SYMPTOM_CELLS = {"ge_1": 207, "ge_2": 38, "le_-1": 101, "le_-1.5": 9}


def dannv2_cells_ok(cells: dict) -> tuple[int, int]:
    """九个核对格与定稿方案第 4 节一致的个数（人数相同；出现时平均、权值到小数点后两位；预计病例数到一位）。"""
    ok = 0
    for k, (n, m, e, v) in D_EXPECT_CELLS.items():
        c = cells.get(k, {})
        ok += bool(c) and c["人数"] == n and abs(c["出现时平均"] - m) < 0.006 and abs(c["预计病例数"] - e) < 0.06 and abs(c["权值"] - v) < 0.006
    return ok, len(D_EXPECT_CELLS)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def jload(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def main() -> int:
    params = jload(EXPLAIN / "explain_params.json")
    ref = jload(EXPLAIN / "reference_row.json")
    recs = {k: jload(EXPLAIN / k / f) for k, f in (("dannv2_beta", "fidelity.json"), ("random_forest", "additivity_check.json"), ("tabicl", "rebuild_check.json"))}
    probe = jload(EXPLAIN / "tabicl" / "probe.json")
    tops = {k: pd.read_csv(EXPLAIN / k / "top5_by_label.csv", encoding="utf-8-sig") for k, _, _ in MODELS if (EXPLAIN / k / "top5_by_label.csv").exists()}
    labels = list(params.get("explain_set", {}) and jload(HERE / "data" / "obesity_syndrome" / "data.json").get("label_names", []))

    md = ["# 可解释性补实验 执行记录", "",
          f"生成时间：{params.get('updated', params.get('date', '—'))}；执行：{params.get('executor', '—')}；方案：随机森林、TabICL 按《可解释性补实验方案（定稿）》，"
          "DANNV2-beta 的特征权值按《DANNV2 原生权值：定稿方案与最终代码》（2026-10-11）。", "",
          "## 1 验收表", "", "| 项 | 标准 | 结果 |", "|---|---|---|"]
    d, r, t = recs["dannv2_beta"], recs["random_forest"], recs["tabicl"]
    dn = params.get("models", {}).get("dannv2_beta", {})
    cells_ok, cells_n = dannv2_cells_ok(dn.get("check_cells", {}))
    md += [
        f"| DANNV2-beta 模型与存档预测 | ≤ 1e-6 | 测试集 308 例上模型概率与存档预测最大差 {fmt(d.get('fidelity_max_abs_diff'))}；算完后预测与算之前逐位相同 {fmt(d.get('predict_proba_identical_after_explain'))} |",
        f"| DANNV2-beta 权值与模型输出对应 | 最大 0.0026、平均 0.00020；判定 0 处不一致 | 参照人对数几率 + 各特征权值之和与模型输出对数几率之差：最大 {d.get('additivity_max_err', float('nan')):.4f}、平均 {d.get('additivity_mean_err', float('nan')):.5f}；"
        f"由权值复算的判定与模型判定不一致 {d.get('decision_mismatch', '—')} / {d.get('decision_total', '—')}；双精度前向与 predict_proba 最大差 {fmt(d.get('double_forward_vs_predict_proba'))} |",
        f"| DANNV2-beta 各段权值加回特征权值 | 1.6e-14 | 最大差 {fmt(d.get('segment_sum_vs_feature_weight_max'))} |",
        f"| DANNV2-beta 权值表 | 358 行、症状 345 个、空格 0 个；症状格 ≥+1、≥+2、≤−1、≤−1.5 为 207、38、101、9；九个核对格到小数点后两位 | {d.get('table_rows', '—')} 行、症状 {d.get('table_symptoms', '—')} 个、空格 {d.get('table_empty_cells', '—')} 个；"
        f"症状格 {'、'.join(str(dn.get('symptom_cells', {}).get(k, '—')) for k in D_EXPECT_SYMPTOM_CELLS)}（{'一致' if dn.get('symptom_cells') == D_EXPECT_SYMPTOM_CELLS else '不一致'}）；九个核对格一致 {cells_ok} / {cells_n} |",
        f"| DANNV2-beta 分段权值表 | 21 行，没有病人碰到的段 0 个 | {d.get('segment_table_rows', '—')} 行，没有病人碰到的段 {d.get('segment_table_untouched', '—')} 个 |",
        f"| 随机森林 保真 / 可加性 | ≤ 1e-6 / < 1e-6 | 保真 {fmt(r.get('fidelity_max_abs_diff'))}；k = {r.get('scale_factor_k', '—')}，可加性误差 {fmt(r.get('additivity_max_err'))}；归因后预测逐位相同 {fmt(r.get('predict_proba_identical_after_explain'))} |",
        f"| TabICL 重建 / 可加性 | 重建 ≤ 1e-5；可加性仅记录（§5.3） | 重建差 {fmt(t.get('rebuild_max_abs_diff'))}；可加性误差 {fmt(t.get('additivity_max_err'))}（各证素 {min(t.get('additivity_max_err_by_label', {}).values(), default=0):.1e}–{max(t.get('additivity_max_err_by_label', {}).values(), default=0):.1e}）；精确枚举例数 {t.get('exact_enumeration_cases(M<=11)', '—')}/308；归因后预测逐位相同 {fmt(t.get('predict_proba_identical_after_explain'))} |",
        *([f"| TabICL 可加性误差来源 | 诊断（见 additivity_diagnostic.json） | {diag.get('conclusion', '—')} |"] if (diag := jload(EXPLAIN / 'tabicl' / 'additivity_diagnostic.json')) else []),
        f"| explain_params.json | §6 字段填实 | 模型记录：{', '.join(params.get('models', {}).keys()) or '—'} |",
        f"| 耗时 | — | DANNV2-beta {fmt(d.get('runtime_s'))} s（{d.get('steps', '—')} 步，{d.get('device', '—')}，A100 服务器）；随机森林 {fmt(r.get('runtime_s'))} s；TabICL {fmt((t.get('runtime_s') or 0) / 3600, 3)} 小时 |",
        "", "## 2 三模型每证素正向前 5（各模型自己的量，归一化到本列最大值；只在 345 个症状里排）", ""]
    if tops:
        md += ["| 证素 | " + " | ".join(disp for k, disp, _ in MODELS if k in tops) + " |", "|---|" + "---|" * len(tops)]
        for label in labels:
            cells = []
            for k, _, _ in MODELS:
                if k not in tops:
                    continue
                g = tops[k][tops[k]["label"] == label].sort_values("rank")
                if k == "dannv2_beta":
                    cells.append("；".join(f"{row.feature.split('：')[-1]} {row.weight:+.2f}（{row.normalized:.2f}，n={row.n_present}）" for row in g.itertuples()))
                else:
                    cells.append("；".join(f"{row.feature.split('：')[-1]} {row.signed_mean:.3f}（{row.normalized:.2f}，n={row.n_positive_test}）" for row in g.itertuples()))
            md.append(f"| {label} | " + " | ".join(cells) + " |")
    md += ["", "说明：DANNV2-beta 一列是模型自带的特征权值（权值表格值，单位对数几率；解释集为开发集加测试集 1586 例，括号内为归一化值与出现该症状的人数）；"
           "另两模型是 SHAP 值在 308 例测试集上的带符号均值（概率尺度，括号内为归一化值与该症状在测试集的阳性人数）。三者单位不同，只在同一模型内比大小。", "",
           "## 3 公共设置", "",
           f"- 解释集：随机森林、TabICL 为外部测试集 {params.get('explain_set', {}).get('n', '—')} 例（case_ids SHA-256 {params.get('explain_set', {}).get('case_ids_sha256', '—')[:16]}…）；"
           f"DANNV2-beta 的特征权值为开发集加测试集 {dn.get('explain_set', {}).get('n', '—')} 例（先开发集 {dn.get('explain_set', {}).get('n_dev', '—')} 例、后测试集 {dn.get('explain_set', {}).get('n_test', '—')} 例）；数据 {params.get('explain_set', {}).get('data_file', '—')}。",
           f"- 参考样本：{ref.get('rule', '—')}；取值 {ref.get('values', '—')}。",
           f"- 版本：{params.get('versions', {})}。",
           f"- 排除的方法：{params.get('excluded_methods', [])}。{params.get('excluded_methods_note', '')}", "",
           "## 4 TabICL 探针（§5.2）", ""]
    if probe:
        md += [f"- 重建一致性差 {fmt(probe.get('rebuild_max_abs_diff'))}；kv_cache 参数：{probe.get('kv_cache_param') or '该版本适配器没有此参数（不开）'}；"
               f"kv_cache 开/关预测差：{fmt(probe.get('kv_cache_max_abs_diff'))}。",
               f"- 一次调用耗时：50 行 {probe.get('t_query_50_s', '—')} s、2,000 行 {probe.get('t_query_2000_s', '—')} s；整模型 308 例 {probe.get('predict_full_308_s', '—')} s。",
               f"- 单例 KernelSHAP：中位数例（M={probe.get('kernelshap_one_case_median', {}).get('M', '—')}）{probe.get('kernelshap_one_case_median', {}).get('seconds', '—')} s，"
               f"最多例（M={probe.get('kernelshap_one_case_max', {}).get('M', '—')}）{probe.get('kernelshap_one_case_max', {}).get('seconds', '—')} s；估计全量 {probe.get('estimated_total_hours_at_median', '—')} 小时。",
               "- 记忆压扁证据（各子模型，开发集前 100 行 vs 测试集，准确率 / AUROC）：", "", "| 证素 | dev100 | test |", "|---|---|---|"]
        for row in probe.get("memorization", []):
            md.append(f"| {row['label']} | {row['dev100_acc']:.3f} / {fmt(row.get('dev100_auroc'))} | {row['test_acc']:.3f} / {fmt(row.get('test_auroc'))} |")
    md += ["", "## 5 参数记录（explain_params.json 原文）", "", "```json", json.dumps(params, ensure_ascii=False, indent=2), "```", ""]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"产出：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
