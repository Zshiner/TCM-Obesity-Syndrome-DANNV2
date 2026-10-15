"""把框架 `report` 的主结果（`summary/main/`，框架 eb9d56d 起；批次 G `6d8d4d6` 起含数据描述、共现、合并种子区间、
成对差值、部署开销）排成论文用的表，并写说明与核对报告。第二版（2026-10-08）在 skill 原型上新增：
表 0 数据描述与开发集证素共现、表 2 带合并种子 bootstrap 95% 区间、表 5 学习器两两成对差值、计算开销加部署侧两口径、
可解释性三模型并排表（`--explain`）。第三版（2026-10-13）加通用大模型：大模型零样本、不做阈值选择，依赖阈值的指标
取模型自己的判定（框架 `report` 已按此出主表，本脚本按行识别并在表注里写明）；另出 `大模型调用统计.csv`（从结果目录的
`llm_cache/` 读每次调用的耗时与 token，不重算任何指标）；计算开销表里大模型的部署侧不测、注明原因。
第四版（2026-10-14，框架批次 J 的 report 产物）按《模型比较统计检验规范》出成对检验：表 5 改为主指标的成对检验矩阵
（Holm 校正后有统计学意义的加 *），另出 `成对检验_Holm.csv`（全部报告指标 × 全部两两，列按规范第三节）；`--reference`
给出课题模型时，表 2 与它差异无统计学意义的格加 †；核对 P 与 Holm（由存档的结果矩阵独立复算、逐位相同）、族大小、
重抽次数与种子；主结果只含能比较的指标（不含 ECE、MCE、Coverage error）。批次 J 之前的产物（成对差值表没有 p_holm）拒绝。

框架已经把主口径的全部数算好；本脚本只做呈现：学习器与指标的论文用名、"均值 ± 标准差 [区间]"文字、各标签的宽表、
逐种子的宽表、开销的汇总，并把"怎么重跑"与人工写的项目说明并进 `说明_先读我.md`。不导入框架、不重算指标；
每张表都能从 `main/` 复算，核对报告记下结果。

用法：
  python build_paper_tables.py --summary <结果目录>/summary --out <包目录> [--label-term 证素]
        [--title 标题] [--names 名称表.json] [--notes 项目说明.md] [--rerun-notes 怎么重跑.md]
        [--data data.json] [--explain <运行目录>/explain] [--main-metric macro_f1]
        [--llm-cache <结果目录>/llm_cache] [--reference 课题模型的学习器名] [--clean]
  --reference：课题模型（框架里的学习器名，由用户指定，脚本不自行挑选）；给了就在表 2 里把与它差异无统计学意义的格标 †。
  --names 的 JSON：{"learners": {框架名: 论文用名}, "metrics": {框架名: 论文用名}, "labels": {标签: 论文用名}}
  --data：给了就用数据文件逐项复核数据描述表（例数、各标签阳性数、缺失数、零标签例数）。
  --explain：可解释性补实验的产出目录（random_forest/、tabicl/ 各有 mean_signed.csv、top5_by_label.csv；
            dannv2_beta/ 有模型自带特征权值的 native_weights/ 与 top5_by_label.csv、fidelity.json）。
  --llm-cache：结果目录的 llm_cache/。给了就出 `大模型调用统计.csv`（每个不同输入一条：耗时、token、回答形态；重试用尽与
            从说明文字取出 JSON 的预测数读各分片的降级记录；框架 2c0fe44（批次 I）起分片另记回答的逐例形态异常，
            有记录的类别各加一列“……的预测数（全部分片合计）”），并核对各大模型的请求全文逐字节相同（只换模型）、
            缓存条目数 ≤ 开发集 + 测试集例数（要 --data）。
输出目录里已有 主结果/ 时脚本会停下；确认是上一次生成的内容后加 --clean。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LEARNER_NAMES = {
    "logistic_regression": "Logistic Regression", "naive_bayes": "Naive Bayes", "knn": "KNN", "svm": "SVM",
    "decision_tree": "Decision Tree", "random_forest": "Random Forest", "extra_trees": "Extra Trees",
    "adaboost": "AdaBoost", "gradient_boosting": "GBDT", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "realmlp": "RealMLP", "tabpfn_v2": "TabPFN v2", "tabpfn_v3": "TabPFN v3", "tabicl": "TabICL",
    "tabdpt": "TabDPT", "tabnet": "TabNet", "resnet": "ResNet", "ft_transformer": "FT-Transformer",
    "tabm": "TabM", "dann": "DANN", "dannv2_beta": "DANNV2-beta", "zy_bert": "ZY-BERT", "tcm_bert": "TCM-BERT",
    # 内置大模型适配器的论文用名，对应 2026-10 的缺省接口模型（deepseek-v4-pro、glm-5.3、qwen3.8-max-0902）；
    # 适配器换了接口模型就用 --names 改（大模型调用统计表的“接口模型名”列是实际调用的名字，可对照）
    "deepseek": "DeepSeek-V4-Pro", "glm": "GLM-5.3", "qwen": "Qwen3.8-Max",
}
METRIC_NAMES = {
    "macro_f1": "Macro-F1", "micro_f1": "Micro-F1", "weighted_f1": "Weighted-F1", "sample_f1": "Sample-F1",
    "macro_precision": "Macro-Precision", "macro_recall": "Macro-Recall", "subset_accuracy": "Subset accuracy",
    "hamming_loss": "Hamming loss", "macro_auroc": "Macro-AUROC", "micro_auroc": "Micro-AUROC",
    "macro_auprc": "Macro-AUPRC", "micro_auprc": "Micro-AUPRC", "label_ranking_average_precision": "LRAP",
    "label_ranking_loss": "Ranking loss", "coverage_error": "Coverage error", "brier_score": "Brier score",
    "expected_calibration_error": "ECE", "maximum_calibration_error": "MCE", "accuracy": "Accuracy",
}
PER_LABEL_NAMES = {
    "f1": "F1", "precision": "精确率", "recall": "召回率", "specificity": "特异度", "auroc": "AUROC",
    "auprc": "AUPRC", "brier": "Brier 分数", "ece": "ECE",
}
PER_LABEL_ORDER = ["f1", "precision", "recall", "specificity", "auroc", "auprc", "brier"]  # 逐标签 ECE 不进报告（批次 J）
# 不能做重抽样区间与成对检验、不进报告的内置指标（与框架注册表 bootstrap_safe=False 一致；用户 2026-10-14）
NOT_REPORTABLE = ("expected_calibration_error", "maximum_calibration_error", "coverage_error")
# 《模型比较统计检验规范》第四节的统计学方法句与表注
METHOD_SENTENCE = ("外部测试集上的模型比较采用患者级成对 bootstrap 检验，重抽样 {n_boot} 次，两模型使用同一组重抽样本，"
                   "按差值落在 0 两侧的比例计算双侧 P 值，每项指标的全部两两比较按 Holm 法校正，校正后 P<0.05 视为差异有统计学意义。")
CI_TEST_NOTE = "差值的 95% 置信区间未作多重比较校正，差异有无统计学意义以 Holm 校正后的 P 值判断。"
ALPHA = 0.05
THRESHOLD_TEXT = {"raw": "0.5", "at_tau_nested": "嵌套阈值", "at_tau_per_label": "部署阈值", "at_tau_micro": "部署阈值（micro）"}
DESC_ITEMS = [("n_rows", "例数"), ("n_features", "特征数"), ("n_labels", "证素数"), ("labels_per_case_mean", "每例证素数均值"),
              ("labels_per_case_median", "每例证素数中位数"), ("n_zero_label", "无任何证素的例数")]
EXPLAIN_MODELS = [("dannv2_beta", "特征权值（模型自带）"), ("random_forest", "TreeSHAP（interventional，单行参考）"),
                  ("tabicl", "KernelSHAP（nsamples=auto，l1_reg=False，单行参考）")]
WEIGHT_TABLE_MODELS = ("dannv2_beta",)   # 自带特征权值表的模型：前 5 由权值表取，不是解释集上的 SHAP 均值
NON_SYMPTOM = ("年龄", "性别", "病程")
MISSING = -1
BJ = timezone(timedelta(hours=8))
# 框架批次 I 起，分片降级记录里大模型回答的逐例形态异常（消息里写 n_cases=）：类别 → 大模型调用统计表的列名
LLM_ANOMALY_COLS = {
    "predict_response_unknown_label": "回答含候选外{term}名的预测数（全部分片合计）",
    "predict_response_duplicate_label": "同一{term}重复出现的预测数（全部分片合计）",
    "predict_response_length_mismatch": "out与prob长度不同的预测数（全部分片合计）",
    "predict_prob_non_numeric": "置信度不是数的预测数（全部分片合计）",
    "predict_prob_clamped": "置信度越界被截断的预测数（全部分片合计）",
    "predict_no_label_selected": "没有有效候选{term}的预测数（全部分片合计）",
}


def now() -> str:
    return datetime.now(BJ).isoformat(timespec="seconds")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", encoding="utf-8-sig")


def fmt(mean: float, sd: float) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(sd):
        return f"{mean:.3f}"
    if sd == 0:
        return f"{mean:.3f} ± 0"
    return f"{mean:.3f} ± " + ("<0.001" if sd < 0.0005 else f"{sd:.3f}")


def fmt_ci(lo: float, hi: float) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f" [{lo:.3f}, {hi:.3f}]"


def fnum(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def truthy(v) -> bool:
    """CSV 里读回的布尔列（True / False / 空）。"""
    return str(v).strip().lower() in ("true", "1", "1.0")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main_kinds(frame: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """每个指标的主口径（非 raw 学习器的口径；都 raw 时为 raw）与用模型自己判定的学习器。

    框架 `report`（批次 H 起）对大模型一律取 raw（模型回答里选中的标签记阳性），其余学习器取嵌套 / 部署阈值。
    旧写法 `{m: k for m, k in zip(...)}` 取到的是最后一行的口径，混排时会错。
    """
    kinds: dict[str, str] = {}
    for m, g in frame.groupby("metric", sort=False):
        nonraw = g.loc[g["threshold"] != "raw", "threshold"]
        kinds[m] = str(nonraw.mode().iloc[0]) if len(nonraw) else "raw"
    own = list(dict.fromkeys(
        r.learner for r in frame.itertuples() if kinds.get(r.metric, "raw") != "raw" and r.threshold == "raw"
    ))
    return kinds, own


def own_note(own: list[str], lname, term: str, split: str) -> str:
    if not own:
        return ""
    names = "、".join(lname(x) for x in own)
    text = (f"通用大模型（{names}）零样本、不做阈值选择：依赖阈值的指标取模型自己的判定（回答里选中的{term}记阳性）；"
            f"概率是模型对选中{term}给出的置信度，没选中的记 0。")
    if split == "test":
        text += "大模型每例只调用一次、各种子共用同一回答，外部测试集上各种子的结果相同，“±”为 0。"
    elif split == "cv":
        text += "大模型每例只调用一次、各种子共用同一回答，交叉验证的“±”只反映各种子分折的不同。"
    return text


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int, str, str]] = []

    def add(self, what: str, n: int, ok: bool, note: str = "") -> None:
        self.rows.append((what, n, "一致" if ok else "**不一致**", note))

    @property
    def ok(self) -> bool:
        return all(r[2] == "一致" for r in self.rows)


def main() -> int:
    # 中文 Windows 控制台缺省 GBK，打印核对报告里的“⇔”等字符会报 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True, help="框架 report 的汇总目录（含 main/、archive/、summary.md）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label-term", default="标签")
    ap.add_argument("--title", default=None)
    ap.add_argument("--names", type=Path, default=None)
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--rerun-notes", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=None, help="数据文件（data.json），用于复核数据描述表")
    ap.add_argument("--explain", type=Path, default=None, help="可解释性补实验产出目录（explain/）")
    ap.add_argument("--main-metric", default="macro_f1", help="成对差值矩阵用的主指标（缺省 macro_f1）")
    ap.add_argument("--llm-cache", type=Path, default=None, help="结果目录的 llm_cache/（大模型调用统计）")
    ap.add_argument("--reference", default=None, help="课题模型的学习器名（表 2 的 † 以它为参照；由用户指定）")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    main_dir, archive = a.summary / "main", a.summary / "archive"
    if not (main_dir / "cv.csv").exists():
        sys.exit(f"{main_dir} 里没有 cv.csv：需要框架 eb9d56d 起 report 的两层产物")
    names = {"learners": {}, "metrics": {}, "labels": {}}
    if a.names:
        names.update(json.loads(a.names.read_text(encoding="utf-8")))
    L = {**LEARNER_NAMES, **names.get("learners", {})}
    M = {**METRIC_NAMES, **names.get("metrics", {})}
    LB = names.get("labels", {})
    lname = lambda x: L.get(x, x)  # noqa: E731
    mname = lambda x: M.get(x, x)  # noqa: E731
    lbname = lambda x: LB.get(x, x)  # noqa: E731
    term = a.label_term

    out_main = a.out / "主结果"
    if out_main.exists():
        if not a.clean:
            sys.exit(f"{out_main} 已存在；确认是上一次生成的内容后加 --clean")
        shutil.rmtree(out_main)
    out_main.mkdir(parents=True)
    (out_main / "作图数据").mkdir()
    chk = Checks()
    has_tables: list[str] = []

    cv, test = read(main_dir / "cv.csv"), read(main_dir / "test.csv")
    cv_long, test_long = read(main_dir / "cv_long.csv"), read(main_dir / "test_long.csv")
    learners = list(dict.fromkeys(cv["learner"]))  # 定义里的顺序
    metrics = [m for m in M if m in set(cv["metric"])] + [m for m in dict.fromkeys(cv["metric"]) if m not in M]
    summary_text = (a.summary / "summary.md").read_text(encoding="utf-8")
    rule = summary_text.split("## 主口径", 1)[1].split("## ", 1)[0].strip() if "## 主口径" in summary_text else ""
    has_ci = {"ci_lower_pooled", "ci_upper_pooled"}.issubset(test.columns)
    n_boot_text = ""
    if has_ci and "n_boot" in test.columns and test["n_boot"].notna().any():
        n_boot_text = f"重抽 {int(test['n_boot'].dropna().iloc[0])} 次、公共索引矩阵种子 {int(test['bootstrap_seed'].dropna().iloc[0])}"

    # 主结果只含能比较的指标（框架批次 J 起由 report 保证；这里再核一遍）
    shown = list(dict.fromkeys([*cv["metric"], *test["metric"]]))
    bad = [m for m in shown if m in NOT_REPORTABLE]
    chk.add("主结果只含能比较的指标（不含 ECE、MCE、Coverage error）", len(shown), not bad, "、".join(bad) or "")

    # 成对检验（框架批次 J 起，test_paired_differences.csv 带 p_value / p_holm / significant / family_size）
    pd_path = main_dir / "test_paired_differences.csv"
    pdiff = read(pd_path) if pd_path.exists() else pd.DataFrame()
    if not pdiff.empty and "p_holm" not in pdiff.columns:
        sys.exit(f"{pd_path} 没有 p_holm 列：这是框架批次 J 之前的 report 产物，没有成对检验；先用新框架重出 report")
    test_kind = main_kinds(test)[0] if not test.empty else {}
    pm = (pdiff[[test_kind.get(m) == k for m, k in zip(pdiff["metric"], pdiff["threshold"])]].copy()
          if not pdiff.empty else pdiff)
    ref = a.reference
    if ref and ref not in learners:
        sys.exit(f"--reference {ref} 不在主表的学习器里：{'、'.join(learners)}")

    def vs_ref(lr: str, m: str):
        """与课题模型比较的那一行；没有（或就是课题模型本身）返回 None。"""
        if pm.empty or not ref or lr == ref:
            return None
        hit = pm[(pm["metric"] == m) & (((pm["learner_a"] == ref) & (pm["learner_b"] == lr))
                                        | ((pm["learner_a"] == lr) & (pm["learner_b"] == ref)))]
        return None if hit.empty else hit.iloc[0]

    # ---- 表 0：数据描述、开发集证素共现 ----
    desc_path = main_dir / "data_description.csv"
    labels_desc: list[str] = []
    if desc_path.exists():
        desc = read(desc_path)
        splits = [s for s in ("dev", "test") if s in set(desc["split"])]
        split_text = {"dev": "开发集", "test": "外部测试集"}
        glob = desc[desc["label"].isna()].pivot(index="item", columns="split", values="value")
        lab = desc[desc["label"].notna()]
        labels_desc = list(dict.fromkeys(lab[lab["item"] == "label_positives"]["label"]))
        miss_feats = list(dict.fromkeys(lab[lab["item"] == "missing"]["label"]))
        rows = []
        for key, zh in DESC_ITEMS:
            if key in glob.index:
                vals = [glob.loc[key, s] for s in splits]
                rows.append([zh, *[(f"{v:.0f}" if key.startswith("n_") or key.endswith("median") else f"{v:.2f}") for v in vals]])
        for f in miss_feats:
            vals = [lab[(lab["item"] == "missing") & (lab["label"] == f) & (lab["split"] == s)]["value"] for s in splits]
            rows.append([f"{f} 缺失例数", *[f"{float(v.iloc[0]):.0f}" if len(v) else "—" for v in vals]])
        text = [f"# 表 0　数据描述", "", "数据文件把样本固定分成开发集与外部测试集；下表由框架 `report` 的 `data_description.csv` 排版而来。"
                "缺失指原始数据里记为 −1 的取值（框架按定义文件的声明识别）。", "",
                md_table(["项目", *[split_text[s] for s in splits]], rows), "", f"## 各{term}的阳性例数（阳性率）", ""]
        rows2 = []
        for x in labels_desc:
            cells = []
            for s in splits:
                pos = lab[(lab["item"] == "label_positives") & (lab["label"] == x) & (lab["split"] == s)]["value"]
                rate = lab[(lab["item"] == "label_positive_rate") & (lab["label"] == x) & (lab["split"] == s)]["value"]
                cells.append(f"{float(pos.iloc[0]):.0f}（{100 * float(rate.iloc[0]):.1f}%）" if len(pos) and len(rate) else "—")
            rows2.append([lbname(x), *cells])
        text += [md_table([term, *[split_text[s] for s in splits]], rows2), ""]
        (out_main / "表0_数据描述.md").write_text("\n".join(text), encoding="utf-8")
        d2 = desc.rename(columns={"split": "划分", "item": "条目", "label": f"{term}或特征", "value": "取值"})
        d2["划分"] = d2["划分"].map(split_text).fillna(d2["划分"])
        d2.drop(columns=["dataset"]).to_csv(out_main / "表0_数据描述.csv", index=False, encoding="utf-8-sig")
        has_tables.append("表0_数据描述")
        # 核对：阳性率 = 阳性数 / 例数；每例证素数均值 = 阳性数之和 / 例数
        worst = 0.0
        n_items = 0
        for s in splits:
            n = float(glob.loc["n_rows", s])
            pos = lab[(lab["item"] == "label_positives") & (lab["split"] == s)].set_index("label")["value"]
            rate = lab[(lab["item"] == "label_positive_rate") & (lab["split"] == s)].set_index("label")["value"]
            worst = max(worst, float((pos / n - rate.reindex(pos.index)).abs().max()))
            n_items += len(pos)
            if "labels_per_case_mean" in glob.index:
                worst = max(worst, abs(float(pos.sum()) / n - float(glob.loc["labels_per_case_mean", s])))
                n_items += 1
        chk.add("表 0：阳性率 = 阳性数 / 例数；每例证素数均值 = 阳性数之和 / 例数", n_items, worst <= 1e-12, f"最大差 {worst:.1e}")
        if a.data and a.data.exists():
            raw = json.loads(a.data.read_text(encoding="utf-8"))
            fn, ln = raw["feature_names"], raw["label_names"]
            diffs, n_items = 0, 0
            for s, key in (("dev", "train_samples"), ("test", "test_samples")):
                if s not in splits:
                    continue
                X = np.array([r["features"] for r in raw[key]], dtype=float)
                Y = np.array([r["labels"] for r in raw[key]], dtype=int)
                want = {("n_rows", ""): len(X), ("n_features", ""): X.shape[1], ("n_labels", ""): Y.shape[1],
                        ("n_zero_label", ""): int((Y.sum(1) == 0).sum()), ("labels_per_case_mean", ""): float(Y.sum(1).mean()),
                        ("labels_per_case_median", ""): float(np.median(Y.sum(1)))}
                for j, x in enumerate(ln):
                    want[("label_positives", x)] = int(Y[:, j].sum())
                for f in miss_feats:
                    want[("missing", f)] = int((X[:, fn.index(f)] == MISSING).sum())
                for (item, lbl), v in want.items():
                    got = desc[(desc["split"] == s) & (desc["item"] == item) & ((desc["label"] == lbl) if lbl else desc["label"].isna())]["value"]
                    n_items += 1
                    if len(got) != 1 or abs(float(got.iloc[0]) - float(v)) > 1e-9:
                        diffs += 1
            chk.add("表 0：例数、特征数、证素数、各证素阳性数、缺失数、零证素例数、每例证素数与数据文件逐项相同", n_items, diffs == 0, f"不同 {diffs} 项")
    co_path, co_long_path = main_dir / "label_cooccurrence_matrix.csv", main_dir / "label_cooccurrence.csv"
    if co_path.exists():
        co = read(co_path)
        lab_cols = [c for c in co.columns if c not in ("dataset", "label")]
        mat = co.set_index("label")[lab_cols]
        rows = [[lbname(x), *[f"{int(mat.loc[x, y])}" for y in lab_cols]] for x in mat.index]
        text = [f"# 表 0b　开发集{term}共现（例数）", "",
                f"第 i 行第 j 列是开发集里同时判为{term} i 与{term} j 的例数；对角线是各{term}的阳性例数。"
                "`表0b_证素共现_开发集_长表.csv` 另给 Jaccard 系数与条件比例 P(列 | 行)。外部测试集的共现在存档（`summary/archive/label_cooccurrence_test*.csv`）。", "",
                md_table([term, *[lbname(y) for y in lab_cols]], rows), ""]
        (out_main / f"表0b_{term}共现_开发集.md").write_text("\n".join(text), encoding="utf-8")
        shutil.copyfile(co_path, out_main / f"表0b_{term}共现_开发集_矩阵.csv")
        n_sym = 0
        sym_ok = True
        for x in mat.index:
            for y in lab_cols:
                n_sym += 1
                sym_ok = sym_ok and int(mat.loc[x, y]) == int(mat.loc[y, x])
        chk.add(f"表 0b：{term}共现矩阵对称", n_sym, sym_ok)
        if labels_desc:
            desc = read(desc_path)
            pos = desc[(desc["split"] == "dev") & (desc["item"] == "label_positives")].set_index("label")["value"]
            diag_ok = all(int(mat.loc[x, x]) == int(pos[x]) for x in mat.index if x in pos.index)
            chk.add(f"表 0b：共现矩阵对角线 = 开发集各{term}阳性数", len(mat.index), diag_ok)
        if co_long_path.exists():
            cl = read(co_long_path)
            diffs = sum(int(r.count) != int(mat.loc[r.label_a, r.label_b]) for r in cl.itertuples())
            chk.add(f"表 0b：共现长表的例数 = 矩阵对应格", len(cl), diffs == 0, f"不同 {diffs} 项")
            c2 = cl.drop(columns=["dataset"]).rename(columns={"label_a": f"{term}A", "label_b": f"{term}B", "count": "同时阳性例数",
                                                               "jaccard": "Jaccard", "p_b_given_a": "P(B|A)"})
            c2.to_csv(out_main / f"表0b_{term}共现_开发集_长表.csv", index=False, encoding="utf-8-sig")
        has_tables.append(f"表0b_{term}共现_开发集")

    # ---- 表 1 / 表 2：总体性能 ----
    for split, frame, long, title, fname, intro in (
        ("cv", cv, cv_long, "表 1　总体性能（交叉验证）", "表1_总体性能_交叉验证",
         "嵌套交叉验证：每个种子先对外层折取均值，表中是各种子的均值 ± 标准差。"),
        ("test", test, test_long, "表 2　总体性能（外部测试集）", "表2_总体性能_外部测试集",
         "外部测试集：每个种子用全量训练的模型评估一次，表中是各种子的均值 ± 标准差"
         + ("，方括号内是合并种子的 bootstrap 95% 区间。" if has_ci else "。")),
    ):
        if frame.empty:
            continue
        kinds, own = main_kinds(frame)
        header = ["学习器", *[mname(m) for m in metrics if m in kinds]]
        rows, csv_rows = [], []
        n_dagger = n_na = 0
        for lr in learners:
            sub = frame[frame["learner"] == lr].set_index("metric")
            cells = []
            for m in metrics:
                if m not in kinds:
                    continue
                if m not in sub.index:
                    cells.append("—")
                    continue
                cell = fmt(sub.loc[m, "mean"], sub.loc[m, "sd_seeds"])
                if split == "test" and has_ci:
                    cell += fmt_ci(sub.loc[m, "ci_lower_pooled"], sub.loc[m, "ci_upper_pooled"])
                if split == "test" and ref:
                    r = vs_ref(lr, m)
                    if r is not None and not truthy(r["available"]):
                        cell += "§"
                        n_na += 1
                    elif r is not None and not truthy(r["significant"]):
                        cell += "†"
                        n_dagger += 1
                cells.append(cell)
            rows.append([lname(lr), *cells])
            csv_rows.append([lname(lr), *cells])
        thr_note = "；".join(f"{mname(m)} 取{THRESHOLD_TEXT.get(k, k)}" for m, k in kinds.items() if k != "raw")
        text = [f"# {title}", "", intro, "“±”后面是随机种子之间的标准差，只反映换种子带来的波动（分折方式、调参路径、训练随机性），不是置信区间，不能用来判断两个模型的差别是否显著。学习器按定义文件里的顺序排列，不代表优劣。"]
        if split == "test" and has_ci:
            text.append("“[ , ]”是合并种子的 bootstrap 95% 区间：对外部测试集做患者级重抽样，每次重抽样对 5 个种子各算一遍指标后取均值，再取 2.5% 与 97.5% 百分位"
                        + (f"（{n_boot_text}）" if n_boot_text else "") + "。它是“跨种子均值”这个点估计的测试集抽样区间，条件于冻结的模型与阈值，不含训练与调参的随机性，未作多重比较校正。"
                        "逐种子的区间在 `外部测试集_逐种子置信区间.csv`。")
        if split == "test" and not pm.empty:
            text.append("两个模型有无差异看成对检验（表 5 与 `成对检验_Holm.csv`），不看区间或“±”是否重叠。")
            if ref:
                text.append(f"† 表示经成对 bootstrap 检验（每项指标内全部两两比较按 Holm 法校正），与 {lname(ref)} 的差异无统计学意义；"
                            f"未标 † 的与 {lname(ref)} 差异有统计学意义；{lname(ref)} 本行不标。"
                            + ("§ 表示该比较在重抽样上不可计算（有效重抽次数不足 90%）。" if n_na else ""))
        if thr_note:
            text.append(f"阈值依赖的指标的口径：{thr_note}；其余指标与阈值无关。")
        if own:
            text.append(own_note(own, lname, term, split))
        text += ["", md_table(header, rows), ""]
        (out_main / f"{fname}.md").write_text("\n".join(text), encoding="utf-8")
        pd.DataFrame(csv_rows, columns=header).to_csv(out_main / f"{fname}.csv", index=False, encoding="utf-8-sig")
        has_tables.append(fname)
        # 核对：逐种子的均值回算与 main 表一致
        if split == "cv":
            per_seed = long.groupby(["learner", "metric", "seed"], sort=False)["value"].mean().reset_index()
        else:
            per_seed = long[["learner", "metric", "seed", "value"]]
        agg = per_seed.groupby(["learner", "metric"], sort=False)["value"].agg(["mean", "std", "count"]).reset_index()
        merged = frame.merge(agg, on=["learner", "metric"], how="left")
        worst = float((merged["mean_x"] - merged["mean_y"]).abs().max()) if len(merged) else 0.0
        worst_sd = float((merged["sd_seeds"] - merged["std"]).abs().max()) if len(merged) else 0.0
        chk.add(f"{title}：由逐种子长表回算的均值与标准差", len(merged), max(worst, worst_sd) <= 1e-12, f"最大差 {max(worst, worst_sd):.1e}")
        if split == "test" and has_ci:
            n_missing = int(frame["ci_lower_pooled"].isna().sum() + frame["ci_upper_pooled"].isna().sum())
            order_ok = bool((frame["ci_lower_pooled"] <= frame["ci_upper_pooled"]).all())
            outside = int(((frame["mean"] < frame["ci_lower_pooled"] - 1e-12) | (frame["mean"] > frame["ci_upper_pooled"] + 1e-12)).sum())
            chk.add("表 2：每个 学习器 × 指标 都有合并种子区间，且下限 ≤ 上限", len(frame), n_missing == 0 and order_ok,
                    f"缺 {n_missing} 项；跨种子均值落在区间外的有 {outside} 项（百分位区间不保证包含点估计，仅记录）")

    # ---- 长表（逐种子的宽表）----
    def wide(long: pd.DataFrame, keys: list[str], split_text: str) -> pd.DataFrame:
        if long.empty:
            return pd.DataFrame()
        if "fold" in long.columns and split_text == "交叉验证":
            long = long.groupby([*keys, "seed"], sort=False)["value"].mean().reset_index()
        piv = long.pivot_table(index=keys, columns="seed", values="value", aggfunc="first")
        piv.columns = [f"种子{int(s)}" for s in piv.columns]
        piv = piv.reset_index()
        stats = long.groupby(keys, sort=False)["value"].agg(均值="mean", 标准差="std", 种子数="count").reset_index()
        return stats.merge(piv, on=keys, how="left").assign(划分=split_text)

    overall = []
    for split_text, long in (("交叉验证", cv_long), ("外部测试集", test_long)):
        w = wide(long, ["learner", "metric", "threshold"], split_text)
        if not w.empty:
            w.insert(1, "论文用名", w["learner"].map(lname))
            w.insert(3, "指标论文用名", w["metric"].map(mname))
            w["阈值口径"] = w["threshold"].map(lambda k: THRESHOLD_TEXT.get(k, k))
            if split_text == "外部测试集" and has_ci:
                w = w.merge(test[["learner", "metric", "threshold", "ci_lower_pooled", "ci_upper_pooled"]], on=["learner", "metric", "threshold"], how="left")
                w = w.rename(columns={"ci_lower_pooled": "合并种子95%区间下限", "ci_upper_pooled": "合并种子95%区间上限"})
            overall.append(w)
    if overall:
        o = pd.concat(overall, ignore_index=True)
        seed_cols = [c for c in o.columns if c.startswith("种子") and c != "种子数"]
        ci_cols = [c for c in ("合并种子95%区间下限", "合并种子95%区间上限") if c in o.columns]
        o = o[["划分", "learner", "论文用名", "metric", "指标论文用名", "阈值口径", "均值", "标准差", "种子数", *ci_cols, *seed_cols]]
        o = o.rename(columns={"learner": "学习器", "metric": "指标"})
        o.to_csv(out_main / "总体性能_长表.csv", index=False, encoding="utf-8-sig")
    if {"ci_lower", "ci_upper"}.issubset(test_long.columns):
        t = test_long.copy()
        t.insert(2, "论文用名", t["learner"].map(lname))
        t.insert(5, "指标论文用名", t["metric"].map(mname))
        t["阈值口径"] = t["threshold"].map(lambda k: THRESHOLD_TEXT.get(k, k))
        t = t.rename(columns={"learner": "学习器", "seed": "种子", "metric": "指标", "value": "取值", "ci_lower": "95%区间下限", "ci_upper": "95%区间上限"})
        t = t[["学习器", "论文用名", "种子", "指标", "指标论文用名", "阈值口径", "取值", "95%区间下限", "95%区间上限"]]
        t.to_csv(out_main / "外部测试集_逐种子置信区间.csv", index=False, encoding="utf-8-sig")
        chk.add("外部测试集逐种子区间：每行都有下限与上限", len(t), bool(t["95%区间下限"].notna().all() and t["95%区间上限"].notna().all()))

    # ---- 表 3 / 表 4：各标签 ----
    pl_long_all = read(archive / "per_label_long.csv") if (archive / "per_label_long.csv").exists() else pd.DataFrame()
    pl_cv, pl_test = read(main_dir / "per_label_cv.csv"), read(main_dir / "per_label_test.csv")
    labels = list(dict.fromkeys(pl_test["label"])) if not pl_test.empty else list(dict.fromkeys(pl_cv["label"]))
    for frame, split, title, fname, perf_frame in (
        (pl_test, "test", f"表 3　各{term}的结果（外部测试集）", f"表3_各{term}_外部测试集", test),
        (pl_cv, "cv", f"表 4　各{term}的结果（交叉验证）", f"表4_各{term}_交叉验证", cv),
    ):
        if frame.empty:
            continue
        _, own_pl = main_kinds(perf_frame)
        nonown = frame[~frame["learner"].isin(own_pl)]
        kind = (nonown if len(nonown) else frame)["threshold"].iloc[0]
        text = [f"# {title}", "",
                f"口径：{THRESHOLD_TEXT.get(kind, kind)}（与总体性能表相同）；跨种子的均值 ± 标准差。AUROC、AUPRC、Brier 分数与阈值无关。"
                f"各{term} F1 的平均就是总体性能表里的 Macro-F1（见核对报告）。"
                + (own_note(own_pl, lname, term, split) if own_pl else ""), ""]
        for lr in learners:
            sub = frame[frame["learner"] == lr]
            if sub.empty:
                continue
            text += [f"## {lname(lr)}", ""]
            header = [term, *[lbname(x) for x in labels]]
            rows = []
            for m in PER_LABEL_ORDER:
                part = sub[sub["metric"] == m].set_index("label")
                rows.append([PER_LABEL_NAMES.get(m, m), *[fmt(part.loc[x, "mean"], part.loc[x, "sd_seeds"]) if x in part.index else "—" for x in labels]])
            text += [md_table(header, rows), ""]
            f1 = sub[sub["metric"] == "f1"]["mean"].mean()
            macro = perf_frame[(perf_frame["learner"] == lr) & (perf_frame["metric"] == "macro_f1")]["mean"]
            if len(macro) == 1:
                chk.add(f"{title}：{lname(lr)} 各{term} F1 的平均等于 Macro-F1", 1, abs(float(macro.iloc[0]) - float(f1)) <= 1e-9, f"差 {abs(float(macro.iloc[0]) - float(f1)):.1e}")
        (out_main / f"{fname}.md").write_text("\n".join(text), encoding="utf-8")
        has_tables.append(fname)
    if not pl_long_all.empty:
        parts = []
        for split_text, split, kind_frame in (("交叉验证", "cv", pl_cv), ("外部测试集", "test", pl_test)):
            if kind_frame.empty:
                continue
            _, own_pl = main_kinds(cv if split == "cv" else test)
            nonown = kind_frame[~kind_frame["learner"].isin(own_pl)]
            kind = (nonown if len(nonown) else kind_frame)["threshold"].iloc[0]
            sel = pl_long_all["learner"].isin(own_pl)
            # 只取主结果逐标签表里有的指标：逐标签 ECE 不进报告（批次 J），只留在 archive/per_label_long.csv
            long = pl_long_all[(pl_long_all["split"] == split) & pl_long_all["metric"].isin(set(kind_frame["metric"])) & (
                (~sel & (pl_long_all["threshold"] == kind)) | (sel & (pl_long_all["threshold"] == "raw")))]
            w = wide(long, ["learner", "label", "metric"], split_text)
            if not w.empty:
                parts.append(w)
        if parts:
            p = pd.concat(parts, ignore_index=True)
            p.insert(1, "论文用名", p["learner"].map(lname))
            p.insert(4, "指标名", p["metric"].map(lambda m: PER_LABEL_NAMES.get(m, m)))
            seed_cols = [c for c in p.columns if c.startswith("种子") and c != "种子数"]
            p = p[["划分", "learner", "论文用名", "label", "metric", "指标名", "均值", "标准差", "种子数", *seed_cols]]
            p = p.rename(columns={"learner": "学习器", "label": term, "metric": "指标"})
            p.to_csv(out_main / f"各{term}_长表.csv", index=False, encoding="utf-8-sig")

    # ---- 表 5 与 成对检验_Holm.csv：学习器两两的成对检验（外部测试集；《模型比较统计检验规范》）----
    if not pm.empty and not test.empty:
        main_kind, own_t = main_kinds(test)
        mm = a.main_metric if a.main_metric in main_kind else metrics[0]
        sub = pm[pm["metric"] == mm]
        cell = {}
        for r in sub.itertuples():
            if not truthy(r.available):
                cell[(r.learner_a, r.learner_b)] = cell[(r.learner_b, r.learner_a)] = "不可计算"
                continue
            star = "*" if truthy(r.significant) else ""
            cell[(r.learner_a, r.learner_b)] = f"{r.diff:+.3f} [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]{star}"
            cell[(r.learner_b, r.learner_a)] = f"{-r.diff:+.3f} [{-r.ci_upper:+.3f}, {-r.ci_lower:+.3f}]{star}"
        rows = [[lname(x), *["—" if x == y else cell.get((x, y), "—") for y in learners]] for x in learners]
        nb = int(sub["n_boot"].iloc[0]) if len(sub) else int(pm["n_boot"].iloc[0])
        bs = int(sub["bootstrap_seed"].iloc[0]) if len(sub) else int(pm["bootstrap_seed"].iloc[0])
        fam = sorted({int(v) for v in sub["family_size"].dropna()}) if len(sub) else []
        text = [f"# 表 5　学习器两两的成对检验（外部测试集，{mname(mm)}，{THRESHOLD_TEXT.get(main_kind.get(mm), main_kind.get(mm))}）", "",
                "每格是“行学习器 − 列学习器”的跨种子均值之差，方括号内是差值的 95% 置信区间；* 表示差异有统计学意义（Holm 校正后 P<0.05）。",
                METHOD_SENTENCE.format(n_boot=nb)
                + f"同一指标内的 {'、'.join(str(x) for x in fam) or '—'} 对比较为一族；重抽种子 {bs}，所有学习器共用同一批重抽样本；每次重抽样先对各种子算指标再取均值。"
                + CI_TEST_NOTE + "不据此给学习器排序或指定“最终模型”。全部指标的检验结果（含 P 原始与 P_Holm）在 `成对检验_Holm.csv`。"
                + (f"阈值依赖的指标按主口径冻结阈值；涉及通用大模型（{'、'.join(lname(x) for x in own_t)}）的对，大模型一侧取模型自己的判定。" if own_t else ""), "",
                md_table(["行 − 列", *[lname(y) for y in learners]], rows), ""]
        (out_main / "表5_成对检验_外部测试集.md").write_text("\n".join(text), encoding="utf-8")
        order = {m: i for i, m in enumerate(metrics)}
        hrows = []
        for r in sorted(pm.itertuples(), key=lambda r: order.get(r.metric, len(order))):
            hrows.append({
                "指标": mname(r.metric), "阈值口径": r.threshold, "模型A": lname(r.learner_a), "模型B": lname(r.learner_b),
                "差值A减B": r.diff, "95%区间下限": r.ci_lower, "95%区间上限": r.ci_upper, "P原始": r.p_value, "P_Holm": r.p_holm,
                "有统计学意义": ("是" if truthy(r.significant) else "否") if truthy(r.available) else "不可计算",
                "族大小": int(r.family_size), "重抽次数": int(r.n_boot), "重抽种子": int(r.bootstrap_seed),
            })
        pd.DataFrame(hrows).to_csv(out_main / "成对检验_Holm.csv", index=False, encoding="utf-8-sig")
        has_tables.append("表5_成对检验_外部测试集")
        # 核对：差值 = 表 2 两均值之差；下限 ≤ 差值 ≤ 上限；对数 = C(n,2) × 指标数
        tm = test.set_index(["learner", "metric", "threshold"])["mean"]
        worst, n_bad, n_outside, n_order = 0.0, 0, 0, 0
        kk = lambda lr, k: "raw" if lr in own_t else k  # noqa: E731 大模型一侧取 raw（与表 2 一致）
        for r in pm.itertuples():
            try:
                d = (float(tm.loc[(r.learner_a, r.metric, kk(r.learner_a, r.threshold))])
                     - float(tm.loc[(r.learner_b, r.metric, kk(r.learner_b, r.threshold))]))
            except KeyError:
                n_bad += 1
                continue
            worst = max(worst, abs(d - float(r.diff)))
            if float(r.ci_lower) > float(r.ci_upper):
                n_order += 1
            if not (float(r.ci_lower) - 1e-12 <= float(r.diff) <= float(r.ci_upper) + 1e-12):
                n_outside += 1
        n_pairs = len(learners) * (len(learners) - 1) // 2
        chk.add("成对检验：差值 = 表 2 两学习器均值之差（主口径），区间下限 ≤ 上限", len(pm), worst <= 1e-9 and n_bad == 0 and n_order == 0,
                f"最大差 {worst:.1e}；缺项 {n_bad}；点估计落在区间外的有 {n_outside} 对（百分位区间不保证包含点估计，仅记录）")
        chk.add("成对检验：主口径的成对数 = C(学习器数, 2) × 指标数", 1, len(pm) == n_pairs * len(main_kind), f"（{len(pm)} 对，{n_pairs} × {len(main_kind)}）")
        # 族大小 = 可计算的对数；P_Holm ≥ P原始、按 P原始排序后单调；有统计学意义 ⇔ P_Holm < 0.05
        n_fam = n_holm = n_sig = n_na = 0
        for _m, g in pm.groupby(["dataset", "metric"] if "dataset" in pm.columns else ["metric"], sort=False):
            av = g[g["available"].map(truthy)]
            n_na += len(g) - len(av)
            n_fam += int(set(g["family_size"].astype(int)) != {len(av)})
            n_holm += int((av["p_holm"] < av["p_value"]).any())
            srt = av.sort_values("p_value", kind="stable")["p_holm"].to_numpy()
            n_holm += int((np.diff(srt) < 0).any())
            n_sig += int((av["significant"].map(truthy) != (av["p_holm"] < ALPHA)).sum())
        chk.add("成对检验：族大小 = 每项指标内可计算的对数；P_Holm ≥ P原始且按 P原始排序后单调；有统计学意义 ⇔ P_Holm < 0.05",
                len(pm), n_fam == 0 and n_holm == 0 and n_sig == 0,
                f"族大小不符 {n_fam} 项；Holm 不符 {n_holm} 项；判定不符 {n_sig} 行；不可计算 {n_na} 对（不报 P、不计入族）")
        # 重抽次数与种子（规范第二节第 2 条：B = 10000、固定种子 = 协议第一个种子；不为显著改次数或种子）
        proto_seed = None
        lock_path = a.summary.parent / "experiment.lock.json"
        if lock_path.exists():
            seeds_l = (json.loads(lock_path.read_text(encoding="utf-8")).get("protocol") or {}).get("seeds") or []
            proto_seed = int(seeds_l[0]) if seeds_l else None
        nbs, bss = sorted(set(pm["n_boot"].astype(int))), sorted(set(pm["bootstrap_seed"].astype(int)))
        chk.add("成对检验：重抽次数 = 10000，重抽种子 = 协议第一个种子", len(pm),
                nbs == [10000] and proto_seed is not None and bss == [proto_seed],
                f"重抽次数 {nbs}；重抽种子 {bss}；协议第一个种子 {proto_seed if proto_seed is not None else '（读不到锁文件）'}")
        # 独立复算（规范第五节自检）：由存档的结果矩阵按规范公式重算区间、P 与 Holm，与 report 的数逐位相同
        npz_path = archive / "test_bootstrap_matrix.npz"
        if npz_path.exists():
            n_items = n_diff = 0
            with np.load(npz_path) as z:
                conf = float(z["confidence"])
                alpha_ci = 1.0 - conf
                for ds in [str(x) for x in z["datasets"]]:
                    vals, units_z = z[f"{ds}/values"], [str(u) for u in z[f"{ds}/units"]]
                    n_b = vals.shape[-1]
                    part = pdiff[pdiff["dataset"] == ds] if "dataset" in pdiff.columns else pdiff
                    for j, (m, k) in enumerate(zip([str(x) for x in z["metrics"]], [str(x) for x in z["thresholds"]])):
                        pooled = vals[:, :, j, :].mean(axis=1)
                        g = part[(part["metric"] == m) & (part["threshold"] == k)]
                        p_raw, keep = [], []
                        for r in g.itertuples():
                            d = pooled[units_z.index(r.unit_a)] - pooled[units_z.index(r.unit_b)]
                            fin = d[np.isfinite(d)]
                            ok_r = fin.size > 0 and not fin.size < 0.9 * n_b
                            n_items += 1
                            if not ok_r:
                                n_diff += int(truthy(r.available))
                                p_raw.append(float("nan"))
                                keep.append(r)
                                continue
                            lo = float(np.percentile(fin, 100.0 * (alpha_ci / 2.0)))
                            hi = float(np.percentile(fin, 100.0 * (1.0 - alpha_ci / 2.0)))
                            n_f = fin.size
                            p = min(1.0, 2.0 * min((1 + int(np.count_nonzero(fin <= 0))) / (n_f + 1),
                                                   (1 + int(np.count_nonzero(fin >= 0))) / (n_f + 1)))
                            n_diff += int(not (lo == float(r.ci_lower) and hi == float(r.ci_upper) and p == float(r.p_value)))
                            p_raw.append(p)
                            keep.append(r)
                        fin_idx = [i for i, p in enumerate(p_raw) if p == p]
                        holm = [float("nan")] * len(p_raw)
                        running = 0.0
                        for rank, i in enumerate(sorted(fin_idx, key=lambda i: p_raw[i])):
                            running = max(running, min(1.0, (len(fin_idx) - rank) * p_raw[i]))
                            holm[i] = running
                        for r, q in zip(keep, holm):
                            if q == q:
                                n_diff += int(q != float(r.p_holm))
            chk.add("成对检验：由存档的结果矩阵独立复算区间、P原始与 P_Holm，与 report 逐位相同（规范第五节自检）", n_items, n_diff == 0, f"不同 {n_diff} 项")
        else:
            chk.add("成对检验：存档里有结果矩阵 test_bootstrap_matrix.npz（独立复算用）", 1, False, "缺文件")
        # 未校正区间与 P原始在边界上可能差一格（百分位插值 vs 计数），只记录
        av_all = pm[pm["available"].map(truthy)]
        excl0 = (av_all["ci_lower"] > 0) | (av_all["ci_upper"] < 0)
        n_bd = int((excl0 != (av_all["p_value"] < ALPHA)).sum())
        chk.add("成对检验：未校正区间是否含 0 与 P原始 < 0.05 的对应（边界上的不一致只记录）", len(av_all), True,
                f"不一致 {n_bd} 对；判定一律以 P_Holm 为准")

    # ---- 部署阈值、最终模型参数、计算开销 ----
    thr = read(main_dir / "thresholds.csv")
    if not thr.empty:
        rows = []
        for (lr, seed), sub in thr.groupby(["learner", "seed"], sort=False):
            row = {"学习器": lr, "论文用名": lname(lr), "种子": int(seed)}
            micro = sub[sub["kind"] == "micro"]["tau"]
            glob_t = sub[sub["kind"] == "global"]["tau"]
            row["micro 指标共用阈值"] = float(micro.iloc[0]) if len(micro) else float("nan")
            row["补位阈值"] = float(glob_t.iloc[0]) if len(glob_t) else float("nan")
            for x in labels:
                hit = sub[(sub["kind"] == "per_label") & (sub["label"] == x)]["tau"]
                row[lbname(x)] = float(hit.iloc[0]) if len(hit) else float("nan")
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_main / "部署阈值.csv", index=False, encoding="utf-8-sig")
    params = pd.read_csv(main_dir / "final_params.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not params.empty:
        tuned: dict[str, bool] = {}
        if (main_dir / "learners.csv").exists():
            lt = pd.read_csv(main_dir / "learners.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
            name_col = "学习器" if "学习器" in lt.columns else "learner"
            dims_col = "可搜索维度数" if "可搜索维度数" in lt.columns else "n_search_dims"
            if name_col in lt.columns and dims_col in lt.columns:
                tuned = {str(r[name_col]): int(float(r[dims_col] or 0)) > 0 for r in lt.to_dict(orient="records")}
        p = params.rename(columns={"learner": "学习器", "seed": "种子", "param": "参数", "value": "取值"})
        p.insert(1, "论文用名", p["学习器"].map(lname))
        p["来由"] = p["学习器"].map(lambda x: "在全部开发集上调参选出" if tuned.get(x, True) else "固定取值（这个学习器不调参）")
        p[["学习器", "论文用名", "种子", "参数", "取值", "来由"]].to_csv(out_main / "最终模型参数.csv", index=False, encoding="utf-8-sig")

    cost_main = read(main_dir / "compute_cost.csv") if (main_dir / "compute_cost.csv").exists() else pd.DataFrame()
    cost_all = read(archive / "compute_cost_all.csv") if (archive / "compute_cost_all.csv").exists() else pd.DataFrame()
    if cost_all.empty and not cost_main.empty and "fold" in cost_main.columns:
        cost_all = cost_main  # 旧框架：main 表本身就是逐分片
    deploy_modes = [("train_device", "训练设备口径"), ("cpu", "CPU单线程口径")]
    has_deploy = not cost_main.empty and "deploy_source" in cost_main.columns
    deploy_long = read(archive / "deploy_cost_long.csv") if (archive / "deploy_cost_long.csv").exists() else pd.DataFrame()
    _, own_cost = main_kinds(test if not test.empty else cv)

    def deploy_note(lr: str, mode: str) -> tuple[str, str, str]:
        """（实测逐例例数、加载用时口径、说明）：由 archive/deploy_cost_long.csv 的数字组成，不改写框架的取值。"""
        if deploy_long.empty:
            return "", "", ""
        sub = deploy_long[(deploy_long["learner"] == lr) & (deploy_long["mode"] == mode)]
        if sub.empty:
            return "", "", ""
        n = sub["n_single_measured"].dropna().astype(int)
        n_text = "" if n.empty else (f"{n.min()}" if n.min() == n.max() else f"{n.min()}–{n.max()}")
        load = "；".join(sorted({"文件反序列化" if x == "file" else "pickle 往返（补算时没有落盘的模型文件）" if x == "pickle_roundtrip" else str(x)
                                for x in sub["load_source"].dropna().astype(str)}))
        caps = sorted({float(m) for m in sub["note"].dropna().astype(str).str.extract(r"超过时间上限 ([0-9.]+) s")[0].dropna()})
        cap_text = f"{caps[0]:g} 秒" if caps else "时间上限"
        timeouts = sub["timeout_projected"].fillna("").astype(str)
        parts = []
        if (timeouts.str.contains("single") & (sub["n_single_measured"].fillna(0) == 0)).all():
            parts.append(f"单例预测超过 {cap_text}未返回、已中止，逐例与整批均未测得（下界：单例 > {cap_text}）")
        else:
            if timeouts.str.contains("single").any():
                per = sub["single_ms_median"].median() / 1000.0
                parts.append(f"逐例按时间上限只测了 {n_text} 例（每例约 {per:.1f} 秒），不足 {int(sub['n_single'].max())} 例")
            if timeouts.str.contains("batch").any() and sub["batch_s"].isna().all():
                parts.append(f"整批预计 {sub['projected_batch_s'].median():.0f} 秒超过时间上限、未跑（外推值）")
        bad = sub[sub["runnable"].astype(str).str.lower().isin(["false", "0"])]
        if len(bad):
            parts.append("不可运行：" + "；".join(sorted(set(bad["reason"].dropna().astype(str)))))
        return n_text, load, "；".join(parts)
    if not cost_all.empty:
        wall = "shard_wall_s" if "shard_wall_s" in cost_all.columns else "model_wall_clock_s"
        vram = "shard_peak_vram_mb" if "shard_peak_vram_mb" in cost_all.columns else "peak_gpu_mb"
        rows = []
        for lr in learners:
            sub = cost_all[cost_all["learner"] == lr]
            folds = sub[sub["fold"].astype(str) != "final"]
            finals = sub[sub["fold"].astype(str) == "final"]
            row = {
                "学习器": lr, "论文用名": lname(lr), "训练设备": str(sub["device"].iloc[0]) if "device" in sub.columns and len(sub) else "",
                "外层折分片墙钟均值（分）": round(folds[wall].mean() / 60, 2), "全量分片墙钟均值（分）": round(finals[wall].mean() / 60, 2),
                "全部分片墙钟合计（时）": round(sub[wall].sum() / 3600, 2),
                "每个外层折调参用时均值（秒）": round(folds["p1_hpo_search_s"].mean(), 1),
                "全量调参用时均值（秒）": round(finals["p2_hpo_search_s"].mean(), 1),
                "最终模型训练用时均值（秒）": round(finals["p3_train_s"].mean(), 3),
                "模型序列化大小均值（MB）": round(finals["serialized_size_mb"].mean(), 2),
                "训练峰值显存最大值（MB）": float(round(float(sub[vram].fillna(0.0).max()))),
            }
            if lr in own_cost:
                row["说明"] = ("云端接口调用（零样本，不训练、不调参）。每例只问一次：首个种子的分片真正调用接口，墙钟主要是等接口返回；"
                              "其余种子全部命中缓存、不调用接口，外层折分片只要几秒，全量分片的时间几乎都花在外部测试集的 bootstrap 区间上。"
                              "这里的均值是两种分片的平均，不代表调用接口的耗时。"
                              "部署侧开销不测（模型不在本地）；每次调用的耗时与 token 见 `大模型调用统计.csv`")
            if has_deploy:
                dm = cost_main[cost_main["learner"] == lr]
                for mode, zh in deploy_modes:
                    dev = dm[f"{mode}_device"].dropna().astype(str).unique() if f"{mode}_device" in dm.columns else []
                    runnable = dm[f"{mode}_runnable"] if f"{mode}_runnable" in dm.columns else pd.Series(dtype=object)
                    timeout = dm[f"{mode}_timeout_projected"] if f"{mode}_timeout_projected" in dm.columns else pd.Series(dtype=object)
                    reasons = dm[f"{mode}_reason"].dropna().astype(str).unique() if f"{mode}_reason" in dm.columns else []
                    row[f"{zh}_设备"] = "/".join(dev)
                    row[f"{zh}_可运行种子数"] = int(runnable.fillna(False).astype(bool).sum()) if len(runnable) else 0
                    row[f"{zh}_加载用时均值（秒）"] = round(dm[f"{mode}_load_s"].mean(), 3) if f"{mode}_load_s" in dm.columns else float("nan")
                    row[f"{zh}_单例延迟中位数均值（毫秒）"] = round(dm[f"{mode}_single_ms_median"].mean(), 3) if f"{mode}_single_ms_median" in dm.columns else float("nan")
                    row[f"{zh}_单例延迟p95均值（毫秒）"] = round(dm[f"{mode}_single_ms_p95"].mean(), 3) if f"{mode}_single_ms_p95" in dm.columns else float("nan")
                    row[f"{zh}_整批推理用时均值（秒）"] = round(dm[f"{mode}_batch_s"].mean(), 4) if f"{mode}_batch_s" in dm.columns else float("nan")
                    row[f"{zh}_峰值内存最大值（MB）"] = round(dm[f"{mode}_peak_rss_mb"].max(), 1) if f"{mode}_peak_rss_mb" in dm.columns else float("nan")
                    row[f"{zh}_峰值显存最大值（MB）"] = round(dm[f"{mode}_peak_gpu_mb"].max(), 1) if f"{mode}_peak_gpu_mb" in dm.columns else float("nan")
                    row[f"{zh}_超限外推种子数"] = int(timeout.fillna("").astype(str).str.len().gt(0).sum()) if len(timeout) else 0
                    n_text, load_text, note_text = deploy_note(lr, mode)
                    row[f"{zh}_实测逐例例数"] = n_text
                    row[f"{zh}_加载用时口径"] = load_text
                    row[f"{zh}_说明"] = "；".join(x for x in [note_text, *[r for r in reasons if r and r not in note_text]] if x)
                row["整批例数"] = int(dm["n_test"].dropna().iloc[0]) if "n_test" in dm.columns and dm["n_test"].notna().any() else None
                row["逐例例数"] = int(dm["n_single"].dropna().iloc[0]) if "n_single" in dm.columns and dm["n_single"].notna().any() else None
                row["部署侧来源"] = "/".join(sorted(dm["deploy_source"].dropna().astype(str).unique()))
                row["测量机CPU"] = "/".join(dm["cpu_model"].dropna().astype(str).unique()) if "cpu_model" in dm.columns else ""
                row["测量机显卡"] = "/".join(dm["gpu_name"].dropna().astype(str).unique()) if "gpu_name" in dm.columns else ""
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_main / "计算开销.csv", index=False, encoding="utf-8-sig")
        if not cost_main.empty:
            shutil.copyfile(main_dir / "compute_cost.csv", out_main / "计算开销_逐种子.csv")
            if not cost_main.empty and "p3_train_s" in cost_main.columns and "fold" not in cost_main.columns:
                f_main = cost_main.groupby("learner")["p3_train_s"].mean()
                f_all = cost_all[cost_all["fold"].astype(str) == "final"].groupby("learner")["p3_train_s"].mean()
                worst = float((f_main - f_all.reindex(f_main.index)).abs().max())
                chk.add("计算开销：最终模型训练用时（逐种子表）与逐分片表的全量分片一致", len(f_main), worst <= 1e-9, f"最大差 {worst:.1e}")
        if has_deploy:
            n_seeds = int(cv["n_seeds"].max()) if "n_seeds" in cv.columns else cost_main.groupby("learner").size().max()
            local = [x for x in learners if x not in own_cost]
            want = len(local) * n_seeds
            got = int(cost_main[cost_main["learner"].isin(local)]["deploy_source"].notna().sum())
            chk.add("计算开销：部署侧每个 学习器 × 种子 都有记录" + ("（通用大模型是云端接口，不测）" if own_cost else ""), want, got == want,
                    f"有记录 {got}/{want}；来源 {sorted(cost_main['deploy_source'].dropna().astype(str).unique())}")
            n_to = 0
            for mode, _ in deploy_modes:
                if f"{mode}_timeout_projected" in cost_main.columns:
                    n_to += int(cost_main[f"{mode}_timeout_projected"].fillna(False).astype(bool).sum())
            chk.add("计算开销：部署侧超过时间上限、按试跑外推的项数（仅记录）", n_to, True, "ICL 类模型的 CPU 单线程口径允许超限，取值为外推")
    for src, dst in (("decision_curve_test.csv", "决策曲线_外部测试集.csv"), ("calibration_bins_test.csv", "校准曲线_外部测试集.csv"), ("calibration_bins_cv.csv", "校准曲线_交叉验证.csv")):
        if (main_dir / "curves" / src).exists() and (main_dir / "curves" / src).stat().st_size > 2:
            shutil.copyfile(main_dir / "curves" / src, out_main / "作图数据" / dst)
    if (main_dir / "methods.md").exists():
        shutil.copyfile(main_dir / "methods.md", out_main / "方法_实验设置.md")

    # ---- 大模型调用统计（从 llm_cache/ 读；不重算任何指标）----
    llm_note = ""
    if a.llm_cache and a.llm_cache.exists():
        import sqlite3
        import tempfile

        import re

        rows_llm = []
        prompts: dict[str, set[str]] = {}
        label_set = set(labels)
        runs_dir = a.summary.parent / "shards" / "runs"
        for db in sorted(a.llm_cache.rglob("*.sqlite")):
            lr = db.stem
            if lr not in learners:  # 只列主表范围内的学习器
                continue
            # 复制到临时目录再读（连同 -wal / -shm）：只读打开带 WAL 的库会漏读或打不开，也不改包里的文件
            with tempfile.TemporaryDirectory() as tmp:
                for f in db.parent.glob(db.name + "*"):
                    shutil.copyfile(f, Path(tmp) / f.name)
                con = sqlite3.connect(str(Path(tmp) / db.name))
                try:
                    recs = con.execute("SELECT api_model_id, api_usage_json, latency_ms, wall_time_iso FROM llm_cache").fetchall()
                    reqs = con.execute("SELECT request_json, response_raw FROM llm_cache").fetchall()
                finally:
                    con.close()
            if not recs:
                continue
            usage = [json.loads(u) if u else {} for _m, u, _l, _w in recs]
            lat = np.array([float(x) for _m, _u, x, _w in recs if x is not None]) / 1000.0
            ptok = np.array([float(u.get("prompt_tokens") or 0) for u in usage])
            ctok = np.array([float(u.get("completion_tokens") or 0) for u in usage])
            times = sorted(pd.to_datetime([w for *_x, w in recs], utc=True).tz_convert(BJ))
            prompts[lr] = {json.dumps(json.loads(q)["messages"], ensure_ascii=False, sort_keys=True) for q, _r in reqs}
            # 回答的形态（不导入框架；与框架解析规则一致：整段 JSON → 整段代码块 → 正文里唯一的 out/prob 对象）
            n_not_json = n_unknown = n_empty = 0
            for _q, raw in reqs:
                try:
                    obj = json.loads(raw)
                except Exception:  # noqa: BLE001
                    n_not_json += 1
                    obj = None
                    dec = json.JSONDecoder()
                    pos = raw.find("{")
                    while pos != -1 and obj is None:
                        try:
                            cand, end = dec.raw_decode(raw, pos)
                            if isinstance(cand, dict) and "out" in cand:
                                obj = cand
                            pos = raw.find("{", end)
                        except ValueError:
                            pos = raw.find("{", pos + 1)
                outs = obj.get("out", []) if isinstance(obj, dict) else []
                outs = outs if isinstance(outs, list) else [outs]
                n_unknown += any(str(x) not in label_set for x in outs)
                n_empty += not [x for x in outs if str(x) in label_set]
            # 重试用尽、按全 0 计入的病例与从说明文字取出 JSON 的病例：读各分片的降级记录
            # 框架批次 I 起另有逐例形态异常（n_cases=）；同一条消息在一个分片里出现多次时 CSV 只留一行、count 累加
            n_failed = n_extracted = 0
            n_anom = {k: 0 for k in LLM_ANOMALY_COLS}
            for deg in sorted((runs_dir / db.parent.name / lr).glob(f"s*/seed_*/{db.parent.name}/csv/degradations.csv")):
                dg = read(deg)
                if "message" not in dg.columns:
                    continue
                cnts = pd.to_numeric(dg["count"], errors="coerce").fillna(1).astype(int) if "count" in dg.columns else [1] * len(dg)
                for msg, c in zip(dg["message"].fillna("").astype(str), cnts):
                    m1 = re.search(r"predict_sample_retry_exhausted: n_failed=(\d+)/", msg)
                    m2 = re.search(r"predict_response_json_extracted: n_extracted=(\d+)/", msg)
                    m3 = re.search(r"LLM (predict_[a-z_]+): n_cases=(\d+)/", msg)
                    n_failed += int(m1.group(1)) * c if m1 else 0
                    n_extracted += int(m2.group(1)) * c if m2 else 0
                    if m3 and m3.group(1) in n_anom:
                        n_anom[m3.group(1)] += int(m3.group(2)) * c
            rows_llm.append({
                "学习器": lr, "论文用名": lname(lr), "接口模型名": "/".join(sorted({str(m) for m, *_x in recs})),
                "数据集": db.parent.name, "缓存条目数（不同的输入）": len(recs),
                "单次调用耗时中位数（秒）": round(float(np.median(lat)), 1), "单次调用耗时P95（秒）": round(float(np.percentile(lat, 95)), 1),
                "单次调用耗时最大值（秒）": round(float(lat.max()), 1),
                "输入token均值": round(float(ptok.mean()), 1), "输出token均值（含思考）": round(float(ctok.mean()), 1),
                "输出token最大值": int(ctok.max()), "输入token合计": int(ptok.sum()), "输出token合计": int(ctok.sum()),
                "首次调用（北京时间）": times[0].strftime("%Y-%m-%d %H:%M"), "末次调用（北京时间）": times[-1].strftime("%Y-%m-%d %H:%M"),
                "回答不是整段JSON的条数": n_not_json, f"回答含候选外{term}名的条数": n_unknown, f"回答未选任何候选{term}的条数": n_empty,
                "重试用尽按全0计入的预测数（全部分片合计）": n_failed, "从说明文字取出JSON的预测数（全部分片合计）": n_extracted,
                **{col.format(term=term): n_anom[k] for k, col in LLM_ANOMALY_COLS.items()},
            })
        if rows_llm:
            # 逐例形态异常的列只在有分片记了这一类时才列（旧框架的分片不记，全 0 会被误读成“没有”）
            for k, col in LLM_ANOMALY_COLS.items():
                if not any(n for r in rows_llm for c, n in r.items() if c == col.format(term=term)):
                    for r in rows_llm:
                        r.pop(col.format(term=term), None)
            pd.DataFrame(rows_llm).to_csv(out_main / "大模型调用统计.csv", index=False, encoding="utf-8-sig")
            llm_note = (f"- `大模型调用统计.csv`：{len(rows_llm)} 个通用大模型每次接口调用的耗时与 token、回答的形态、按全 0 计入的预测数"
                        "（读自结果目录的 `llm_cache/` 与各分片的降级记录；每个不同的输入一条，外部测试集与交叉验证的全部病例各问一次）。")
            if len(prompts) >= 2:
                first = next(iter(prompts.values()))
                same = all(v == first for v in prompts.values())
                chk.add(f"大模型提示词：{'、'.join(lname(k) for k in prompts)} 的请求全文逐字节相同（只换模型）", len(first), same,
                        "；".join(f"{lname(k)} {len(v)} 条" for k, v in prompts.items()))
            n_dev_test = None
            if a.data and a.data.exists():
                raw = json.loads(a.data.read_text(encoding="utf-8"))
                n_dev_test = len(raw["train_samples"]) + len(raw["test_samples"])
            for r in rows_llm:
                if n_dev_test is not None:
                    chk.add(f"大模型调用统计：{r['论文用名']} 的缓存条目数 ≤ 开发集 + 测试集例数（相同病例文字只问一次）", 1,
                            r["缓存条目数（不同的输入）"] <= n_dev_test, f"{r['缓存条目数（不同的输入）']} / {n_dev_test}")

    # ---- 可解释性（补实验）----
    explain_note = ""
    if a.explain and a.explain.exists():
        ex_out = out_main / "可解释性"
        ex_out.mkdir()
        params_ex = json.loads((a.explain / "explain_params.json").read_text(encoding="utf-8")) if (a.explain / "explain_params.json").exists() else {}
        tops: dict[str, pd.DataFrame] = {}
        present = []
        for key, method in EXPLAIN_MODELS:
            d = a.explain / key
            if not (d / "top5_by_label.csv").exists():
                continue
            present.append(key)
            tops[key] = read(d / "top5_by_label.csv")
            if key in WEIGHT_TABLE_MODELS:
                nw = d / "native_weights"
                for src, dst in ((nw / "native_weight_table.csv", "特征权值表.csv"), (nw / "native_segment_table.csv", "分段权值表.csv"),
                                 (d / "top5_by_label.csv", "每证素正向前5.csv")):
                    shutil.copyfile(src, ex_out / f"{lname(key)}_{dst}")
                # 核对：前 5 表可由特征权值表复算（只在症状行里排、只取正值、归一化到本列最大值）
                wt = read(nw / "native_weight_table.csv")
                sg = read(nw / "native_segment_table.csv")
                ck = json.loads((nw / "native_weights_check.json").read_text(encoding="utf-8"))
                wlabels = [c for c in wt.columns if c not in ("类别", "特征", "取值", "人数")]
                sym = (wt["取值"] == "出现").to_numpy()
                names = (wt["类别"].astype(str) + "：" + wt["特征"].astype(str)).tolist()
                n_items, n_bad = 0, 0
                for lbl, g in tops[key].groupby("label", sort=False):
                    col = wt[lbl].to_numpy(dtype=float)
                    top = [j for j in np.argsort(-col, kind="stable") if sym[j] and col[j] > 0][:5]
                    g = g.sort_values("rank")
                    n_bad += abs(len(top) - len(g))
                    for rank, (j, r) in enumerate(zip(top, g.itertuples()), start=1):
                        n_items += 1
                        same = (names[j] == r.feature and int(r.rank) == rank and int(wt["人数"].iloc[j]) == int(r.n_present)
                                and abs(col[j] - float(r.weight)) <= 1e-9 and abs(col[j] / col[top[0]] - float(r.normalized)) <= 1e-5)
                        n_bad += not same
                chk.add(f"可解释性：{lname(key)} 每证素正向前 5 可由特征权值表复算（症状行、正值、本列最大值归一化）", n_items, n_bad == 0, f"不同 {n_bad} 项")
                n_empty = int(wt[wlabels].isna().sum().sum())
                struct_ok = (len(wt) == ck["table_rows"] and int(sym.sum()) == ck["table_symptoms"] and n_empty == 0
                             and len(sg) == ck["segment_table_rows"] and ck["segment_table_untouched"] == 0)
                chk.add(f"可解释性：{lname(key)} 特征权值表的行数、症状行数、无空格，分段权值表的行数与运行记录一致", 5, struct_ok,
                        f"权值表 {len(wt)} 行 × {len(wlabels)} {term}（症状 {int(sym.sum())} 行，空格 {n_empty} 个）；分段权值表 {len(sg)} 行")
                weights_ok = (ck["decision_mismatch"] == 0 and ck["predict_proba_identical_after"] is True
                              and float(ck["segment_sum_vs_feature_weight_max"]) < 1e-9)
                chk.add(f"可解释性：{lname(key)} 由权值复算的判定与模型判定一致、算权值前后预测逐位相同、各段权值加回特征权值", 3, weights_ok,
                        f"判定不一致 {ck['decision_mismatch']} / {ck['decision_total']}；各段加回最大差 {float(ck['segment_sum_vs_feature_weight_max']):.1e}；"
                        f"参照样本对数几率 + 权值之和与模型输出对数几率的最大差 {float(ck['completeness_err_max']):.4f}（仅记录）")
                continue
            for src, dst in (("mean_signed.csv", "特征贡献_带符号均值.csv"), ("mean_abs.csv", "特征贡献_绝对值均值.csv"), ("top5_by_label.csv", "每证素正向前5.csv")):
                if (d / src).exists():
                    shutil.copyfile(d / src, ex_out / f"{lname(key)}_{dst}")
            # 核对：前 5 表可由带符号均值表复算（只在症状列里排、只取正值、归一化到本列最大值）
            ms = read(d / "mean_signed.csv")
            feats = list(ms["feature"])
            allowed = np.array([not f.startswith(NON_SYMPTOM) and f not in NON_SYMPTOM for f in feats])
            n_items, n_bad = 0, 0
            for lbl, g in tops[key].groupby("label", sort=False):
                col = ms[lbl].to_numpy(dtype=float)
                order = np.argsort(-col, kind="stable")
                top = [j for j in order if col[j] > 0 and allowed[j]][:5]
                g = g.sort_values("rank")
                for rank, (j, r) in enumerate(zip(top, g.itertuples()), start=1):
                    n_items += 1
                    close = abs(col[j] - float(r.signed_mean)) <= 1e-5 * max(1.0, abs(col[j]))
                    same = feats[j] == r.feature or (close and abs(col[j] - col[feats.index(r.feature)]) <= 1e-5 * max(1.0, abs(col[j])))
                    if not (same and close) or int(r.rank) != rank:
                        n_bad += 1
            chk.add(f"可解释性：{lname(key)} 每证素正向前 5 可由带符号均值表复算（症状列、正值、本列最大值归一化）", n_items, n_bad == 0, f"不同 {n_bad} 项")
        # 保真 / 可加性核对表 + 三模型并排前 5
        fid = {"dannv2_beta": "fidelity.json", "random_forest": "additivity_check.json", "tabicl": "rebuild_check.json"}
        rows = []
        crit_ok = True
        for key, method in EXPLAIN_MODELS:
            if key not in present:
                continue
            f = a.explain / key / fid[key]
            rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            fidelity = rec.get("fidelity_max_abs_diff", rec.get("rebuild_max_abs_diff"))
            addit = rec.get("additivity_max_err")
            ident = rec.get("predict_proba_identical_after_explain")
            limit = 1e-5 if key == "tabicl" else 1e-6
            # 方案 §9：保真 ≤ 1e-6（TabICL 重建 ≤ 1e-5）；可加性 < 1e-6 只对随机森林的 TreeSHAP 作验收标准；
            # KernelSHAP 的可加性误差只记录（求和约束由解释器精确施加，误差来自模型推理本身按批次的浮点差）。
            addit_ok = addit is None or key != "random_forest" or float(addit) < 1e-6
            ok = fidelity is not None and float(fidelity) <= limit and addit_ok and ident is not False
            crit_ok = crit_ok and ok
            addit_text = "不适用" if addit is None else (f"{float(addit):.2e}" + ("（对数几率，仅记录）" if key in WEIGHT_TABLE_MODELS else "（仅记录）" if key != "random_forest" else ""))
            rows.append([lname(key), method, f"{float(fidelity):.2e}" if fidelity is not None else "—", addit_text, fnum(ident), "是" if ok else "否"])
        chk.add("可解释性：三模型都有产出", 3, len(present) == 3, f"有 {len(present)}：{', '.join(lname(k) for k in present)}")
        chk.add("可解释性：保真（≤1e-6；TabICL 重建 ≤1e-5）、随机森林可加性（<1e-6）、归因后预测不变", len(rows), crit_ok)
        labels_ex = list(dict.fromkeys(pd.concat(tops.values())["label"])) if tops else []
        n_test = params_ex.get("explain_set", {}).get("n", "—")
        n_weight = params_ex.get("models", {}).get("dannv2_beta", {}).get("explain_set", {}).get("n", "—")
        text = ["# 可解释性：三模型每证素正向前 5 症状", "",
                "三个模型各用自己的量给出“特征 × 证素”的带符号数值。DANNV2-beta 用模型自带的特征权值："
                f"`DANNV2-beta_特征权值表.csv` 的格值，单位是对数几率，由开发集加测试集 {n_weight} 例算出。"
                f"随机森林、TabICL 用 SHAP 值在外部测试集 {n_test} 例上的带符号均值，单位是概率。"
                "下表每格是该证素正向数值最大的 5 个症状（只在 345 个症状里排；年龄、性别、病程见各模型的完整表），括号内为归一化值（除以本列最大值）与人数"
                f"（DANNV2-beta 为 {n_weight} 例里出现该症状的人数，另两模型为该症状在测试集的阳性人数）。"
                "三个模型的量单位不同、解释集不同，原始数值只在同一模型内比较大小；跨模型比较用各自归一化到 0–1 之后的数值与排序。", "",
                "| 模型 | 方法 | 保真（与存档预测的最大绝对差） | 可加性误差 | 算完后预测不变 | 达标 |", "|---|---|---|---|---|---|"]
        text += ["| " + " | ".join(r) + " |" for r in rows]
        text += ["", "达标标准：保真 ≤ 1e-6（TabICL 为重建模型与存档预测之差，≤ 1e-5）；可加性 < 1e-6 只对随机森林的 TreeSHAP 要求。"
                 "DANNV2-beta 的“可加性误差”是参照样本的对数几率加各特征权值之和，与模型输出对数几率之差的最大值（对数几率单位，只记录；"
                 "由权值复算的判定与模型判定全部一致，见核对报告）。"
                 "TabICL 的 KernelSHAP 可加性误差只记录——解释器对每例精确施加“各特征贡献之和 = f(x) − f(参考)”的约束，表中数值是它内部算的 f(x) 与整批预测之差，"
                 "反映 TabICL 推理结果随批次组成的浮点波动（量级见 `归因参数记录.json`）。"]
        text += ["", "| 证素 | " + " | ".join(lname(k) for k in present) + " |", "|---|" + "---|" * len(present)]
        long_rows = []
        for lbl in labels_ex:
            cells = []
            for key in present:
                g = tops[key][tops[key]["label"] == lbl].sort_values("rank")
                if key in WEIGHT_TABLE_MODELS:
                    cells.append("；".join(f"{r.feature.split('：')[-1]} {r.weight:+.2f}（{r.normalized:.2f}，n={int(r.n_present)}）" for r in g.itertuples()))
                    for r in g.itertuples():
                        long_rows.append({"模型": lname(key), "证素": lbl, "名次": int(r.rank), "特征": r.feature, "值": r.weight,
                                          "归一化": r.normalized, "量": "特征权值（对数几率）", "人数": int(r.n_present),
                                          "人数口径": f"开发集加测试集 {n_weight} 例里出现该症状的人数", "绝对值均值": np.nan})
                    continue
                cells.append("；".join(f"{r.feature.split('：')[-1]} {r.signed_mean:.3f}（{r.normalized:.2f}，n={int(r.n_positive_test)}）" for r in g.itertuples()))
                for r in g.itertuples():
                    long_rows.append({"模型": lname(key), "证素": lbl, "名次": int(r.rank), "特征": r.feature, "值": r.signed_mean,
                                      "归一化": r.normalized, "量": "SHAP 带符号均值（概率）", "人数": int(r.n_positive_test),
                                      "人数口径": f"测试集 {n_test} 例里该症状的阳性人数", "绝对值均值": r.mean_abs})
            text.append(f"| {lbname(lbl)} | " + " | ".join(cells) + " |")
        text += ["", f"参数记录：`归因参数记录.json`（方法、参考样本、版本、核对数字）；参考样本：`参考样本.json`。"]
        (ex_out / "三模型每证素正向前5.md").write_text("\n".join(text), encoding="utf-8")
        pd.DataFrame(long_rows).to_csv(ex_out / "三模型每证素正向前5_长表.csv", index=False, encoding="utf-8-sig")
        for src, dst in (("explain_params.json", "归因参数记录.json"), ("reference_row.json", "参考样本.json")):
            if (a.explain / src).exists():
                shutil.copyfile(a.explain / src, ex_out / dst)
        explain_note = ("- `可解释性/`：DANNV2-beta 的特征权值表与分段权值表（模型自带）、随机森林与 TabICL 的特征 × 证素 SHAP 矩阵、"
                        "三模型每证素正向前 5（并排表与长表）、归因参数记录、参考样本。")

    # 主结果的每个 csv 都不含不进报告的指标（总体的 ECE、MCE、Coverage error 与逐标签 ECE；批次 J）
    hidden = {*NOT_REPORTABLE, *(mname(m) for m in NOT_REPORTABLE), "ece", "ECE"}
    leak, n_csv = [], 0
    for f in sorted(out_main.rglob("*.csv")):
        n_csv += 1
        t = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        for c in t.columns:
            if c == "metric" or "指标" in str(c):
                hit = sorted(set(t[c].dropna()) & hidden)
                if hit:
                    leak.append(f"{f.relative_to(out_main)} 的 {c} 列：{'、'.join(hit)}")
    chk.add("主结果的全部 csv 都不含不进报告的指标（ECE、MCE、Coverage error 与逐标签 ECE）", n_csv, not leak, "；".join(leak[:5]))

    # ---- 说明与核对报告 ----
    title = a.title or "复现包"
    lines = [f"# {title}", "", f"生成时间（北京时间）：{now()}。表由框架 `report` 的主结果（`summary/main/`）排版而来，不重算任何指标。", "",
             "## 范围", "", f"- 学习器（按定义顺序）：{'、'.join(lname(x) for x in learners)}。", f"- 指标：{'、'.join(mname(m) for m in metrics)}。", f"- {term}：{'、'.join(lbname(x) for x in labels)}。", "",
             "## 主口径", "", rule, ""]
    if not pm.empty:
        nb0, bs0 = int(pm["n_boot"].iloc[0]), int(pm["bootstrap_seed"].iloc[0])
        lines += ["## 统计检验（《模型比较统计检验规范》）", "",
                  "方法句：" + METHOD_SENTENCE.format(n_boot=nb0), "",
                  f"- 一族 = 一项指标内全部学习器的两两比较；重抽种子 {bs0}；所有学习器、所有种子、所有指标共用同一张重抽样索引矩阵（成对）；"
                  "每次重抽样对各种子算指标后取均值；阈值按主口径冻结，大模型取自身判定。",
                  "- " + CI_TEST_NOTE + "有效重抽次数不足 90% 的比较记为不可计算，不报 P、不计入族。交叉验证一侧只作描述，不做检验。",
                  "- 不要这样做：用均值 ± 标准差或区间是否重叠判断有无差异；拿种子做 t 检验或秩和检验；把交叉验证各折当独立样本做检验；"
                  "只报未校正的结果或只挑有利的几对；没有检验结果支撑就写“相当”“接近”“显著优于”。",
                  "- ECE、MCE、Coverage error 在重抽样下有偏或不稳，不出区间、不做检验，不进报告；校准看 Brier 与校准曲线。", ""]
    lines += ["## `主结果/` 里有什么", ""]
    if "表0_数据描述" in has_tables:
        lines.append(f"- 表 0：数据描述（两个划分的例数、特征数、每例{term}数、缺失、各{term}阳性数）；表 0b：开发集{term}共现矩阵与长表。")
    lines += [f"- 表 1、表 2：总体性能（交叉验证 / 外部测试集），`.md` 给人看、`.csv` 同内容" + ("；表 2 带合并种子的 bootstrap 95% 区间。" if has_ci else "。"),
              f"- 表 3、表 4：各{term}的结果。"]
    if "表5_成对检验_外部测试集" in has_tables:
        lines.append("- 表 5：主指标上学习器两两的成对检验（差值与 95% 区间，Holm 校正后有统计学意义的加 *）；"
                     "`成对检验_Holm.csv`：全部报告指标 × 全部两两的差值、区间、P原始、P_Holm、有统计学意义、族大小、重抽次数与种子。"
                     + (f"表 2 里的 † 以 {lname(ref)} 为参照。" if ref else ""))
    lines += [f"- `总体性能_长表.csv`、`各{term}_长表.csv`：逐种子取值与均值、标准差" + ("（外部测试集行附合并种子区间）；`外部测试集_逐种子置信区间.csv`。" if has_ci else "。"),
              "- `部署阈值.csv`、`最终模型参数.csv`、`计算开销.csv`（训练侧 + 部署侧两口径）、`计算开销_逐种子.csv`、`作图数据/`、`方法_实验设置.md`。"]
    if llm_note:
        lines.append(llm_note)
    if explain_note:
        lines.append(explain_note)
    lines.append("")
    if a.rerun_notes:
        lines += ["## 怎么重跑实验", "", a.rerun_notes.read_text(encoding="utf-8").strip(), ""]
    if a.notes:
        lines += ["## 这个项目要特别知道的事（人工写的，不是脚本生成的）", "", a.notes.read_text(encoding="utf-8").strip(), ""]
    (a.out / "说明_先读我.md").write_text("\n".join(lines), encoding="utf-8")
    rep = [f"# 核对报告", "", f"生成时间（北京时间）：{now()}", "", "| 核对内容 | 项数 | 结论 | 说明 |", "|---|---|---|---|"]
    rep += [f"| {w} | {n} | {r} | {note} |" for w, n, r, note in chk.rows]
    rep += ["", "**结论**：" + ("全部一致" if chk.ok else "有不一致，不能交付")]
    (a.out / "核对报告.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    return 0 if chk.ok else 1


if __name__ == "__main__":
    sys.exit(main())
