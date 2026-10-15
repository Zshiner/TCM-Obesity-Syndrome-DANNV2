# -*- coding: utf-8 -*-
"""可解释性补实验的公共工具（《可解释性补实验方案（定稿）》§2、§6）。

explain_rf.py / explain_tabicl.py / bishan_native_weights.py 等脚本共用：读数据、参考样本、
存档预测、保真核对、汇总表（带符号均值 / 平均绝对值 / 每证素正向前 5）、参数记录。
只读模型与实验产物；产出写到本目录下 explain/<模型>/。
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPLAIN_DIR = HERE / "explain"
DEFAULT_DATA = HERE / "data" / "obesity_syndrome" / "data.json"
SYMPTOM_PREFIXES = ("症状", "舌", "脉", "问诊")
CONTINUOUS = ("年龄", "病程")
CATEGORICAL = ("性别",)
MISSING = -1.0
BEIJING = timezone(timedelta(hours=8))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def now_beijing() -> str:
    return datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M:%S +08:00")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("shap", "sklearn", "torch", "tabicl", "numpy", "joblib"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v is None and mod == "tabicl":
                try:
                    from importlib.metadata import version as _v
                    v = _v("tabicl")
                except Exception:  # noqa: BLE001
                    v = "?"
            out[mod] = str(v)
        except Exception:  # noqa: BLE001
            out[mod] = "未安装"
    return out


# ---------------- 数据 ----------------

class Data:
    def __init__(self, path: Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.path = Path(path)
        self.feature_names: list[str] = [str(f) for f in d["feature_names"]]
        self.label_names: list[str] = [str(l) for l in d["label_names"]]
        self.X_dev = np.array([s["features"] for s in d["train_samples"]], dtype=np.float64)
        self.Y_dev = np.array([s["labels"] for s in d["train_samples"]], dtype=np.int64)
        self.X_test = np.array([s["features"] for s in d["test_samples"]], dtype=np.float64)
        self.Y_test = np.array([s["labels"] for s in d["test_samples"]], dtype=np.int64)
        self.case_ids = list(range(len(self.X_test)))  # 与框架 cases 导出表的 case_id 相同（测试集行序）

    @property
    def symptom_mask(self) -> np.ndarray:
        return np.array([f.startswith(SYMPTOM_PREFIXES) for f in self.feature_names])

    def case_ids_sha256(self) -> str:
        return sha256_text(json.dumps(self.case_ids))


def load_data(path: Optional[str | Path] = None) -> Data:
    return Data(Path(path) if path else DEFAULT_DATA)


# ---------------- 参考样本（§2.1） ----------------

def build_reference_row(data: Data) -> tuple[np.ndarray, dict[str, Any]]:
    """一行：症状全 0；年龄、病程取开发集中位数；性别取开发集众数（三者都只在非缺失值上算，缺失标记 -1）。"""
    ref = np.zeros(len(data.feature_names), dtype=np.float64)
    rules: dict[str, Any] = {}
    for j, name in enumerate(data.feature_names):
        col = data.X_dev[:, j]
        valid = col[col != MISSING]
        if name in CONTINUOUS:
            ref[j] = float(np.median(valid))
            rules[name] = {"rule": "开发集中位数（排除缺失标记 -1）", "value": ref[j], "n_valid": int(valid.size)}
        elif name in CATEGORICAL:
            vals, cnt = np.unique(valid, return_counts=True)
            ref[j] = float(vals[int(np.argmax(cnt))])
            rules[name] = {"rule": "开发集众数（排除缺失标记 -1）", "value": ref[j], "n_valid": int(valid.size)}
        else:
            ref[j] = 0.0
    record = {
        "rule": "症状=0；年龄、病程=开发集中位数；性别=开发集众数",
        "n_symptom_columns_zero": int(sum(1 for n in data.feature_names if n not in CONTINUOUS + CATEGORICAL)),
        "non_symptom_columns": rules,
        "values": {name: float(ref[j]) for j, name in enumerate(data.feature_names) if name in CONTINUOUS + CATEGORICAL},
        "data_file": str(data.path.name), "data_sha256": sha256_file(data.path),
    }
    return ref, record


def write_reference_row(data: Data, out_dir: Path = EXPLAIN_DIR) -> tuple[np.ndarray, Path]:
    ref, record = build_reference_row(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reference_row.json"
    full = dict(record)
    full["row"] = {name: float(ref[j]) for j, name in enumerate(data.feature_names)}
    path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    return ref, path


# ---------------- 存档预测与保真（§2.4） ----------------

def load_final_npz(path: Path, data: Data) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out = {k: npz[k] for k in npz.files}
    fn = [str(f) for f in out["feature_names"]]
    ln = [str(l) for l in out["label_names"]]
    if fn != data.feature_names or ln != data.label_names:
        raise RuntimeError("存档 final_test.npz 的特征 / 证素名与数据文件不一致")
    if not np.array_equal(out["X"].astype(np.float64), data.X_test):
        raise RuntimeError("存档 final_test.npz 的 X 与数据文件的测试集不一致")
    if not np.array_equal(out["y_true"].astype(np.int64), data.Y_test):
        raise RuntimeError("存档 final_test.npz 的 y_true 与数据文件的测试集标签不一致")
    return out


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


# ---------------- 汇总表（§2.3） ----------------

def positive_counts(data: Data) -> np.ndarray:
    """每个特征在解释集（测试集）里的阳性人数（症状 >0；连续列记非缺失人数）。"""
    X = data.X_test
    out = np.zeros(X.shape[1], dtype=int)
    for j, name in enumerate(data.feature_names):
        if name in CONTINUOUS + CATEGORICAL:
            out[j] = int((X[:, j] != MISSING).sum())
        else:
            out[j] = int((X[:, j] > 0).sum())
    return out


def matrix_frame(M: np.ndarray, data: Data) -> pd.DataFrame:
    df = pd.DataFrame(M, index=pd.Index(data.feature_names, name="feature"), columns=data.label_names)
    return df


def write_matrix(M: np.ndarray, data: Data, path: Path) -> None:
    matrix_frame(M, data).to_csv(path, encoding="utf-8-sig", float_format="%.8g")


def top5_table(signed: np.ndarray, mean_abs: np.ndarray, data: Data, k: int = 5, symptoms_only: bool = True) -> pd.DataFrame:
    """每证素正向前 k（按带符号均值降序，只取 >0），归一化 = 值 / 本列最大值（问卷口径 §2.3、§7）；
    附该特征在测试集的阳性人数，以及"每个有该症状者的平均效应"（辅助列，不参与排名）。
    ``symptoms_only``：只在 345 个症状列里排（年龄、性别、病程不进问卷与前 5 表；方案 §3.3 的验收数字即此口径），
    三个连续 / 类别列的贡献仍在整表 mean_signed.csv / mean_abs.csv 里。"""
    rows = []
    npos = positive_counts(data)
    X = data.X_test
    allowed = data.symptom_mask if symptoms_only else np.ones(len(data.feature_names), dtype=bool)
    for li, label in enumerate(data.label_names):
        col = signed[:, li]
        order = np.argsort(-col)
        top = [j for j in order if col[j] > 0 and allowed[j]][:k]
        vmax = col[top[0]] if top else np.nan
        for rank, j in enumerate(top, start=1):
            present = X[:, j] > 0 if data.feature_names[j] not in CONTINUOUS + CATEGORICAL else X[:, j] != MISSING
            rows.append({
                "label": label, "rank": rank, "feature": data.feature_names[j],
                "signed_mean": float(col[j]), "normalized": float(col[j] / vmax) if vmax else np.nan,
                "mean_abs": float(mean_abs[j, li]), "n_positive_test": int(npos[j]),
                "mean_effect_among_positive": float(col[j] * len(X) / max(1, int(present.sum()))) if present.any() else np.nan,
            })
    return pd.DataFrame(rows)


def write_summary_tables(signed: np.ndarray, mean_abs: np.ndarray, data: Data, out_dir: Path,
                         signed_name: str = "mean_signed.csv", abs_name: str = "mean_abs.csv") -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(signed, data, out_dir / signed_name)
    write_matrix(mean_abs, data, out_dir / abs_name)
    top = top5_table(signed, mean_abs, data)
    top.to_csv(out_dir / "top5_by_label.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    return top


def print_top5(top: pd.DataFrame, labels: Optional[list[str]] = None) -> None:
    for label, g in top.groupby("label", sort=False):
        if labels and label not in labels:
            continue
        items = "、".join(f"{r.feature.split('：')[-1]} {r.signed_mean:.3f}" for r in g.itertuples())
        print(f"  {label} = {items}")


# ---------------- 参数记录（§6） ----------------

PARAMS_PATH = EXPLAIN_DIR / "explain_params.json"


def update_params(model_key: str, record: dict[str, Any], *, data: Data, reference_path: Path) -> Path:
    """把一个模型的记录并进 explain/explain_params.json（结构按方案 §6）。"""
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    if PARAMS_PATH.exists():
        params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    else:
        params = {
            "date": now_beijing(), "executor": "Claude Code（补实验执行）",
            "versions": {},
            "explain_set": {"split": "external_test", "n": int(len(data.X_test)), "case_ids_sha256": data.case_ids_sha256(),
                            "data_file": data.path.name, "data_sha256": sha256_file(data.path)},
            "reference_row": {"file": reference_path.name, "rule": "症状=0；年龄、病程=开发集中位数；性别=开发集众数"},
            "models": {},
            "aggregation": {"global": "mean over explain set (signed and abs)",
                            "questionnaire": "signed mean, positive top-5, normalized by column max"},
            "excluded_methods": ["flip_effect", "per_label_impurity_split", "deep_explainer", "gradient_explainer",
                                 "divide_by_activation_rate_ranking"],
        }
    params["versions"].update({k: v for k, v in versions().items() if v not in ("未安装", "?") or k not in params["versions"]})
    params["models"][model_key] = record
    params["updated"] = now_beijing()
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return PARAMS_PATH


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
