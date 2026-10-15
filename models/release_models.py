"""从实验时保存的原模型文件导出公开用的最终模型，并在原服务器上逐位核对。

两步（同一脚本，第一个参数选）：
  export：需能导入 tcmhubd（RealMLP 的适配器类在框架里）。对 5 个学习器 × 5 个种子：
          1. 原文件 SHA-256 与 manifest.json 里的实验记录比对；
          2. 加载原模型，对存档的外部测试集特征 X 预测，与存档 y_prob 比对（CPU；DANNV2-beta、
             RealMLP 另在 GPU 上比对——论文的预测是训练完在 GPU 上算的）；
          3. 去掉框架依赖（RealMLP 的薄子类换回 pytabkit 原类）、torch 张量搬到 CPU，
             joblib compress=3 存成 <输出目录>/<学习器>_seed<种子>.joblib；
          4. 重新加载导出文件，在 CPU 上与原模型的 CPU 预测逐位比对（导出没有改动模型）。
  verify：屏蔽 tcmhubd（import 即报错），只用公开的东西加载导出文件，分别在 CPU 与 GPU 上预测，
          与存档 y_prob 比对，并按部署阈值（概率 ≥ 阈值）统计判定翻转数。

用法：
  python release_models.py export <runs 根> <输出目录> <dannv2_beta 源码目录> <结果CSV>
  python release_models.py verify <runs 根> <输出目录> <dannv2_beta 源码目录> <部署阈值.csv> <结果CSV>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

UNITS = {"decision_tree": "decision_tree", "random_forest": "random_forest", "extra_trees": "extra_trees",
         "dannv2_beta": "dannv2_beta", "realmlp": "br-realmlp"}
TORCH_LEARNERS = {"dannv2_beta", "realmlp"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_paths(runs: Path, learner: str, seed: int):
    base = runs / learner / f"s0{seed}final" / f"seed_{seed}" / "obesity_syndrome"
    unit = UNITS[learner]
    return (base / "models" / unit / "final" / "best_model.joblib",
            base / "models" / unit / "manifest.json",
            base / "predictions" / unit / "final_test.npz")


def load_test(npz: Path):
    import pandas as pd

    z = np.load(npz, allow_pickle=True)
    feat = [str(x) for x in z["feature_names"]]
    labels = [str(x) for x in z["label_names"]]
    X = pd.DataFrame(np.asarray(z["X"], dtype=float), columns=feat)
    return X, labels, np.asarray(z["y_prob"], dtype=float)


def pos_proba(model, X) -> np.ndarray:
    """与框架同一口径取各证素正类概率：list 输出逐列取正类，单类列按 classes_ 补 0 或 1。"""
    p = model.predict_proba(X)
    if not isinstance(p, list):
        return np.asarray(p, dtype=float)
    ests = getattr(model, "estimators_", None)
    cols = []
    for k, pk in enumerate(p):
        pk = np.asarray(pk)
        if pk.shape[1] == 2:
            cols.append(pk[:, 1])
        else:
            classes = ests[k].classes_ if ests is not None and hasattr(ests[k], "classes_") else model.classes_[k]
            cols.append(np.ones(len(pk)) if int(classes[0]) == 1 else np.zeros(len(pk)))
    return np.column_stack(cols).astype(float)


def to_device(model, learner: str, dev: str):
    """把 torch 模型整体搬到 dev（"cpu" 或 "cuda:0"）；树模型原样返回。"""
    import torch

    if learner == "realmlp":
        for est in model.estimators_:
            est.to(dev)
    elif learner == "dannv2_beta":
        m = model.model_
        m.emb.to(dev)
        for k in ("_W", "_s0", "_s1"):
            setattr(m, k, torch.nn.Parameter(getattr(m, k).detach().to(dev)))
        m._r = m._r.detach().to(dev)
        m.device = str(dev)
    return model


def predict(model, learner: str, X):
    if learner == "realmlp":
        return pos_proba(model, np.asarray(X, dtype=np.float32))  # 与训练、评估时的输入精度一致
    return pos_proba(model, X)


def compare(saved, new):
    return bool(np.array_equal(saved, new)), float(np.max(np.abs(saved - new)))


def export(runs, out_dir, dann_dir, result_csv) -> None:
    sys.path.insert(0, dann_dir)
    import joblib

    runs, out_dir = Path(runs), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for learner in UNITS:
        for seed in range(5):
            src, manifest, npz = unit_paths(runs, learner, seed)
            rec = json.loads(manifest.read_text(encoding="utf-8"))["final_model"]["sha256"]
            h = sha256(src)
            X, labels, saved = load_test(npz)
            model = joblib.load(src)
            if learner in TORCH_LEARNERS:
                eq_gpu, d_gpu = compare(saved, predict(to_device(model, learner, "cuda:0"), learner, X))
                to_device(model, learner, "cpu")
            else:
                eq_gpu, d_gpu = None, None
            p_cpu = predict(model, learner, X)
            eq_cpu, d_cpu = compare(saved, p_cpu)
            if learner == "realmlp":
                from pytabkit import RealMLP_TD_Classifier
                from sklearn.multioutput import MultiOutputClassifier

                assert isinstance(model, MultiOutputClassifier), type(model)
                # 框架的 RealMLPClassifierAdapter 只是 RealMLP_TD_Classifier 的薄子类：没有自己的状态，
                # 只在 fit/predict 入口把 X 转 float32。换回原类后，按 float32 输入预测结果不变。
                for est in [model.estimator, *model.estimators_]:
                    assert type(est).__mro__[1] is RealMLP_TD_Classifier, type(est).__mro__
                    # 子类只定义了 fit / predict / predict_proba；set_*_request 是 scikit-learn 给每个子类自动生成的
                    own = {k for k in vars(type(est)) if not (k.startswith("set_") and k.endswith("_request"))}
                    assert own <= {"__module__", "__doc__", "fit", "predict", "predict_proba", "__qualname__",
                                   "__firstlineno__", "__static_attributes__", "__annotations__"}, sorted(own)
                    est.__class__ = RealMLP_TD_Classifier
            out = out_dir / f"{learner}_seed{seed}.joblib"
            joblib.dump(model, out, compress=3)
            back = joblib.load(out)
            same_as_orig = bool(np.array_equal(p_cpu, predict(back, learner, X)))
            row = [learner, seed, h, "一致" if h == rec else f"不一致（记录 {rec}）",
                   "逐位一致" if eq_cpu else f"不一致 {d_cpu:.3g}",
                   "" if eq_gpu is None else ("逐位一致" if eq_gpu else f"不一致 {d_gpu:.3g}"),
                   out.name, out.stat().st_size, sha256(out), "逐位一致" if same_as_orig else "不一致"]
            rows.append(row)
            print(*row, flush=True)
    with open(result_csv, "w", encoding="utf-8-sig", newline="") as g:
        wr = csv.writer(g)
        wr.writerow(["学习器", "种子", "原文件 SHA-256", "原文件与实验记录", "原模型 CPU 预测与存档",
                     "原模型 GPU 预测与存档", "导出文件", "字节", "导出文件 SHA-256", "导出文件与原模型（CPU）预测"])
        wr.writerows(rows)


def verify(runs, out_dir, dann_dir, thr_csv, result_csv) -> None:
    sys.modules["tcmhubd"] = None  # 屏蔽框架：任何对 tcmhubd 的导入都会失败
    sys.path.insert(0, dann_dir)
    import joblib
    import pandas as pd
    import torch

    thr = pd.read_csv(thr_csv, encoding="utf-8-sig")
    has_gpu = torch.cuda.is_available()
    rows = []
    for learner in UNITS:
        for seed in range(5):
            _, _, npz = unit_paths(Path(runs), learner, seed)
            X, labels, saved = load_test(npz)
            row_t = thr[(thr["学习器"] == learner) & (thr["种子"] == seed)]
            assert len(row_t) == 1, (learner, seed)
            tau = row_t[labels].to_numpy(dtype=float).ravel()
            f = Path(out_dir) / f"{learner}_seed{seed}.joblib"
            res = [learner, seed, f.name, sha256(f)]
            for dev in ["cpu", "cuda:0"]:
                if dev != "cpu" and (learner not in TORCH_LEARNERS or not has_gpu):
                    res += ["", "", ""]
                    continue
                model = to_device(joblib.load(f), learner, dev) if dev != "cpu" else joblib.load(f)
                new = predict(model, learner, X)
                eq, d = compare(saved, new)
                flips = int(np.sum((saved >= tau) != (new >= tau)))
                res += ["逐位一致" if eq else "不一致", f"{d:.3g}", flips]
            rows.append(res)
            print(*res, flush=True)
    with open(result_csv, "w", encoding="utf-8-sig", newline="") as g:
        wr = csv.writer(g)
        wr.writerow(["学习器", "种子", "文件", "SHA-256", "CPU 预测与存档", "CPU 最大绝对差", "CPU 判定翻转数",
                     "GPU 预测与存档", "GPU 最大绝对差", "GPU 判定翻转数"])
        wr.writerows(rows)
    print(f"torch {torch.__version__}；GPU：{torch.cuda.get_device_name(0) if has_gpu else '无'}")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "export":
        export(*sys.argv[2:6])
    elif step == "verify":
        verify(*sys.argv[2:7])
    else:
        raise SystemExit(__doc__)
