# -*- coding: utf-8 -*-
"""TabICL KernelSHAP（《可解释性补实验方案（定稿）》§5）。在课题组 GPU 服务器上跑。

  探针（§5.2，只读，约 10 分钟）：
    python explain_tabicl.py --model <rebuilt best_model.joblib> --npz <final_test.npz> --probe
  正式（可按证素分批、可续跑；每个证素的结果先落 parts/，全部齐了再汇总）：
    python explain_tabicl.py --model ... --npz ... [--labels 0,1,...] [--kv-cache auto|on|off]

模型：框架 `rebuild --learner tabicl --seed 0 --final --out` 重建并与存档预测逐位比对过的部署级全量模型
（BR：MultiOutputClassifier 里 11 个 TabICLClassifier 子模型）。设置：每个证素子模型一个
shap.KernelExplainer(f_l, 参考样本一行)，f_l = 子模型 predict_proba 的类 1 列；shap_values(X_test,
nsamples="auto", l1_reg=False)。只读模型；产出写到 explain/tabicl/。
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

import explain_common as ec

OUT = ec.EXPLAIN_DIR / "tabicl"
PARTS = OUT / "parts"


def load_model(path: Path):
    m = joblib.load(path)
    from sklearn.multioutput import MultiOutputClassifier
    if not isinstance(m, MultiOutputClassifier):
        raise RuntimeError(f"期望 BR 的 MultiOutputClassifier，收到 {type(m).__name__}")
    subs = list(m.estimators_)
    return m, subs


def kv_param_name(sub) -> str | None:
    return next((k for k in sub.get_params() if "kv" in k.lower() and "cache" in k.lower()), None)


def set_kv(subs, name: str | None, value: bool) -> None:
    if name is None:
        return
    for s in subs:
        s.set_params(**{name: value})


def proba_full(m, X: np.ndarray) -> np.ndarray:
    P = m.predict_proba(X)
    return np.column_stack([p[:, 1] for p in P])


def f_for(sub):
    def f(X):
        return np.asarray(sub.predict_proba(np.asarray(X, dtype=np.float64)))[:, 1]
    return f


def varying_counts(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """每例与参考样本取值不同的特征个数 M（KernelExplainer 只对这些特征做联盟；M ≤ 11 时全枚举、精确）。"""
    return (X != ref[None, :]).sum(axis=1)


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


# ---------------- 探针 ----------------

def probe(m, subs, data: ec.Data, ref: np.ndarray, saved: dict, model_path: Path) -> dict:
    import shap
    from sklearn.metrics import roc_auc_score

    rec: dict = {"date": ec.now_beijing(), "model_path": str(model_path), "model_sha256": ec.sha256_file(model_path)}
    sub0 = subs[0]
    rec["sub_model_class"] = f"{type(sub0).__module__}.{type(sub0).__name__}"
    rec["sub_model_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in sub0.get_params().items()}
    kv = kv_param_name(sub0)
    rec["kv_cache_param"] = kv
    kv_initial = sub0.get_params().get(kv) if kv else None
    rec["kv_cache_initial"] = kv_initial

    # ① 重建一致性（整模型）
    P, t = timed(proba_full, m, data.X_test)
    rec["rebuild_max_abs_diff"] = ec.max_abs_diff(P, saved["y_prob"])
    rec["predict_full_308_s"] = round(t, 2)
    print(f"① 重建一致性：predict_proba 与存档 y_prob 最大绝对差 {rec['rebuild_max_abs_diff']:.3g}（阈值 1e-5），整模型 308 例 {t:.1f} s")

    # ② kv_cache 开 / 关 的预测差（证素 0 子模型）
    f0 = f_for(sub0)
    if kv is not None:
        try:
            set_kv([sub0], kv, False)
            p_off, t_off = timed(f0, data.X_test)
            set_kv([sub0], kv, True)
            p_on_1, t_on1 = timed(f0, data.X_test)      # 第一次：建缓存
            p_on_2, t_on2 = timed(f0, data.X_test)      # 第二次：用缓存
            rec["kv_cache_max_abs_diff"] = ec.max_abs_diff(p_on_2, p_off)
            rec["kv_cache_first_vs_second_call_max_abs_diff"] = ec.max_abs_diff(p_on_1, p_on_2)
            rec["t_308_kv_off_s"] = round(t_off, 2); rec["t_308_kv_on_first_s"] = round(t_on1, 2); rec["t_308_kv_on_second_s"] = round(t_on2, 2)
            print(f"② kv_cache 开/关预测最大绝对差 {rec['kv_cache_max_abs_diff']:.3g}；308 例耗时 关 {t_off:.1f} s、开(首次) {t_on1:.1f} s、开(再次) {t_on2:.1f} s")
            set_kv([sub0], kv, kv_initial)
        except Exception as exc:  # noqa: BLE001
            rec["kv_cache_error"] = f"{type(exc).__name__}: {exc}"
            print(f"② kv_cache 切换失败：{rec['kv_cache_error']}")
            try:
                set_kv([sub0], kv, kv_initial)
            except Exception:  # noqa: BLE001
                pass
    else:
        print("② 子模型没有 kv_cache 类参数，跳过")

    # ③ 50 行与 2,000 行查询的耗时（当前设置）
    f0(data.X_test[:5])  # 预热
    _, t50 = timed(f0, data.X_test[:50])
    X2000 = np.tile(data.X_test, (7, 1))[:2000]
    _, t2000 = timed(f0, X2000)
    rec["t_query_50_s"] = round(t50, 2); rec["t_query_2000_s"] = round(t2000, 2)
    print(f"③ 一次调用：50 行 {t50:.2f} s，2,000 行 {t2000:.2f} s（当前 kv 设置 {sub0.get_params().get(kv) if kv else '无'}）")

    # ④ 记忆压扁证据：各子模型对开发集前 100 行与测试集的准确率 / AUROC
    mem = []
    for l, (sub, name) in enumerate(zip(subs, data.label_names)):
        f = f_for(sub)
        pd_ = f(data.X_dev[:100]); yd = data.Y_dev[:100, l]
        pt_ = f(data.X_test); yt = data.Y_test[:, l]
        row = {"label": name, "dev100_acc": float(((pd_ >= 0.5) == yd).mean()), "test_acc": float(((pt_ >= 0.5) == yt).mean()),
               "dev100_pos": int(yd.sum()), "test_pos": int(yt.sum())}
        for key, (p_, y_) in (("dev100_auroc", (pd_, yd)), ("test_auroc", (pt_, yt))):
            row[key] = float(roc_auc_score(y_, p_)) if 0 < y_.sum() < len(y_) else None
        mem.append(row)
    rec["memorization"] = mem
    print("④ 开发集前 100 行 vs 测试集（准确率 / AUROC）：")
    for r in mem:
        print(f"   {r['label']}：dev100 {r['dev100_acc']:.3f} / {r['dev100_auroc'] if r['dev100_auroc'] is None else round(r['dev100_auroc'], 3)}，test {r['test_acc']:.3f} / {r['test_auroc'] if r['test_auroc'] is None else round(r['test_auroc'], 3)}")

    # KernelSHAP 单例计时（证素 0）：M 的分布、中位数例与最多例各一次
    M = varying_counts(data.X_test, ref)
    rec["varying_features_M"] = {"min": int(M.min()), "median": float(np.median(M)), "max": int(M.max()),
                                 "n_exact_enumeration(M<=11)": int((M <= 11).sum()), "n_cases": int(len(M))}
    ref_df = pd.DataFrame([ref], columns=data.feature_names)
    ex = shap.KernelExplainer(f0, ref_df)
    i_med = int(np.argsort(M)[len(M) // 2]); i_max = int(np.argmax(M))
    for tag, i in (("median", i_med), ("max", i_max)):
        sv, t = timed(ex.shap_values, data.X_test[i:i + 1], nsamples="auto", l1_reg=False, silent=True)
        sv = np.asarray(sv).reshape(-1)
        add_err = float(abs(sv.sum() + float(np.ravel(ex.expected_value)[0]) - f0(data.X_test[i:i + 1])[0]))
        rec[f"kernelshap_one_case_{tag}"] = {"case_id": i, "M": int(M[i]), "seconds": round(t, 2), "additivity_err": add_err}
        print(f"KernelSHAP 单例（{tag}，M={M[i]}）：{t:.1f} s，可加性误差 {add_err:.2e}")
    est_h = (rec["kernelshap_one_case_median"]["seconds"] * len(M) * len(subs)) / 3600
    rec["estimated_total_hours_at_median"] = round(est_h, 2)
    print(f"按中位数例估计全量：308 × 11 ≈ {est_h:.1f} 小时")
    P_after = proba_full(m, data.X_test)
    rec["predict_proba_identical_after_probe"] = bool(np.array_equal(P, P_after))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"探针记录：{OUT / 'probe.json'}")
    return rec


# ---------------- 正式 ----------------

def run(m, subs, data: ec.Data, ref: np.ndarray, saved: dict, model_path: Path, labels: list[int], kv_mode: str) -> int:
    import shap

    kv = kv_param_name(subs[0])
    kv_initial = subs[0].get_params().get(kv) if kv else None
    if kv_mode == "auto":
        pj = OUT / "probe.json"
        use_kv = False
        if pj.exists():
            pr = json.loads(pj.read_text(encoding="utf-8"))
            d = pr.get("kv_cache_max_abs_diff")
            use_kv = d is not None and d < 1e-4
        print(f"kv_cache：auto → {'开' if use_kv else '关'}（探针差值判定 < 1e-4 才开）")
    else:
        use_kv = kv_mode == "on"
    set_kv(subs, kv, use_kv)

    P_before = proba_full(m, data.X_test)
    fid = ec.max_abs_diff(P_before, saved["y_prob"])
    print(f"重建模型 predict_proba 与存档 y_prob 最大绝对差：{fid:.3g}（阈值 1e-5）")
    if fid > 1e-5:
        print("保真核对不通过，停止。")
        return 2
    ref_df = pd.DataFrame([ref], columns=data.feature_names)
    M = varying_counts(data.X_test, ref)
    PARTS.mkdir(parents=True, exist_ok=True)
    for l in labels:
        part = PARTS / f"label_{l:02d}.npz"
        if part.exists():
            print(f"证素 {data.label_names[l]}：已有 {part.name}，跳过")
            continue
        f = f_for(subs[l])
        ex = shap.KernelExplainer(f, ref_df)
        t0 = time.perf_counter()
        sv = np.asarray(ex.shap_values(data.X_test, nsamples="auto", l1_reg=False, silent=True), dtype=np.float64)
        secs = time.perf_counter() - t0
        ev = float(np.ravel(ex.expected_value)[0])
        fx = f(data.X_test)
        add_err = float(np.abs(sv.sum(axis=1) + ev - fx).max())
        np.savez_compressed(part, shap_values=sv, expected_value=np.array(ev), f_x=fx, seconds=np.array(secs),
                            additivity_max_err=np.array(add_err), label=np.array(data.label_names[l]))
        print(f"证素 {data.label_names[l]}：{secs / 60:.1f} 分钟，可加性误差 {add_err:.2e}，{ec.now_beijing()}")
    done = [PARTS / f"label_{l:02d}.npz" for l in range(len(data.label_names))]
    if not all(p.exists() for p in done):
        print(f"已完成 {sum(p.exists() for p in done)}/{len(done)} 个证素；全部齐后再运行一次汇总。")
        set_kv(subs, kv, kv_initial)
        return 0
    parts = [np.load(p) for p in done]
    shap_vals = np.stack([p["shap_values"] for p in parts], axis=-1)  # (N, F, L)
    evs = np.array([float(p["expected_value"]) for p in parts])
    fxs = np.column_stack([p["f_x"] for p in parts])
    add_errs = {data.label_names[l]: float(p["additivity_max_err"]) for l, p in enumerate(parts)}
    secs = {data.label_names[l]: float(p["seconds"]) for l, p in enumerate(parts)}
    fx_vs_saved = ec.max_abs_diff(fxs, saved["y_prob"])
    P_after = proba_full(m, data.X_test)
    identical_after = bool(np.array_equal(P_before, P_after))
    np.savez_compressed(OUT / "shap_values.npz", shap_values=shap_vals, expected_value=evs, case_id=np.array(data.case_ids),
                        feature_names=np.array(data.feature_names), label_names=np.array(data.label_names))
    signed = shap_vals.mean(axis=0)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top = ec.write_summary_tables(signed, mean_abs, data, OUT)
    print("正向前 5（带符号均值 SHAP）：")
    ec.print_top5(top)
    check = {
        "rebuild_max_abs_diff": fid, "f_x_vs_saved_y_prob_max_abs_diff": fx_vs_saved,
        "additivity_max_err_by_label": add_errs, "additivity_max_err": max(add_errs.values()),
        "seconds_by_label": secs, "runtime_s": round(sum(secs.values()), 1),
        "kv_cache_param": kv, "kv_cache_used": bool(use_kv) if kv else None,
        "nsamples": "auto", "l1_reg": False, "background": "reference_row (1 row)",
        "varying_features_M": {"min": int(M.min()), "median": float(np.median(M)), "max": int(M.max())},
        "exact_enumeration_cases(M<=11)": int((M <= 11).sum()),
        "predict_proba_identical_after_explain": identical_after,
        "run_seed_from_rebuild": None, "date": ec.now_beijing(),
    }
    rb = model_path.parent.parent / "rebuild.json"
    if rb.exists():
        rj = json.loads(rb.read_text(encoding="utf-8"))
        check["run_seed_from_rebuild"] = rj.get("run_seed")
        check["wrapper_random_state"] = rj.get("wrapper_random_state")
        check["rebuild_comparison"] = rj.get("comparison")
    (OUT / "rebuild_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    sub_params = subs[0].get_params()
    ec.update_params("tabicl", {
        "method": "kernel_shap", "background": "reference_row", "nsamples": "auto", "l1_reg": False,
        "n_estimators": sub_params.get("n_estimators"), "random_state": sub_params.get("random_state"),
        "run_seed_from_rebuild": check["run_seed_from_rebuild"],
        "rebuild_max_abs_diff": fid, "kv_cache": check["kv_cache_used"], "kv_cache_param": kv,
        "kv_cache_max_abs_diff": (json.loads((OUT / "probe.json").read_text(encoding="utf-8")).get("kv_cache_max_abs_diff")
                                  if (OUT / "probe.json").exists() else None),
        "exact_enumeration_cases": check["exact_enumeration_cases(M<=11)"], "additivity_max_err": check["additivity_max_err"],
        "model_path": str(model_path), "model_sha256": ec.sha256_file(model_path), "seed": 0,
        "predict_proba_identical_after_explain": identical_after, "runtime_s": check["runtime_s"],
        "outputs": ["shap_values.npz", "mean_abs.csv", "mean_signed.csv", "top5_by_label.csv", "rebuild_check.json", "probe.json"],
    }, data=data, reference_path=ec.EXPLAIN_DIR / "reference_row.json")
    set_kv(subs, kv, kv_initial)
    print(f"产出：{OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--labels", default=None, help="证素下标，逗号分隔；缺省全部")
    ap.add_argument("--kv-cache", default="auto", choices=["auto", "on", "off"])
    args = ap.parse_args()
    data = ec.load_data(args.data)
    ref, _ = ec.write_reference_row(data)
    model_path = Path(args.model).resolve()
    saved = ec.load_final_npz(Path(args.npz).resolve(), data)
    m, subs = load_model(model_path)
    if len(subs) != len(data.label_names):
        raise RuntimeError(f"子模型 {len(subs)} 个，证素 {len(data.label_names)} 个")
    if args.probe:
        probe(m, subs, data, ref, saved, model_path)
        return 0
    labels = list(range(len(data.label_names))) if not args.labels else [int(x) for x in args.labels.split(",")]
    return run(m, subs, data, ref, saved, model_path, labels, args.kv_cache)


if __name__ == "__main__":
    sys.exit(main())
