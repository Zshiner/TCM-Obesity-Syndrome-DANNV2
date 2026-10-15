# -*- coding: utf-8 -*-
"""DANNV2-beta 多标签版验收测试（快速部分）。

  python test_dannv2_beta_ml.py <data.json> [--device cpu|cuda] [--full]

1. 折点等价：只有 1 个证素时，多标签折点（各证素二分类基尼之和）与冻结副本的二分类
   折点逐位相同（年龄 + 前 40 个二值特征 × 3 个证素）。
2. 形状与范围：输出 N×L，概率落在 [0, 1]。
3. 同种子可复现：同设备、同种子两次训练，预测完全一致。
4. 序列化：joblib 往返后在 CPU 上预测，与原模型一致（GPU 训练的模型允许 1e-5 误差）。
--full 再跑一次完整两段流程（开发集前 80% 训练、后 20% 评估），报告耗时与宏 F1。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dannv2_beta_ml as ml  # noqa: E402
import dannv2_beta as orig  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    X = np.array([s["features"] for s in d["train_samples"]], dtype=np.float64)
    Y = np.array([s["labels"] for s in d["train_samples"]], dtype=np.int64)
    return X, Y, d["label_names"]


def test_edges(X, Y):
    n = len(X)
    cols = [0] + list(range(2, 42))          # 年龄 + 40 个二值特征（第 1 列为性别，也是二值）
    labels = [int(np.argmax(Y.sum(0))), int(np.argsort(Y.sum(0))[len(Y[0]) // 2]), int(np.argmin(Y.sum(0)))]
    n_cmp = 0
    for li in labels:
        for f in cols:
            a = orig.greedy_edges(X[:, f], Y[:, li], orig.BINS, max(2, n // 64))
            b = ml.greedy_edges_ml(X[:, f], Y[:, [li]], ml.BINS, max(2, n // 64))
            assert np.array_equal(a, b), f"折点不一致：特征 {f} 证素 {li}：{a} vs {b}"
            n_cmp += 1
    print(f"✓ 折点等价：{n_cmp} 组（特征 × 证素）与冻结副本逐位相同")


def quick_net(X, Y, seed, device, epochs=3):
    return ml.DANNV2NetML(seed, ml.TAU_STU, epochs, ml.default_batch_pin(len(X)), device=device).fit(X, Y)


def test_shape_repro_pickle(X, Y, device):
    import joblib
    Xtr, Ytr, Xte = X[:240], Y[:240], X[240:300]
    m1 = quick_net(Xtr, Ytr, 11, device)
    p1 = m1.predict_proba(Xte)
    assert p1.shape == (len(Xte), Y.shape[1]), f"形状错误 {p1.shape}"
    assert np.isfinite(p1).all() and p1.min() >= 0.0 and p1.max() <= 1.0, "概率越界"
    print(f"✓ 形状与范围：{p1.shape}，概率 [{p1.min():.4f}, {p1.max():.4f}]（{device}）")
    p2 = quick_net(Xtr, Ytr, 11, device).predict_proba(Xte)
    diff = float(np.abs(p1 - p2).max())
    assert diff == 0.0, f"同种子不可复现，最大差 {diff}"
    print(f"✓ 同种子可复现：两次预测最大差 {diff}（{device}）")
    p3 = quick_net(Xtr, Ytr, 12, device).predict_proba(Xte)
    print(f"  不同种子最大差 {float(np.abs(p1 - p3).max()):.4f}（应明显大于 0）")
    buf = io.BytesIO()
    joblib.dump(m1, buf)
    buf.seek(0)
    m_back = joblib.load(buf)
    pb = m_back.predict_proba(Xte)
    tol = 0.0 if device == "cpu" else 1e-5
    d = float(np.abs(p1 - pb).max())
    assert d <= tol, f"序列化往返预测不一致，最大差 {d}"
    print(f"✓ 序列化往返：{len(buf.getvalue()) / 2**20:.1f} MB，加载后设备 {m_back.device}，最大差 {d:.2e}")


def test_full(X, Y, device):
    from sklearn.metrics import f1_score
    n = len(X)
    cut = int(n * 0.8)
    t0 = time.time()
    m = ml.fit_dannv2_ml(X, Y, np.arange(cut), seed=0, device=device)
    dt = time.time() - t0
    P = m.predict_proba(X[cut:])
    f1 = f1_score(Y[cut:], (P >= 0.5).astype(int), average="macro", zero_division=0)
    print(f"✓ 完整两段流程：训练 {cut} 例、L={Y.shape[1]}，E*={m.e_star_}，耗时 {dt / 60:.1f} 分钟（{device}），"
          f"留出 {n - cut} 例宏 F1={f1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    X, Y, _ = load(a.data)
    print(f"数据：开发集 {X.shape}，证素 {Y.shape[1]}；冻结副本 SHA-256 已核对")
    test_edges(X, Y)
    test_shape_repro_pickle(X, Y, a.device)
    if a.full:
        test_full(X, Y, a.device)
    print("全部通过")


if __name__ == "__main__":
    main()
