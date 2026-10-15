# -*- coding: utf-8 -*-
"""DANNV2 参考实现（定稿配方 lean，2026-08-19 固化）。

本文件是论文方法的唯一权威实现，自包含、无历史开关、无消融旁路。
与探针版（mech_probe.EGMEBN 的 lean 路径）在 seed0-3 共 276 片上
逐位对拍一致（验证脚本 experiments/probes/verify_dannv2.py）。

方法由三个大组件构成，全部参数全局固定，免调参。

大组件 A 证据分段编码
  每特征用贪心基尼分箱在训练折上学出至多 16 段折点（证据分离度
  极大化，零外部学习器依赖），标量展开为分段线性向量（段内相对
  位置、之下 1、之上 0），拼接原值直通列。不做 QT，不做行归一。

大组件 B 先验分歧集成
  共享 MLP 主干（深 2 宽 768，ReLU，Dropout 0.10）上并置 32 个成员，
  输入端每成员一组固定符号调制（翻转概率 0.25），成员再各带一条
  类先验偏置 tau_j·log(pi)（tau 取等距谱，谱宽 1.5，只进训练损失），
  全局标量闭式温度校准，推理为各成员 softmax 概率平均。

大组件 C 两段自蒸馏
  段一在训练折内做三折交叉验证（教师谱心 0.25），产出验证曲线
  极大值位置 E* 与全体折外预测 Q。段二在完整训练折上训练
  round(E* x 1.5) 轮（学生谱心 0.0625），软目标为 Q 的锐化版
  （指数 2.0）与硬标签按 0.2 混合，参数以 EMA 0.95 交付。

用法（单片）：
  python dannv2.py <数据集名> <片号>
  片号 0..29 对应 seed0-9 x 外层 3 折，数据在 data/singlelabel/。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------- 定稿常量（全局固定，免调参） ----------------
BINS = 16                 # A 每特征分段数上限
HIDDEN, DEPTH = 768, 2    # B 主干宽度与深度
DROPOUT, LR, WD = 0.10, 0.001, 0.0002
K, FLIP_P = 32, 0.25      # B 成员数与符号翻转概率
TAU_TEA = (-0.5, 1.0)         # 教师 tau 谱（谱心 0.25，谱宽 1.5）
TAU_STU = (-0.6875, 0.8125)   # 学生 tau 谱（谱心 0.0625，谱宽 1.5）
EMA_DECAY = 0.95          # C 交付权重的指数滑动平均
N_FOLD = 3                # C 段一内层折数
MULT = 1.5                # C 段二轮数放大倍数
ALPHA, SHARP = 0.2, 2.0   # C 软目标混合与锐化指数
ES_PATIENCE, ES_MAX_EP = 16, 1000   # 段一早停口径


def default_batch_pin(n_train, s=16, lo=32, hi=1024):
    """批大小按训练样本数定档（协议钦定，batch 不进任何搜索空间）。"""
    import math
    b = 2 ** max(0, math.floor(math.log2(max(1, n_train) / s)))
    return max(1, min(max(lo, min(hi, b)), n_train))


# ---------------- 大组件 A 证据分段编码 ----------------

EPS = 1e-12


def _best_split(xs, ys, L, min_leaf):
    """已排序一维子段上的最优基尼分割，返回 (增益, 阈值, 左段末位)。"""
    n = len(xs)
    if n < 2 * min_leaf:
        return 0.0, None, None
    oh = np.zeros((n, L))
    oh[np.arange(n), ys] = 1.0
    cs = np.cumsum(oh, axis=0)
    tot = cs[-1]
    nt = tot.sum()
    g_par = nt - (tot ** 2).sum() / max(nt, EPS)
    ok = np.flatnonzero(np.diff(xs) > EPS)
    ok = ok[(ok + 1 >= min_leaf) & (n - ok - 1 >= min_leaf)]
    if len(ok) == 0:
        return 0.0, None, None
    left = cs[ok]
    right = tot[None, :] - left
    nl = left.sum(1)
    nr = right.sum(1)
    gl = nl - (left ** 2).sum(1) / np.maximum(nl, EPS)
    gr = nr - (right ** 2).sum(1) / np.maximum(nr, EPS)
    gain = g_par - (gl + gr)
    j = int(np.argmax(gain))
    if gain[j] <= EPS:
        return 0.0, None, None
    i = int(ok[j])
    return float(gain[j]), float((xs[i] + xs[i + 1]) / 2.0), i


def greedy_edges(xf, yi, bins=BINS, min_leaf=None):
    """一维 best-first 贪心基尼分箱（证据分离度极大化）。

    与 sklearn 单特征决策树（max_leaf_nodes=bins）的折点等价，
    零外部学习器依赖。固定规则，无超参、无随机性。"""
    xf = np.asarray(xf, dtype=np.float64)
    yi = np.asarray(yi)
    n = len(xf)
    if min_leaf is None:
        min_leaf = max(2, n // 64)
    L = int(yi.max()) + 1
    o = np.argsort(xf, kind="mergesort")
    xs, ys = xf[o], yi[o]
    segs = []
    g, thr, i = _best_split(xs, ys, L, min_leaf)
    if thr is not None:
        segs.append((g, thr, 0, n, i))
    out = []
    while segs and len(out) < bins - 1:
        segs.sort(key=lambda t: -t[0])
        g, thr, a, b, i = segs.pop(0)
        out.append(thr)
        for a2, b2 in ((a, i + 1), (i + 1, b)):
            if b2 - a2 < 2 * min_leaf:
                continue
            g2, t2, i2 = _best_split(xs[a2:b2], ys[a2:b2], L, min_leaf)
            if t2 is not None:
                segs.append((g2, t2, a2, b2, a2 + i2))
    return np.unique(np.asarray(out, dtype=np.float64))


class SegEncoder:
    """证据分段编码器（A）：贪心基尼折点 + 分段线性展开 + 原值直通。"""

    def fit(self, X, yi):
        n, F = X.shape
        self.edges_ = []
        for f in range(F):
            xf = X[:, f]
            lo, hi = float(xf.min()), float(xf.max())
            if hi - lo <= 1e-12:
                self.edges_.append(None)          # 常量特征输出单列 0
                continue
            th = np.unique(greedy_edges(xf, yi, BINS,
                                        max(2, n // 64)))
            th = th[(th > lo) & (th < hi)]
            edges = np.concatenate([[lo], th, [hi]])
            edges = edges[np.concatenate([[True],
                                          np.diff(edges) > 1e-12])]
            self.edges_.append(edges if len(edges) >= 2 else None)
        self.dim_ = int(sum(1 if e is None else len(e) - 1
                            for e in self.edges_) + F)
        return self

    def transform(self, X):
        cols = []
        for f, edges in enumerate(self.edges_):
            xf = X[:, f]
            if edges is None:
                cols.append(np.zeros((len(xf), 1)))
                continue
            a, b = edges[:-1], edges[1:]
            v = (xf[:, None] - a[None, :]) / (b - a)[None, :]
            cols.append(np.clip(v, 0.0, 1.0))
        cols.append(np.asarray(X, dtype=np.float64))   # 原值直通列
        return np.concatenate(cols, axis=1)


# ---------------- 大组件 B 先验分歧集成 ----------------

class _BEIn(nn.Module):
    """输入端逐成员符号调制：x (B,q) -> (B,K,q)。调制向量固定种子
    生成、可学、免权重衰减。"""

    def __init__(self, k, q, seed):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed) * 31 + 7)
        self.r = nn.Parameter(torch.where(
            torch.rand((k, q), generator=g) < FLIP_P, -1.0, 1.0))

    def forward(self, x):
        return x[:, None, :] * self.r[None, :, :]


class DANNV2Net:
    """先验分歧集成网络（B）。

    tau_spec 指定成员类先验偏置谱 (lo, hi)。early_va 非空时按验证宏F1
    早停（曲线与快照均用 EMA 权重，即交付口径），否则训满 epochs 轮
    并交付 EMA 权重。ext_q 非空时（段二）以其为软目标做自蒸馏。
    """

    def __init__(self, seed, tau_spec, epochs, batch_size,
                 early_va=None, ext_q=None):
        self.seed = int(seed)
        self.tau_spec = tau_spec
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.early_va = early_va
        self.ext_q = ext_q

    def fit(self, X, y):
        torch.set_num_threads(1)
        X = np.asarray(X, dtype=np.float64)
        self.classes_, yi = np.unique(y, return_inverse=True)
        L = len(self.classes_)
        # ---- A 编码 + 全局缩放（与原实现完全同序） ----
        self._enc = SegEncoder().fit(X, yi)
        X = self._enc.transform(X)
        self._min, self._max = X.min(0), X.max(0)
        rng = np.where(self._max - self._min > 0,
                       self._max - self._min, 1.0)
        self._rng = rng
        Xs = np.clip((X - self._min) / rng, 0.0, 1.0)
        Y = np.eye(L)[yi]
        q = X.shape[1]
        E = np.eye(q)
        s = float((Xs @ E).std())
        if s > 1e-12:
            E = E / s
        self._E = E
        Z = Xs @ E
        # ---- 随机单位方向锚（固定种子，与数据无关） ----
        Qr = np.random.default_rng(7).normal(size=(q, L))
        Q = Qr / np.maximum(np.linalg.norm(Qr, axis=0, keepdims=True),
                            1e-12)
        # ---- 主干构装（构装与 W0 映射的 RNG 顺序与原实现一致） ----
        torch.manual_seed(self.seed)
        layers = [_BEIn(K, q, self.seed),
                  nn.Linear(q, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT)]
        for _ in range(DEPTH - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                       nn.Dropout(DROPOUT)]
        self.emb = nn.Sequential(*layers)
        with torch.no_grad():   # train 态映射（dropout 生效，消耗全局 RNG）
            W0 = self.emb(torch.tensor(Q.T, dtype=torch.float))
        self._W = nn.Parameter(W0.permute(1, 0, 2).contiguous().clone())
        self._s0 = nn.Parameter(torch.ones(K))
        self._s1 = nn.Parameter(torch.tensor(0.0))
        self._r = torch.ones(L)

        def logits_of(zb):
            ek = self.emb(zb)                              # (B,K,H)
            wk = nn.functional.normalize(self._W, dim=2)   # (K,L,H)
            gk = self._s0[:, None] + self._s1 * self._r[None, :]
            return torch.einsum("bkd,kld->bkl", ek, wk) * gk[None, :, :]

        self._logits_of = logits_of
        zt = torch.tensor(Z, dtype=torch.float)
        yt = torch.tensor(yi, dtype=torch.long)
        # ---- 全局标量闭式温度（train 态定标，与原实现一致） ----
        with torch.no_grad():
            self._T = max(float(logits_of(zt).std()), 1e-3)
        # ---- 优化器（调制向量免权重衰减） ----
        params = (list(self.emb.parameters()) + [self._s0]
                  + [self._W] + [self._s1])
        rp = list(self.emb[0].parameters())
        rid = {id(p) for p in rp}
        opt = torch.optim.AdamW(
            [{"params": [p for p in params if id(p) not in rid],
              "weight_decay": WD},
             {"params": rp, "weight_decay": 0.0}], lr=LR)
        lossf = nn.CrossEntropyLoss()
        # ---- 成员类先验偏置谱（确定性统计量，只进训练损失） ----
        cnt = np.asarray(Y.sum(0), dtype=np.float64)
        pi = (cnt + 0.5) / float((cnt + 0.5).sum())
        tau = np.linspace(float(self.tau_spec[0]),
                          float(self.tau_spec[1]), K)
        btb = torch.tensor(tau[:, None] * np.log(pi)[None, :],
                           dtype=torch.float)
        # ---- 软目标（段二自蒸馏） ----
        yoh = torch.eye(L)[yt]
        ysoft = None
        if self.ext_q is not None:
            ysoft = ((1.0 - ALPHA) * yoh
                     + ALPHA * torch.tensor(np.asarray(self.ext_q),
                                            dtype=torch.float))
        # ---- 验证集准备（段一早停口径） ----
        va_z, yva = None, None
        if self.early_va is not None:
            from sklearn.metrics import f1_score as _f1
            Xva, yva = self.early_va
            Xva = self._enc.transform(np.asarray(Xva, dtype=np.float64))
            va_z = torch.tensor(
                np.clip((Xva - self._min) / self._rng, 0.0, 1.0)
                @ self._E, dtype=torch.float)
            self.va_curve_ = []
            es_best, es_state, es_wait = -1.0, None, 0
        # ---- EMA（零初始化 + 偏差校正） ----
        gen = torch.Generator().manual_seed(self.seed)

        def ema_named():
            yield from ((f"e.{k_}", v) for k_, v in
                        self.emb.state_dict().items())
            yield "_W", self._W.data
            yield "_s0", self._s0.data
            yield "_s1", self._s1.data

        ema_st = {k_: torch.zeros_like(v) for k_, v in ema_named()}
        ema_n = 0

        def ema_corr():
            c = 1.0 - EMA_DECAY ** max(ema_n, 1)
            return {k_: v / c for k_, v in ema_st.items()}

        def ema_swap(new):
            old = {k_: v.detach().clone() for k_, v in ema_named()}
            with torch.no_grad():
                for k_, v in ema_named():
                    v.copy_(new[k_])
            return old

        # ---- 训练循环 ----
        n_ep = ES_MAX_EP if self.early_va is not None else self.epochs
        self.emb.train()
        n_tr = zt.shape[0]
        bs = self.batch_size
        for _ep in range(n_ep):
            perm = (torch.randperm(n_tr, generator=gen)
                    if bs < n_tr else torch.arange(n_tr))
            for i0 in range(0, n_tr, bs):
                idx = perm[i0:i0 + bs]
                lgk = logits_of(zt[idx]) / self._T + btb[None, :, :]
                lgf = lgk.reshape(-1, L)
                if ysoft is not None:
                    loss = -(ysoft[idx].repeat_interleave(K, dim=0)
                             * torch.log_softmax(lgf, dim=1)
                             ).sum(1).mean()
                else:
                    loss = lossf(lgf, yt[idx].repeat_interleave(K))
                opt.zero_grad()
                loss.backward()
                opt.step()
            ema_n += 1
            with torch.no_grad():
                for k_, v in ema_named():
                    ema_st[k_].mul_(EMA_DECAY).add_(v, alpha=1.0
                                                    - EMA_DECAY)
            if va_z is not None:
                raw = ema_swap(ema_corr())     # 曲线与快照均用 EMA 权重
                self.emb.eval()
                with torch.no_grad():
                    pv = (torch.softmax(logits_of(va_z) / self._T, dim=2)
                          .mean(1).argmax(1).numpy())
                f1v = float(_f1(yva, self.classes_[pv], average="macro",
                                zero_division=0))
                self.va_curve_.append(f1v)
                self.emb.train()
                stop = False
                if f1v > es_best:
                    es_best, es_wait = f1v, 0
                    es_state = ({k_: v.detach().clone() for k_, v in
                                 self.emb.state_dict().items()},
                                self._W.detach().clone(),
                                self._s0.detach().clone())
                else:
                    es_wait += 1
                    if es_wait >= ES_PATIENCE:
                        stop = True
                ema_swap(raw)
                if stop:
                    break
        if va_z is None:
            ema_swap(ema_corr())               # 段二：交付 EMA 权重
        elif es_state is not None:             # 段一：恢复最优轮快照
            emb_sd, w_, s0_ = es_state
            self.emb.load_state_dict(emb_sd)
            with torch.no_grad():
                self._W.data.copy_(w_)
                self._s0.data.copy_(s0_)
        return self

    def predict_proba(self, X):
        """各成员 softmax(logit/T) 概率平均。推理不带类先验偏置。"""
        self.emb.eval()
        X = self._enc.transform(np.asarray(X, dtype=np.float64))
        z = torch.tensor(
            np.clip((X - self._min) / self._rng, 0.0, 1.0) @ self._E,
            dtype=torch.float)
        with torch.no_grad():
            return (torch.softmax(self._logits_of(z) / self._T, dim=2)
                    .mean(1).numpy())


# ---------------- 大组件 C 两段自蒸馏 ----------------

def fit_dannv2(X, y, tr, seed):
    """在训练折 tr 上完成两段流程，返回交付模型。

    段一，StratifiedKFold(3, seed) 三折教师（谱 TAU_TEA），
    E* = 三折验证曲线均值的极大值位置，Q = 全体折外预测。
    段二，round(E* x MULT) 轮学生（谱 TAU_STU），软目标为 Q 的
    SHARP 次幂锐化归一，EMA 交付。
    """
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    cls = np.unique(y[tr])
    L = len(cls)
    cv = StratifiedKFold(N_FOLD, shuffle=True, random_state=seed)
    Q = np.zeros((len(tr), L))
    curves = []
    for itr, iva in cv.split(X[tr], y[tr]):
        m = DANNV2Net(seed, TAU_TEA, ES_MAX_EP,
                      default_batch_pin(len(itr)),
                      early_va=(X[tr][iva], y[tr][iva]))
        m.fit(X[tr][itr], y[tr][itr])
        curves.append(list(m.va_curve_) or [0.0])
        P = m.predict_proba(X[tr][iva])
        for j, c in enumerate(m.classes_):
            Q[iva, int(np.where(cls == c)[0][0])] = P[:, j]
    lmin = max(1, min(len(c) for c in curves))
    E = int(np.argmax(np.mean([c[:lmin] for c in curves], axis=0))) + 1
    Qs = np.power(np.maximum(Q, 1e-12), SHARP)
    Qs = Qs / Qs.sum(1, keepdims=True)
    m2 = DANNV2Net(seed, TAU_STU, max(1, int(round(E * MULT))),
                   default_batch_pin(len(tr)), ext_q=Qs)
    m2.fit(X[tr], y[tr])
    m2.e_star_ = E
    return m2


def main():
    import sys
    from pathlib import Path
    if len(sys.argv) < 3:
        print(__doc__)
        return
    ds, k = sys.argv[1], int(sys.argv[2])
    root = Path(__file__).resolve().parent.parent / "data" / "singlelabel" / ds
    raw = np.load(root / "raw.npz", allow_pickle=True)
    fz = np.load(root / "folds.npz", allow_pickle=True)
    X = np.asarray(raw["X"], dtype=np.float64)
    y = np.asarray(raw["y"]).ravel()
    tr, te = fz[f"train_{k}"], fz[f"test_{k}"]
    seed = int(fz["meta"][k][0])
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        import os
        os.environ.setdefault(v, "1")
    m = fit_dannv2(X, y, tr, seed)
    P = m.predict_proba(X[te])
    pred = np.asarray(m.classes_)[P.argmax(1)]
    from sklearn.metrics import f1_score, roc_auc_score
    f1 = f1_score(y[te], pred, average="macro", zero_division=0)
    acc = float((pred == y[te]).mean())
    print(f"{ds} k{k} seed{seed}  E*={m.e_star_}  "
          f"宏F1={f1:.4f}  准确率={acc:.4f}")


if __name__ == "__main__":
    main()
