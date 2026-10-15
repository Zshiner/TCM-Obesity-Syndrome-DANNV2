# -*- coding: utf-8 -*-
"""DANNV2-beta 原生多标签版（体重管理论文专用，2026-10-02）。

在冻结副本 dannv2_beta.py 旁另建，冻结副本一字不改：本文件先核对其 SHA-256，
再直接复用其中全部定稿常量与未改动组件（输入端符号调制、批大小定档、分段
编码的 transform）。只改写《论文方案》v0.5 第 4.3 节定下的 5 处：

  1. 读标签：直接接收 N×L 的 0/1 矩阵（原：np.unique(y) 当一列类别号）。
  2. 证据分段编码的折点：各证素二分类基尼之和（原：单一类别号的多类基尼）。
     L=1 时与原版二分类基尼逐位相同。
  3. 损失与输出：BCE + sigmoid，各证素独立；早停用多标签宏 F1（阈值 0.5）；
     输出 N×L 概率 = 各成员 sigmoid(logit/T) 的平均（原：CE + softmax、argmax 宏 F1）。
  4. 类先验偏置：tau·log[(n⁺+0.5)/(n⁻+0.5)]，即原版二分类先验的对数几率形式
     （原：tau·log pi，pi 为类别分布）。
  5. 两段自蒸馏：段一用迭代分层三折（固定种子）；软目标逐证素锐化
     q^2/(q^2+(1−q)^2) 后与硬标签按 0.2 混合（原：单标签分层三折、按类归一）。

所有证素共享编码与主干、联合训练。其余定稿值一律沿用：主干 768×2、32 成员、
翻转 0.25、谱心与谱宽、温度定标、EMA 0.95、轮数放大 1.5、软目标混合 0.2、
锐化指数 2、批大小定档、早停 patience 16。

与配方无关的工程适配（不改变任何定稿值）：
  - 支持 device（CPU 或 GPU）。参数初始化与原版同序、在 CPU 上完成后再搬到 device；
  - 原版在 fit 内定义的闭包改为方法，交付模型可被 joblib/pickle 序列化，
    序列化时自动转回 CPU；
  - 线程数可设（默认 1，同原版）。
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
FROZEN_FILE = _HERE / "dannv2_beta.py"
FROZEN_SHA256 = "ce8c912de29e05f8a611ab0b78c2a1e5e70af0a7d40ad8d9dee54953cbf66fc8"

_sha = hashlib.sha256(FROZEN_FILE.read_bytes()).hexdigest()
if _sha != FROZEN_SHA256:
    raise RuntimeError(
        f"冻结副本 {FROZEN_FILE} 的 SHA-256 不符（{_sha}），拒绝加载。"
        "dannv2_beta.py 必须与 DANN2 定稿源文件逐字节一致，见同目录 SOURCE.md。")
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import dannv2_beta as _F  # noqa: E402  冻结副本（只读复用）

# ---------------- 定稿常量：全部取自冻结副本 ----------------
BINS, HIDDEN, DEPTH = _F.BINS, _F.HIDDEN, _F.DEPTH
DROPOUT, LR, WD = _F.DROPOUT, _F.LR, _F.WD
K, FLIP_P = _F.K, _F.FLIP_P
TAU_TEA, TAU_STU = _F.TAU_TEA, _F.TAU_STU
EMA_DECAY, N_FOLD, MULT = _F.EMA_DECAY, _F.N_FOLD, _F.MULT
ALPHA, SHARP = _F.ALPHA, _F.SHARP
ES_PATIENCE, ES_MAX_EP = _F.ES_PATIENCE, _F.ES_MAX_EP
EPS = _F.EPS
default_batch_pin = _F.default_batch_pin
_BEIn = _F._BEIn

ML_THRESHOLD = 0.5     # 早停口径：多标签宏 F1 的判定阈值


# ---------------- 改写 2：证据分段编码的折点 ----------------

def _gini_sum(pos, cnt):
    """各证素二分类基尼之和。pos (m, L) 为阳性数，cnt (m,) 为样本数。"""
    cnt = cnt[:, None]
    neg = cnt - pos
    return (cnt - (pos ** 2 + neg ** 2) / np.maximum(cnt, EPS)).sum(1)


def _best_split_ml(xs, ys, min_leaf):
    """已排序一维子段上的最优分割（各证素二分类基尼之和），返回 (增益, 阈值, 左段末位)。"""
    n = len(xs)
    if n < 2 * min_leaf:
        return 0.0, None, None
    cs = np.cumsum(ys, axis=0)
    tot = cs[-1]
    g_par = float(_gini_sum(tot[None, :], np.array([float(n)]))[0])
    ok = np.flatnonzero(np.diff(xs) > EPS)
    ok = ok[(ok + 1 >= min_leaf) & (n - ok - 1 >= min_leaf)]
    if len(ok) == 0:
        return 0.0, None, None
    nl = (ok + 1).astype(np.float64)
    left = cs[ok]
    right = tot[None, :] - left
    gain = g_par - (_gini_sum(left, nl) + _gini_sum(right, n - nl))
    j = int(np.argmax(gain))
    if gain[j] <= EPS:
        return 0.0, None, None
    i = int(ok[j])
    return float(gain[j]), float((xs[i] + xs[i + 1]) / 2.0), i


def greedy_edges_ml(xf, Y, bins=BINS, min_leaf=None):
    """一维 best-first 贪心分箱（与原版同一流程，增益换成各证素二分类基尼之和）。"""
    xf = np.asarray(xf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(xf)
    if min_leaf is None:
        min_leaf = max(2, n // 64)
    o = np.argsort(xf, kind="mergesort")
    xs, ys = xf[o], Y[o]
    segs = []
    g, thr, i = _best_split_ml(xs, ys, min_leaf)
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
            g2, t2, i2 = _best_split_ml(xs[a2:b2], ys[a2:b2], min_leaf)
            if t2 is not None:
                segs.append((g2, t2, a2, b2, a2 + i2))
    return np.unique(np.asarray(out, dtype=np.float64))


class SegEncoderML(_F.SegEncoder):
    """证据分段编码器：fit 用多标签折点，transform 原样继承冻结副本。"""

    def fit(self, X, Y):
        n, F = X.shape
        self.edges_ = []
        for f in range(F):
            xf = X[:, f]
            lo, hi = float(xf.min()), float(xf.max())
            if hi - lo <= 1e-12:
                self.edges_.append(None)
                continue
            th = np.unique(greedy_edges_ml(xf, Y, BINS, max(2, n // 64)))
            th = th[(th > lo) & (th < hi)]
            edges = np.concatenate([[lo], th, [hi]])
            edges = edges[np.concatenate([[True], np.diff(edges) > 1e-12])]
            self.edges_.append(edges if len(edges) >= 2 else None)
        self.dim_ = int(sum(1 if e is None else len(e) - 1 for e in self.edges_) + F)
        return self


# ---------------- 改写 1、3、4：先验分歧集成（多标签） ----------------

class DANNV2NetML:
    """先验分歧集成网络（多标签）。参数含义同冻结副本 DANNV2Net，另加 device、num_threads。"""

    def __init__(self, seed, tau_spec, epochs, batch_size, early_va=None, ext_q=None,
                 device="cpu", num_threads=1):
        self.seed = int(seed)
        self.tau_spec = tau_spec
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.early_va = early_va
        self.ext_q = ext_q
        self.device = str(device)
        self.num_threads = int(num_threads)

    # 原版 fit 内的闭包 logits_of，改为方法（计算完全相同）
    def _logits(self, zb):
        ek = self.emb(zb)                                   # (B,K,H)
        wk = nn.functional.normalize(self._W, dim=2)        # (K,L,H)
        gk = self._s0[:, None] + self._s1 * self._r[None, :]
        return torch.einsum("bkd,kld->bkl", ek, wk) * gk[None, :, :]

    def _to_z(self, X):
        X = self._enc.transform(np.asarray(X, dtype=np.float64))
        z = np.clip((X - self._min) / self._rng, 0.0, 1.0) @ self._E
        return torch.tensor(z, dtype=torch.float, device=self.device)

    def fit(self, X, Y):
        torch.set_num_threads(self.num_threads)
        dev = torch.device(self.device)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)                 # 改写 1：N×L 0/1 矩阵
        if Y.ndim != 2:
            raise ValueError("DANNV2-beta 多标签版需要 N×L 的 0/1 标签矩阵")
        L = Y.shape[1]
        self.n_labels_ = L
        # ---- A 编码 + 全局缩放（与原版同序） ----
        self._enc = SegEncoderML().fit(X, Y)                # 改写 2
        X = self._enc.transform(X)
        self._min, self._max = X.min(0), X.max(0)
        rng = np.where(self._max - self._min > 0, self._max - self._min, 1.0)
        self._rng = rng
        Xs = np.clip((X - self._min) / rng, 0.0, 1.0)
        q = X.shape[1]
        E = np.eye(q)
        s = float((Xs @ E).std())
        if s > 1e-12:
            E = E / s
        self._E = E
        Z = Xs @ E
        # ---- 随机单位方向锚（固定种子，与原版同） ----
        Qr = np.random.default_rng(7).normal(size=(q, L))
        Qa = Qr / np.maximum(np.linalg.norm(Qr, axis=0, keepdims=True), 1e-12)
        # ---- 主干构装：在 CPU 上按原版 RNG 顺序初始化，再搬到 device ----
        torch.manual_seed(self.seed)
        layers = [_BEIn(K, q, self.seed), nn.Linear(q, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT)]
        for _ in range(DEPTH - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT)]
        self.emb = nn.Sequential(*layers)
        with torch.no_grad():
            W0 = self.emb(torch.tensor(Qa.T, dtype=torch.float))
        self.emb.to(dev)
        self._W = nn.Parameter(W0.permute(1, 0, 2).contiguous().clone().to(dev))
        self._s0 = nn.Parameter(torch.ones(K, device=dev))
        self._s1 = nn.Parameter(torch.tensor(0.0, device=dev))
        self._r = torch.ones(L, device=dev)
        zt = torch.tensor(Z, dtype=torch.float, device=dev)
        yt = torch.tensor(Y, dtype=torch.float, device=dev)
        # ---- 全局标量闭式温度（train 态定标，同原版） ----
        with torch.no_grad():
            self._T = max(float(self._logits(zt).std()), 1e-3)
        # ---- 优化器（调制向量免权重衰减，同原版） ----
        params = list(self.emb.parameters()) + [self._s0] + [self._W] + [self._s1]
        rp = list(self.emb[0].parameters())
        rid = {id(p) for p in rp}
        opt = torch.optim.AdamW(
            [{"params": [p for p in params if id(p) not in rid], "weight_decay": WD},
             {"params": rp, "weight_decay": 0.0}], lr=LR)
        # ---- 改写 4：成员先验偏置 = tau·二分类先验对数几率（只进训练损失） ----
        npos = Y.sum(0)
        prior_logit = np.log((npos + 0.5) / ((len(Y) - npos) + 0.5))
        tau = np.linspace(float(self.tau_spec[0]), float(self.tau_spec[1]), K)
        btb = torch.tensor(tau[:, None] * prior_logit[None, :], dtype=torch.float, device=dev)
        # ---- 改写 5（段二软目标）：与硬标签按 ALPHA 混合 ----
        target = yt
        if self.ext_q is not None:
            target = ((1.0 - ALPHA) * yt
                      + ALPHA * torch.tensor(np.asarray(self.ext_q), dtype=torch.float, device=dev))
        # ---- 验证集（段一早停口径：改写 3，多标签宏 F1） ----
        va_z = yva = None
        if self.early_va is not None:
            from sklearn.metrics import f1_score as _f1
            Xva, yva = self.early_va
            yva = np.asarray(yva).astype(int)
            va_z = self._to_z(Xva)
            self.va_curve_ = []
            es_best, es_state, es_wait = -1.0, None, 0
        # ---- EMA（零初始化 + 偏差校正，同原版） ----
        gen = torch.Generator().manual_seed(self.seed)

        def ema_named():
            yield from ((f"e.{k_}", v) for k_, v in self.emb.state_dict().items())
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
            perm = torch.randperm(n_tr, generator=gen) if bs < n_tr else torch.arange(n_tr)
            perm = perm.to(dev)
            for i0 in range(0, n_tr, bs):
                idx = perm[i0:i0 + bs]
                lgk = self._logits(zt[idx]) / self._T + btb[None, :, :]   # (B,K,L)
                tgt = target[idx][:, None, :].expand_as(lgk)               # 每成员同一目标
                loss = nn.functional.binary_cross_entropy_with_logits(lgk, tgt)  # 改写 3
                opt.zero_grad()
                loss.backward()
                opt.step()
            ema_n += 1
            with torch.no_grad():
                for k_, v in ema_named():
                    ema_st[k_].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
            if va_z is not None:
                raw = ema_swap(ema_corr())          # 曲线与快照均用 EMA 权重（同原版）
                self.emb.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(self._logits(va_z) / self._T).mean(1).cpu().numpy()
                f1v = float(_f1(yva, (pv >= ML_THRESHOLD).astype(int), average="macro",
                                zero_division=0))
                self.va_curve_.append(f1v)
                self.emb.train()
                stop = False
                if f1v > es_best:
                    es_best, es_wait = f1v, 0
                    es_state = ({k_: v.detach().clone() for k_, v in self.emb.state_dict().items()},
                                self._W.detach().clone(), self._s0.detach().clone())
                else:
                    es_wait += 1
                    if es_wait >= ES_PATIENCE:
                        stop = True
                ema_swap(raw)
                if stop:
                    break
        if va_z is None:
            ema_swap(ema_corr())                    # 段二：交付 EMA 权重
        elif es_state is not None:                  # 段一：恢复最优轮快照
            emb_sd, w_, s0_ = es_state
            self.emb.load_state_dict(emb_sd)
            with torch.no_grad():
                self._W.data.copy_(w_)
                self._s0.data.copy_(s0_)
        self.early_va = None                        # 训练数据不随模型保存
        self.ext_q = None
        return self

    def predict_proba(self, X):
        """各成员 sigmoid(logit/T) 概率平均，返回 N×L。推理不带先验偏置。"""
        self.emb.eval()
        with torch.no_grad():
            return torch.sigmoid(self._logits(self._to_z(X)) / self._T).mean(1).cpu().numpy()

    # ---- 序列化：一律转回 CPU，在无 GPU 的机器上也能加载 ----
    def __getstate__(self):
        st = self.__dict__.copy()
        if "emb" in st:
            st["emb"] = copy.deepcopy(self.emb).cpu()
            for k_ in ("_W", "_s0", "_s1"):
                st[k_] = nn.Parameter(getattr(self, k_).detach().cpu().clone())
            st["_r"] = self._r.detach().cpu().clone()
        st["device"] = "cpu"
        return st


# ---------------- 改写 5：两段自蒸馏（多标签） ----------------

def fit_dannv2_ml(X, Y, tr, seed, device="cpu", num_threads=1):
    """在训练折 tr 上完成两段流程，返回交付模型。

    段一：迭代分层三折（MultilabelStratifiedKFold，shuffle，random_state=seed；只有 1 个
    证素时退化为与冻结副本相同的 StratifiedKFold）教师
    （谱 TAU_TEA），E* = 三折验证曲线均值的极大值位置，Q = 全体折外预测（N×L）。
    段二：round(E* x MULT) 轮学生（谱 TAU_STU），软目标为 Q 逐证素锐化
    q^S/(q^S+(1−q)^S)（S=SHARP），与硬标签按 ALPHA 混合，EMA 交付。
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xtr, Ytr = X[tr], Y[tr]
    L = Ytr.shape[1]
    if L >= 2:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        splits = MultilabelStratifiedKFold(n_splits=N_FOLD, shuffle=True,
                                           random_state=int(seed)).split(Xtr, Ytr)
    else:
        # 只有 1 个证素时迭代分层退化为普通分层（迭代分层库不接受单列标签），
        # 即冻结副本所用的 StratifiedKFold(N_FOLD, shuffle=True, random_state=seed)
        from sklearn.model_selection import StratifiedKFold
        splits = StratifiedKFold(N_FOLD, shuffle=True, random_state=int(seed)).split(Xtr, Ytr[:, 0])
    Q = np.zeros((len(tr), L))
    curves = []
    for itr, iva in splits:
        m = DANNV2NetML(seed, TAU_TEA, ES_MAX_EP, default_batch_pin(len(itr)),
                        early_va=(Xtr[iva], Ytr[iva]), device=device, num_threads=num_threads)
        m.fit(Xtr[itr], Ytr[itr])
        curves.append(list(m.va_curve_) or [0.0])
        Q[iva] = m.predict_proba(Xtr[iva])
    lmin = max(1, min(len(c) for c in curves))
    e_star = int(np.argmax(np.mean([c[:lmin] for c in curves], axis=0))) + 1
    Qc = np.clip(Q, 1e-12, 1.0 - 1e-12)
    Qs = Qc ** SHARP / (Qc ** SHARP + (1.0 - Qc) ** SHARP)
    m2 = DANNV2NetML(seed, TAU_STU, max(1, int(round(e_star * MULT))), default_batch_pin(len(tr)),
                     ext_q=Qs, device=device, num_threads=num_threads)
    m2.fit(Xtr, Ytr)
    m2.e_star_ = e_star
    return m2


# ---------------- sklearn 式估计器（供 TCMHUBD 等框架调用） ----------------
# 定义在本模块（可正常 import 的顶层模块）而不是框架动态加载的适配器文件里：
# 框架以 tcmhubd_custom.ml.<文件名> 动态加载自定义模块且不登记父包，定义在那里的类
# 无法被 joblib 序列化（2026-10-02 冒烟实测）。

from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402


class DANNV2BetaMLClassifier(ClassifierMixin, BaseEstimator):
    """DANNV2-beta 多标签分类器：fit(X, Y[N×L]) → predict_proba(X) 返回 N×L 概率。

    random_state：每次拟合的种子（段一分折、参数初始化、调制向量、批次顺序共用）。
    device："cpu"、"cuda"、"cuda:N" 或 "auto"。
    """

    def __init__(self, *, device="cpu", random_state=0):
        self.device = device
        self.random_state = random_state

    def _resolved_device(self):
        d = "cpu" if self.device is None else str(self.device)
        if d == "auto":
            d = "cuda" if torch.cuda.is_available() else "cpu"
        if d.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device={d} 但当前环境没有可用的 CUDA（torch {torch.__version__}）")
        return d

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(y)
        if Y.ndim != 2:
            raise ValueError("DANNV2-beta 多标签版需要 N×L 的 0/1 标签矩阵")
        self.n_labels_ = int(Y.shape[1])
        self.model_ = fit_dannv2_ml(X, Y, np.arange(len(X)), seed=int(self.random_state) % (2 ** 32),
                                    device=self._resolved_device())
        self.e_star_ = int(self.model_.e_star_)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))

    def predict(self, X):
        return (self.predict_proba(X) >= ML_THRESHOLD).astype(int)
