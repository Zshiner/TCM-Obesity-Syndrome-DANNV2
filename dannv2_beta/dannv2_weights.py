# -*- coding: utf-8 -*-
"""DANNV2 原生权值（定稿 2026-10-11）。

只读已训练的 DANNV2 模型，给出模型自带的「特征 → 类别」权值。不参与训练，
不改 predict_proba，不改模型的参数、设备与精度（内部另取一份双精度只读副本）。
同一份代码适用于单标签版（dannv2.py，各成员 softmax 平均）与多标签版
（dannv2_beta_ml.py，各成员 sigmoid 平均）。

一、逐例权值 path_weights
  被分解的量是模型输出概率的对数几率
      D_c(x) = log[ P_c(x) / (1 − P_c(x)) ]，P 即 predict_proba 的输出。
  从参照样本 r 到样本 x，在原始特征空间走直线，每一步经组件 A 的分段编码
  进入网络，累加 ∇D · Δz，再把同一特征的分段列与原值直通列加在一起：
      w[i, f, c] = Σ_步 Σ_{j ∈ 特征 f 的编码列} ∂D_c/∂z_j · Δz_j
  性质：D_c(r) + Σ_f w[i, f, c] = D_c(x_i)，步数足够时成立，函数返回实际误差。
  单位是对数几率，正值支持该类，负值不支持。
  数学上这是以参照样本为基线、沿 A 编码路径的积分梯度，只用模型自身的前向
  与梯度，不调用任何外部解释工具。

二、逐段权值 path_weights 的 W_seg、segment_table
  组件 A 把每个特征切成若干段证据。同一条路径上，特征 f 的权值可以按
  「走过哪一段时累积的」拆开：
      w[i, f, c] = Σ_{s ∈ 特征 f 的各段} w_seg[i, (f, s), c]
  w_seg 等于该段分段列的累加量，加上原值直通列在走过该段时的累加量，拆分
  是精确的。它就是模型在这一段证据上给的权值。
  segment_table 把它汇总成每一段的边际权值：越过该段（由低到高）一整段时
  权值变化多少，同样乘 E / (E + k)。读法是温度计式累加，某取值相对参照值的
  权值等于两者之间各段边际权值之和，取值低于参照值时取负。

三、权值表 weight_table
      格值 = 出现时平均 × E / (E + k)，k = 1
  出现时平均：具有该特征取值的样本上，逐例权值的平均。
  E：这批样本里模型预计的该类例数。带着该特征算一次（Σ P），把该特征换成
     参照值再算一次，取大者。
  E 很大时系数接近 1，格值就是模型给的权值。E 不到 1 例时权值大半不计。
  相当于按预计例数做平均时多放一例权值为 0 的样本。全表不留空。

四、表的行 cut_order、value_rows
  取值不超过四种的特征逐值列出（参照取值照列，权值恒为 0）。其余特征按
  组件 A 最先接受的前三个折点分段。缺失标记单列一行。

五、排名 top_weights
  各类前 k 按绝对值排名，保留正负号。

单元测试见 test_dannv2_weights.py。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as Fn

# ---------------- 定稿常量 ----------------
STEPS = 1024          # 路径步数
PSEUDO_CASES = 1.0    # 权值表里多放的那一例（k）
N_CUTS = 3            # 连续特征的读数分段取 A 最先接受的前三个折点
MAX_LEVELS = 4        # 取值不超过四种的特征逐值列出


def net_of(model):
    """估计器（带 model_ 属性）取其网络，网络本身原样返回。"""
    return getattr(model, "model_", model)


class _ReadOnlyNet:
    """网络的双精度只读副本。原模型一字不动。"""

    def __init__(self, net, device, multilabel):
        self.enc = net._enc
        self.lo = np.asarray(net._min, dtype=np.float64)
        self.rng = np.asarray(net._rng, dtype=np.float64)
        self.E = np.asarray(net._E, dtype=np.float64)
        self.emb = copy.deepcopy(net.emb).to(device).double().eval()
        for p in self.emb.parameters():
            p.requires_grad_(False)
        self.W = net._W.detach().clone().to(device).double()
        self.s0 = net._s0.detach().clone().to(device).double()
        self.s1 = net._s1.detach().clone().to(device).double()
        self.r = net._r.detach().clone().to(device).double()
        self.T = float(net._T)
        self.device = device
        self.multilabel = bool(multilabel)

    def encode(self, X):
        """原始特征 → 网络输入（组件 A 编码、缩放、投影），与 predict_proba 同一路径。"""
        Xe = self.enc.transform(np.asarray(X, dtype=np.float64))
        return np.clip((Xe - self.lo) / self.rng, 0.0, 1.0) @ self.E

    def logodds(self, z):
        """网络输入 (n, q) → 模型输出概率的对数几率 (n, C)。"""
        ek = self.emb(z)                                         # (n, K, H)
        wk = Fn.normalize(self.W, dim=2)                         # (K, C, H)
        gk = self.s0[:, None] + self.s1 * self.r[None, :]
        lg = torch.einsum("bkd,kld->bkl", ek, wk) * gk[None, :, :] / self.T
        if self.multilabel:                                      # P = 各成员 sigmoid 的平均
            return (torch.logsumexp(Fn.logsigmoid(lg), 1)
                    - torch.logsumexp(Fn.logsigmoid(-lg), 1))
        lp = torch.log_softmax(lg, dim=2)                        # P = 各成员 softmax 的平均
        C = lp.shape[2]
        off = ~torch.eye(C, dtype=torch.bool, device=lp.device)
        lq = torch.stack([torch.logsumexp(lp[:, :, off[c]], 2)   # log(1 − p_kc)
                          for c in range(C)], 2)
        return torch.logsumexp(lp, 1) - torch.logsumexp(lq, 1)


def path_weights(model, X, ref, steps=STEPS, device=None, multilabel=None,
                 batch_size=2048, progress=None):
    """逐例权值。

    model       已训练的 DANNV2 模型（估计器或网络）
    X           (N, F) 原始特征
    ref         (F,) 参照样本
    steps       路径步数，定稿 1024
    device      缺省有 GPU 用 GPU
    multilabel  缺省自动判断（网络带 classes_ 属性的是单标签版）
    batch_size  每批样本数，只影响显存占用，不影响结果
    progress    可选回调 progress(已完成步数, 总步数)

    返回 dict：
      W        (N, F, C) 逐例权值，对数几率
      D        (N, C) 模型输出概率的对数几率
      D_ref    (C,) 参照样本的对数几率
      err_max, err_mean   |Σ_f W − (D − D_ref)| 的最大值与平均值
      W_seg    (N, S, C) 逐例逐段权值，S 为组件 A 的分段总数，同一特征各段之和等于 W
      seg_feature (S,) 每一段所属的特征列号
      seg_lo, seg_hi (S,) 每一段的下界与上界（训练时取值恒定的特征为 ∓inf）
      seg_err_max   |Σ_s W_seg − W| 的最大值
      steps, multilabel
    """
    net = net_of(model)
    if multilabel is None:
        multilabel = not hasattr(net, "classes_")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ro = _ReadOnlyNet(net, device, multilabel)
    X = np.asarray(X, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).ravel()
    N, F = X.shape
    if len(ref) != F:
        raise ValueError(f"参照样本有 {len(ref)} 列，X 有 {F} 列")
    cols, n_seg = [], 0                                          # 编码列布局：先各特征分段列，后 F 列原值直通
    for e in ro.enc.edges_:
        n_ = 1 if e is None else len(e) - 1
        cols.append(list(range(n_seg, n_seg + n_)))
        n_seg += n_
    agg = torch.zeros((n_seg + F, F), dtype=torch.float64, device=device)
    for f in range(F):
        agg[cols[f], f] = 1.0
        agg[n_seg + f, f] = 1.0
    if np.count_nonzero(ro.E - np.diag(np.diag(ro.E))):
        raise ValueError("网络输入的投影不是逐列缩放，编码列与网络输入列不再一一对应")
    seg_feature, seg_lo, seg_hi = [], [], []                     # 每个分段列属于哪个特征、对应哪个取值区间
    for f, e in enumerate(ro.enc.edges_):
        if e is None:                                            # 训练时取值恒定：只有一列，直通列的变化全记在它名下
            seg_feature.append(f)
            seg_lo.append(-np.inf)
            seg_hi.append(np.inf)
        else:
            for a_, b_ in zip(e[:-1], e[1:]):
                seg_feature.append(f)
                seg_lo.append(float(a_))
                seg_hi.append(float(b_))
    seg_feature, seg_lo, seg_hi = np.asarray(seg_feature), np.asarray(seg_lo), np.asarray(seg_hi)
    sf = torch.tensor(seg_feature, dtype=torch.long, device=device)
    sa = torch.tensor(seg_lo, device=device)
    sb = torch.tensor(seg_hi, device=device)
    with torch.no_grad():
        d_ref = ro.logodds(torch.tensor(ro.encode(ref[None]), device=device))[0].cpu().numpy()
    C = len(d_ref)
    W = np.zeros((N, F, C))
    W_seg = np.zeros((N, n_seg, C))
    D = np.zeros((N, C))
    ts = np.linspace(0.0, 1.0, steps + 1)
    n_batch = (N + batch_size - 1) // batch_size
    done = 0
    for a in range(0, N, batch_size):
        Xb = X[a:a + batch_size]
        B0 = np.tile(ref, (len(Xb), 1))
        dX = Xb - B0
        Wb = torch.zeros((len(Xb), F, C), dtype=torch.float64, device=device)
        Sb = torch.zeros((len(Xb), n_seg, C), dtype=torch.float64, device=device)
        zp = ro.encode(B0)
        xp = B0
        for j in range(steps):
            xn = B0 + ts[j + 1] * dX
            zn = ro.encode(xn)
            zm = ro.encode(B0 + 0.5 * (ts[j] + ts[j + 1]) * dX)   # 中点法
            z = torch.tensor(zm, device=device, requires_grad=True)
            out = ro.logodds(z)
            dz = torch.tensor(zn - zp, device=device)
            # 这一步里各特征的原值在每一段内走了多远，直通列的累加量按此比例记到各段名下
            xp_t = torch.tensor(xp, device=device)
            xn_t = torch.tensor(xn, device=device)
            lo = torch.minimum(xp_t, xn_t)[:, sf]
            hi = torch.maximum(xp_t, xn_t)[:, sf]
            dist = (torch.minimum(hi, sb) - torch.maximum(lo, sa)).clamp_min(0.0)
            tot = (dist @ agg[:n_seg])[:, sf]
            share = torch.where(tot > 0, dist / tot.clamp_min(1e-300), torch.zeros_like(dist))
            for c in range(C):
                g = torch.autograd.grad(out[:, c].sum(), z, retain_graph=c < C - 1)[0]
                gc = g * dz
                Wb[:, :, c] += gc @ agg
                Sb[:, :, c] += gc[:, :n_seg] + gc[:, n_seg:][:, sf] * share
            zp = zn
            xp = xn
            done += 1
            if progress is not None:
                progress(done, steps * n_batch)
        W[a:a + batch_size] = Wb.cpu().numpy()
        W_seg[a:a + batch_size] = Sb.cpu().numpy()
        with torch.no_grad():
            D[a:a + batch_size] = ro.logodds(torch.tensor(ro.encode(Xb), device=device)).cpu().numpy()
    err = np.abs(W.sum(1) - (D - d_ref[None, :]))
    back = np.zeros_like(W)                                      # 各段权值按特征加回去，应等于 W
    for f in range(F):
        back[:, f, :] = W_seg[:, cols[f], :].sum(1)
    return dict(W=W, D=D, D_ref=d_ref, err_max=float(err.max()), err_mean=float(err.mean()),
                W_seg=W_seg, seg_feature=seg_feature, seg_lo=seg_lo, seg_hi=seg_hi,
                seg_err_max=float(np.abs(back - W).max()),
                steps=int(steps), multilabel=bool(multilabel))


def segment_table(model, X, res, ref, features=None, k=PSEUDO_CASES):
    """分段权值表：组件 A 的每一段一行，给出越过该段（由低到高）一整段的边际权值。

    res 为 path_weights 的返回值（X、ref 须与当时相同），features 可限定只出哪些特征。
    对每一段，先看每个样本相对参照样本越过了多少（delta，向上为正，整段为 1，
    没碰到为 0），再在碰到该段的样本上算
        mean  = Σ sign(delta) · w_seg / Σ |delta|
        value = mean × E / (E + k)
    E 的算法同 weight_table，样本取碰到该段的那些。
    返回 dict：feature (R,) 特征列号，lo、hi (R,) 段的下界与上界，n (R,) 碰到该段的样本数，
    amount (R,) 越过量之和，value、mean、E_with、E_ref (R, C)。
    没有样本碰到的段 n 为 0，权值记 0。
    """
    net = net_of(model)
    X = np.asarray(X, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).ravel()
    W_seg, seg_feature = res["W_seg"], np.asarray(res["seg_feature"])
    S = len(seg_feature)
    delta = net._enc.transform(X)[:, :S] - net._enc.transform(ref[None])[:, :S]
    P = np.asarray(model.predict_proba(X), dtype=np.float64)
    C = P.shape[1]
    keep = (np.arange(S) if features is None
            else np.flatnonzero(np.isin(seg_feature, np.asarray(list(features)))))
    R = len(keep)
    mean = np.zeros((R, C))
    e_with = np.zeros((R, C))
    e_ref = np.zeros((R, C))
    n = np.zeros(R, dtype=int)
    amount = np.zeros(R)
    p_ref = {}
    for r, j in enumerate(keep):
        f = int(seg_feature[j])
        m = np.abs(delta[:, j]) > 1e-12
        n[r] = int(m.sum())
        if n[r] == 0:
            continue
        if f not in p_ref:
            X0 = X.copy()
            X0[:, f] = ref[f]
            p_ref[f] = np.asarray(model.predict_proba(X0), dtype=np.float64)
        d = delta[m, j]
        amount[r] = float(np.abs(d).sum())
        mean[r] = (np.sign(d)[:, None] * W_seg[m, j, :]).sum(0) / amount[r]
        e_with[r] = P[m].sum(0)
        e_ref[r] = p_ref[f][m].sum(0)
    E = np.maximum(e_with, e_ref)
    return dict(feature=seg_feature[keep], lo=np.asarray(res["seg_lo"])[keep],
                hi=np.asarray(res["seg_hi"])[keep], n=n, amount=amount,
                value=mean * E / (E + k), mean=mean, E_with=e_with, E_ref=e_ref)


def weight_table(model, X, W, rows, ref, k=PSEUDO_CASES):
    """权值表。

    rows 为 [(特征列号, 布尔掩码), ...]，每项是表的一行（某特征的某个取值）。
    返回 dict，各项形状 (行数, C)：
      value   格值 = mean × E / (E + k)，即定稿权值
      mean    出现时平均（未打折）
      E_with  带着该特征时模型预计的例数
      E_ref   把该特征换成参照值后模型预计的例数
    另有 n (行数,) 为每行的样本数。
    """
    X = np.asarray(X, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).ravel()
    P = np.asarray(model.predict_proba(X), dtype=np.float64)
    C = P.shape[1]
    R = len(rows)
    mean = np.zeros((R, C))
    e_with = np.zeros((R, C))
    e_ref = np.zeros((R, C))
    n = np.zeros(R, dtype=int)
    for r, (f, m) in enumerate(rows):
        m = np.asarray(m, dtype=bool)
        if not m.any():
            raise ValueError(f"第 {r} 行没有样本")
        n[r] = int(m.sum())
        mean[r] = W[m, f, :].mean(0)
        e_with[r] = P[m].sum(0)
        X0 = X[m].copy()
        X0[:, f] = ref[f]
        e_ref[r] = np.asarray(model.predict_proba(X0), dtype=np.float64).sum(0)
    E = np.maximum(e_with, e_ref)
    return dict(value=mean * E / (E + k), mean=mean, E_with=e_with, E_ref=e_ref, n=n)


def cut_order(xf, Y, best_split, bins=16):
    """重放组件 A 的贪心切分，按接受先后返回折点。

    A 只保存排序后的折点，不保存先后，所以读数分段要取「前三刀」时须重放一遍。
    xf、Y 为拟合 A 时用的那批训练数据的一列特征与标签，best_split(xs, ys, min_leaf)
    为模型文件里的最优分割函数，返回 (增益, 阈值, 左段末位)：
      多标签版  dannv2_beta_ml._best_split_ml，Y 为 (n, L) 浮点矩阵
      单标签版  lambda xs, ys, m: dannv2._best_split(xs, ys, 类数, m)，Y 为类别序号
    """
    xf = np.asarray(xf, dtype=np.float64)
    Y = np.asarray(Y)
    n = len(xf)
    min_leaf = max(2, n // 64)
    o = np.argsort(xf, kind="mergesort")
    xs, ys = xf[o], Y[o]
    segs, out = [], []
    g, t, i = best_split(xs, ys, min_leaf)
    if t is not None:
        segs.append((g, t, 0, n, i))
    while segs and len(out) < bins - 1:
        segs.sort(key=lambda s: -s[0])
        g, t, a, b, i = segs.pop(0)
        out.append(t)
        for a2, b2 in ((a, i + 1), (i + 1, b)):
            if b2 - a2 < 2 * min_leaf:
                continue
            g2, t2, i2 = best_split(xs[a2:b2], ys[a2:b2], min_leaf)
            if t2 is not None:
                segs.append((g2, t2, a2, b2, a2 + i2))
    return out


def value_rows(x, ref_value, cuts=None, missing=None, level_names=None,
               max_levels=MAX_LEVELS, n_cuts=N_CUTS):
    """把一个特征的取值划成表里的若干行，返回 [(取值说明, 布尔掩码), ...]。

    x            (N,) 该特征在解释集上的取值
    ref_value    该特征的参照值
    cuts         cut_order 的结果，取值超过 max_levels 种时必填
    missing      缺失标记（没有就不填），缺失单列一行，不占折点名额
    level_names  {取值: 名称}，逐值列出时用
    """
    x = np.asarray(x, dtype=np.float64)
    miss = np.zeros(len(x), dtype=bool) if missing is None else (x == missing)
    v = np.unique(x[~miss])
    names = level_names or {}
    rows = []
    if len(v) <= max_levels:
        rn = names.get(float(ref_value), f"{ref_value:g}")
        for u in v:
            un = names.get(float(u), f"{u:g}")
            tag = "（参照，权值为 0）" if u == ref_value else f"（相对{rn}）"
            rows.append((un + tag, (x == u) & ~miss))
    else:
        if cuts is None:
            raise ValueError("取值超过 max_levels 种的特征须给 cuts")
        use = sorted([t for t in cuts if t > v.min()][:n_cuts])
        bnd = [-np.inf] + use + [np.inf]
        for a, b in zip(bnd[:-1], bnd[1:]):
            if np.isfinite(a) and np.isfinite(b):
                lab = f"{a:g} 至 {b:g}"
            else:
                lab = f"≤{b:g}" if np.isfinite(b) else f">{a:g}"
            rows.append((lab + f"（相对 {ref_value:g}）", (x > a) & (x <= b) & ~miss))
    if miss.any():
        rows.append(("缺失", miss))
    return [(name, m) for name, m in rows if m.any()]


def top_weights(value, row_names, class_names, k=5, rows=None):
    """各类权值前 k，按绝对值排名、保留正负号。rows 可限定参与排名的行号。

    返回 {类名: [(行名, 带符号权值), ...]}，同绝对值按行序稳定排序。
    """
    value = np.asarray(value, dtype=np.float64)
    idx = np.arange(value.shape[0]) if rows is None else np.asarray(rows)
    out = {}
    for c, name in enumerate(class_names):
        order = idx[np.argsort(-np.abs(value[idx, c]), kind="stable")[:k]]
        out[str(name)] = [(row_names[i], float(value[i, c])) for i in order]
    return out
