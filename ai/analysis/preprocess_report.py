"""전처리 과정 보고서 — 단계마다 그래프 + 정확도 (2차 수집 11명, data/dataset/)

교수님 요청: 전처리 단계를 하나씩 그래프로 보여주고, 어떤 상태로 학습해서 정확도가 어떻게 됐는지.

실행 (저장소 최상위, .venv 켠 상태). 단계 하나씩 돌리고 그래프를 보면서 진행할 수 있다.
    python -m ai.analysis.preprocess_report --step p0     # 원본 그대로 학습
    python -m ai.analysis.preprocess_report --step p1     # 센서 상태 진단 (학습 없음, 몇 초)
    python -m ai.analysis.preprocess_report --step p2     # 고장 많은 채널 제외
    python -m ai.analysis.preprocess_report --step p3     # 접촉 끊김 복구
    python -m ai.analysis.preprocess_report --step p4     # 문제 세션 학습 제외 실험
    python -m ai.analysis.preprocess_report --step p5     # 움직임 감지기 검증 (쉬는시간 = 정답)
    python -m ai.analysis.preprocess_report --step p6     # 정규화 비교 → 증강 → 모델 비교 (가장 오래 걸림)
    python -m ai.analysis.preprocess_report               # 전부 순서대로
    옵션: --fast (빠르게, 흐름 확인용) / --nested (P6 중첩 검증) / --rerun (저장된 정확도 무시) / --no-timelines

결과: data/processed/report/
    pX_*.png        그 단계 그래프          pX_result.md   그 단계 설명·숫자 (노션에 붙여 넣기용 표 포함)
    steps_acc.png   단계별 정확도 한 장     results.json   단계별 정확도 저장 (다음 단계가 다시 계산하지 않음)

평가 규칙 (모든 단계 공통)
    - 한 사람 빼고 학습 → 그 사람으로 시험 (11번, LOSO). 시험할 사람 데이터는 기준값 계산에만 쓰고 학습에 안 섞음
    - 시험 데이터는 모든 단계에서 같다: 각 사람의 '마지막 시도, 자세 유지 구간' 2초 묶음 전부.
      (단계마다 바뀌는 건 학습 데이터와 특징뿐 → 정확도 차이 = 그 단계의 효과)
    - 정확도 = 사람별 정확도의 평균 (사람마다 같은 무게)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from ai.analysis.dataset_v2 import (
    ADC_MAX, ROW_Y, SHORT_DROP, SIDE_X, baselines, load_dataset, repair_dropouts, sensor_qc, session_flags,
)
from ai.classify.rules import ThresholdRule
from ai.features.pressure_pipeline import MovementParams, _jump_events, _spread, show_path

log = logging.getLogger("sitsense.report")

LABELS = ["normal", "lean_left", "lean_right", "cross_left", "cross_right"]
LABEL_KO = {"normal": "바른자세", "lean_left": "왼쪽편향", "lean_right": "오른쪽편향",
            "cross_left": "왼다리꼬기", "cross_right": "오른다리꼬기"}
CH_KO = {"L_front": "앞 왼쪽", "R_front": "앞 오른쪽", "L_mid": "중간 왼쪽", "R_mid": "중간 오른쪽",
         "L_back": "뒤 왼쪽", "C_back": "뒤 가운데", "R_back": "뒤 오른쪽"}
STATUS_KO = {"ok": "정상", "unloaded": "안 눌림", "dropout": "접촉 끊김", "dead": "고장"}
STATUS_COLOR = {"ok": "#4caf50", "unloaded": "#ffc107", "dropout": "#ff7043", "dead": "#9e9e9e"}
SWAP = {"normal": "normal", "lean_left": "lean_right", "lean_right": "lean_left",
        "cross_left": "cross_right", "cross_right": "cross_left"}
NORMS = {
    "A": "A. 기준 비율 (x−b)/b",
    "B": "B. 바닥 있는 기준 비율 (x−b)/max(b,25%)",
    "C": "C. 압력 분포 비율 변화",
    "D": "D. 무게중심 이동",
    "CD": "C+D",
}
MODEL_KO = {"rule": "규칙", "logreg": "로지스틱 회귀", "knn": "KNN", "svm": "SVM", "rf": "Random Forest",
            "hgb": "Gradient Boosting"}
WIN = 4                                    # 2초 묶음


# ---------------------------------------------------------------------------
# 2초 묶음 · 특징
# ---------------------------------------------------------------------------
def make_windows(df: pd.DataFrame, channels: list[str], keep: pd.Series) -> pd.DataFrame:
    """자세 유지 구간(seg)마다 keep인 줄만 4개씩 묶어 채널 중앙값"""
    rows = []
    sub = df[keep.reindex(df.index, fill_value=False)]
    for seg, g in sub.groupby("seg", sort=False):
        x = g[channels].to_numpy(dtype=float)
        for i in range(0, len(g) - WIN + 1, WIN):
            r = {"person": g["person"].iloc[0], "seg": seg, "label": g["label"].iloc[0],
                 "y": g["seat_label"].iloc[0], "row0": int(g.index[i])}
            r.update(dict(zip(channels, np.median(x[i:i + WIN], axis=0))))
            rows.append(r)
    return pd.DataFrame(rows)


def _lr(F: pd.DataFrame, v: pd.DataFrame, channels: list[str], prefix: str) -> None:
    left = [c for c in channels if c.startswith("L_")]
    right = [c for c in channels if c.startswith("R_")]
    for row in ("front", "mid", "back"):
        if f"L_{row}" in channels and f"R_{row}" in channels:
            F[f"LR_{row}"] = v[f"L_{row}"] - v[f"R_{row}"]
    if left and right:
        F["LR_all"] = v[left].mean(axis=1) - v[right].mean(axis=1)
    if {"L_front", "R_back", "R_front", "L_back"} <= set(channels):
        F["diag"] = (v["L_front"] + v["R_back"]) - (v["R_front"] + v["L_back"])


def features(W: pd.DataFrame, channels: list[str], kind: str, base: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """kind: raw / A / B / C / D / CD"""
    X = W[channels].astype(float)
    if kind == "raw":
        return X.add_prefix("v_")
    if isinstance(base, pd.DataFrame):          # 줄마다 기준값이 다를 때 (예: 앉은 위치별 기준)
        B = base.loc[W.index, channels].astype(float)
    else:                                       # 사람마다 기준값 하나
        B = pd.DataFrame([base[p][channels].to_numpy(dtype=float) for p in W["person"]], index=W.index,
                         columns=channels)
    F = pd.DataFrame(index=W.index)
    if kind in ("A", "B"):
        denom = np.maximum(B, 1.0) if kind == "A" else np.maximum(B, 0.25 * ADC_MAX)
        d = ((X - B) / denom).clip(-1, 1)
        F = d.add_prefix("n_")
        _lr(F, d, channels, "n_")
        F["n_low"] = (d < -0.3).sum(axis=1).astype(float)
        return F
    tot_x = X.sum(axis=1).clip(lower=1.0)
    tot_b = B.sum(axis=1).clip(lower=1.0)
    if kind in ("C", "CD"):
        s = X.div(tot_x, axis=0) - B.div(tot_b, axis=0)
        F = s.add_prefix("n_")
        _lr(F, s, channels, "n_")
    if kind in ("D", "CD"):
        px = np.array([SIDE_X[c.split("_")[0]] for c in channels])
        py = np.array([ROW_Y[c.split("_")[1]] for c in channels])
        cop = lambda M, t: (M.to_numpy() @ px / t.to_numpy(), M.to_numpy() @ py / t.to_numpy())   # noqa: E731
        (xx, xy), (bx, by) = cop(X, tot_x), cop(B, tot_b)
        F["cop_x"], F["cop_y"] = xx - bx, xy - by
        F["load"] = tot_x / tot_b - 1.0
    return F


def mirror(F: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """좌우 반전: 왼쪽·오른쪽 센서 특징 교환, 좌우차·대각선·좌우 무게중심 부호 반전, 라벨 좌우 교환"""
    M = F.copy()
    for c in F.columns:
        pre, _, rest = c.partition("_")
        if rest.startswith("L_") and f"{pre}_R_{rest[2:]}" in F.columns:
            M[c] = F[f"{pre}_R_{rest[2:]}"]
        elif rest.startswith("R_") and f"{pre}_L_{rest[2:]}" in F.columns:
            M[c] = F[f"{pre}_L_{rest[2:]}"]
        elif c.startswith(("LR_", "diag", "cop_x")):
            M[c] = -F[c]
    return M, y.map(SWAP)


# ---------------------------------------------------------------------------
# 모델 · 평가
# ---------------------------------------------------------------------------
def make_model(name: str, n_trees: int = 200, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if name == "rf":
        return RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                              random_state=seed)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", class_weight="balanced"))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "rule":
        return ThresholdRule()
    raise ValueError(name)


def loso(F: pd.DataFrame, y: pd.Series, person: pd.Series, test: pd.Series, train: pd.Series,
         model: str = "rf", aug: bool = False, n_trees: int = 200, exclude_people=()) -> dict:
    """사람 하나씩 빼고 학습 → 시험. test/train: 이 줄을 시험/학습에 쓸지 (bool)"""
    from sklearn.metrics import f1_score
    people = sorted(person.unique())
    acc, f1, pred = {}, {}, pd.Series(index=F.index, dtype=object)
    for p in people:
        tr = train & (person != p) & ~person.isin(exclude_people)
        te = test & (person == p)
        Xtr, ytr = F[tr], y[tr]
        if aug:
            M, my = mirror(Xtr, ytr)
            Xtr, ytr = pd.concat([Xtr, M]), pd.concat([ytr, my])
        m = make_model(model, n_trees).fit(Xtr, ytr)
        yp = m.predict(F[te])
        pred[te] = yp
        acc[p] = float((yp == y[te]).mean())
        f1[p] = float(f1_score(y[te], yp, labels=LABELS, average="macro", zero_division=0))
    return {"acc": acc, "f1": f1, "mean": float(np.mean(list(acc.values()))),
            "f1_mean": float(np.mean(list(f1.values()))), "pred": pred}


# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ai.features.inspect_pressure import setup_font
    setup_font(plt)
    return plt


def plot_overview(df, people, out, plt):
    use = df[~df["rest"]]
    t = use.groupby(["person", "label", "last_attempt"]).size().unstack(fill_value=0)
    order = [f"{c}({d})" for c in ("바른자세", "왼쪽편향", "오른쪽편향", "왼다리꼬기", "오른다리꼬기")
             for d in ("앞", "중", "뒤")]
    labels = [l for l in order if l in set(use["label"])]
    pp = sorted(use["person"].unique())
    M = np.zeros((len(pp), len(labels)))
    R = np.zeros_like(M)
    for i, p in enumerate(pp):
        for j, l in enumerate(labels):
            if (p, l) in t.index:
                M[i, j] = t.loc[(p, l)].get(True, 0) * 0.5
                R[i, j] = t.loc[(p, l)].get(False, 0)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
    for i in range(len(pp)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.0f}" + ("\n재측정" if R[i, j] else ""), ha="center", va="center",
                    fontsize=7, color="white" if M[i, j] > M.max() * 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pp)), pp)
    ax.set_title(f"수집 데이터: {len(pp)}명 × 자세 {len(labels)}개 (칸 = 자세 유지 초, '재측정' = 다시 찍어 마지막 시도만 사용)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_timeline(g, channels, moving, repaired_log, out, plt):
    fig, axes = plt.subplots(len(channels) + 1, 1, figsize=(16, 1.25 * (len(channels) + 1)), sharex=True)
    t = np.arange(len(g)) * 0.5 / 60
    rest = g["rest"].to_numpy()
    mv = moving.loc[g.index].to_numpy()
    for ax, c in zip(axes, channels):
        ax.plot(t, g[c].to_numpy(), lw=0.6, color="#1565c0")
        ax.fill_between(t, 0, ADC_MAX, where=rest, color="#bdbdbd", alpha=0.5, lw=0)
        rl = repaired_log[repaired_log["channel"] == c]
        for _, r in rl.iterrows():
            i0 = g.index.get_loc(r["start"])
            ax.axvspan(t[i0], t[min(i0 + r["length"], len(t) - 1)], color="red", alpha=0.6, lw=0)
        ax.set_ylim(0, ADC_MAX)
        ax.set_ylabel(CH_KO.get(c, c), rotation=0, ha="right", fontsize=8)
        ax.tick_params(labelsize=7)
    ax = axes[-1]
    ax.fill_between(t, 0, 1, where=mv, color="orange", lw=0, label="움직임 감지")
    ax.fill_between(t, 0, 1, where=rest, color="#bdbdbd", alpha=0.5, lw=0, label="쉬는시간(정답)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlabel("분")
    starts = g.index[(g["seg"] != g["seg"].shift()) & ~g["rest"]]
    for s in starts:
        i0 = g.index.get_loc(s)
        axes[0].text(t[i0], ADC_MAX * 1.02, g.loc[s, "label"], fontsize=6, rotation=30)
    fig.suptitle(f"{g['person'].iloc[0]} 원본 압력 (회색 = 쉬는시간, 빨강 = 접촉 끊김, 아래 주황 = 움직임 감지)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_qc(qc, flags, channels, out, plt):
    pp = sorted(qc["person"].unique())
    code = {"ok": 0, "unloaded": 1, "dropout": 2, "dead": 3}
    M = qc.pivot(index="person", columns="channel", values="status").loc[pp, channels]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([STATUS_COLOR[k] for k in code])
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(M.replace(code).to_numpy(dtype=float), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    B = qc.pivot(index="person", columns="channel", values="baseline").loc[pp, channels]
    D = qc.pivot(index="person", columns="channel", values="dropouts").loc[pp, channels]
    for i in range(len(pp)):
        for j in range(len(channels)):
            ax.text(j, i, f"{STATUS_KO[M.iat[i, j]]}\n기준 {B.iat[i, j]}\n끊김 {D.iat[i, j]}", ha="center",
                    va="center", fontsize=6.5)
    fl = flags.set_index("person")
    ax.set_yticks(range(len(pp)), [p + (" ⚠" if fl.loc[p, "flagged"] else "") for p in pp])
    ax.set_xticks(range(len(channels)), [CH_KO.get(c, c) for c in channels], fontsize=9)
    ax.set_title("센서 상태 진단 (⚠ = 앞쪽 센서 문제로 학습 제외 후보)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=STATUS_COLOR[k], label=STATUS_KO[k]) for k in code], loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_channel_health(qc, channels, out, plt):
    t = qc.groupby(["channel", "status"]).size().unstack(fill_value=0).reindex(channels).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(channels))
    for k in ("ok", "unloaded", "dropout", "dead"):
        v = t[k].to_numpy() if k in t else np.zeros(len(channels))
        ax.bar([CH_KO.get(c, c) for c in channels], v, bottom=bottom, color=STATUS_COLOR[k], label=STATUS_KO[k])
        bottom += v
    ax.set_ylabel("사람 수")
    ax.set_title("센서별 상태 (11명) — 고장이 많은 센서는 모델에서 제외")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_repair(df, fixed, rlog, out, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [2, 2, 1.3]})
    for ax, kind, title in ((axes[0], "repaired", "짧은 끊김(≤3초) → 앞뒤 값으로 채움"),
                            (axes[1], "unreliable", "긴 끊김(>3초) → 복구 안 함, 학습에서 제외")):
        ex = rlog[rlog["kind"] == kind]
        if ex.empty:
            ax.set_title(title + " (없음)")
            continue
        r = ex.sort_values("length", ascending=(kind == "unreliable")).iloc[len(ex) // 2]
        if kind == "repaired":            # 예시: 앞뒤 15초가 안정적으로 눌려 있는 끊김 (보기 쉬운 것)
            for _, cand in ex.sort_values("length", ascending=False).iterrows():
                around = df.loc[cand["start"] - 30:cand["start"] + cand["length"] + 30]
                around = around[(around["person"] == cand["person"]) & (around["seg"] == cand["seg"])]
                outside = around.drop(index=range(cand["start"], cand["start"] + cand["length"]), errors="ignore")
                if len(around) >= 40 and (outside[cand["channel"]] >= 300).all():
                    r = cand
                    break
        c, s = r["channel"], r["start"]
        lo, hi = max(df.index.min(), s - 30), s + r["length"] + 30
        win = df.loc[lo:hi]
        win = win[win["person"] == r["person"]]
        t = (win.index - s) * 0.5
        ax.plot(t, win[c], color="#e53935", lw=1.2, label="원본")
        if kind == "repaired":
            ax.plot(t, fixed.loc[win.index, c], color="#1565c0", lw=1.2, ls="--", label="복구 후")
        else:
            ax.axvspan(0, r["length"] * 0.5, color="#bdbdbd", alpha=0.6, label="제외 구간")
        ax.set_title(f"{title}\n{r['person']} · {CH_KO.get(c, c)} · {r['label']}", fontsize=9)
        ax.set_xlabel("끊김 시작 기준 초")
        ax.set_ylim(0, ADC_MAX)
        ax.legend(fontsize=7)
    lens = rlog["length"] * 0.5
    axes[2].hist(lens, bins=np.arange(0, max(lens.max(), 4) + 1, 0.5), color="#78909c")
    axes[2].axvline(SHORT_DROP * 0.5, color="red", ls="--")
    axes[2].set_xlim(0, min(lens.max() + 1, 30))
    axes[2].set_title(f"끊김 길이 분포 (총 {len(rlog)}번)\n빨간 선 왼쪽 = 복구", fontsize=9)
    axes[2].set_xlabel("초")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_movement(mv_tab, out, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(mv_tab))
    axes[0].bar(x - 0.2, mv_tab["rest_flagged_pct"], 0.4, label="쉬는시간 중 감지 (높을수록 좋음)", color="#ff9800")
    axes[0].bar(x + 0.2, mv_tab["collect_flagged_pct"], 0.4, label="자세 유지 중 감지 (낮을수록 좋음)", color="#90a4ae")
    axes[0].set_xticks(x, mv_tab["person"])
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8)
    axes[0].set_title("움직임 감지 비율")
    axes[1].bar(x, mv_tab["transitions_caught_pct"], color="#43a047")
    axes[1].set_xticks(x, mv_tab["person"])
    axes[1].set_ylim(0, 105)
    axes[1].set_title("자세 바꾸기(쉬는시간 구간) 중 한 번이라도 잡은 비율")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_norm_view(W, channels, base, out, plt):
    """바른자세 구간의 앞쪽 왼쪽 센서 값이 사람마다 얼마나 다른지 (원본) → 정규화 후"""
    nm = W[W["y"] == "normal"]
    kinds = [("raw", "원본 값 (앞 왼쪽 센서)", "v_L_front"), ("A", "A. 기준 비율", "n_L_front"),
             ("C", "C. 분포 비율 변화", "n_L_front"), ("D", "D. 좌우 무게중심 이동", "cop_x")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(17, 4))
    pp = sorted(W["person"].unique())
    for ax, (k, title, col) in zip(axes, kinds):
        F = features(nm, channels, k, base)
        ax.boxplot([F.loc[nm["person"] == p, col] for p in pp], showfliers=False)
        ax.set_xticks(range(1, len(pp) + 1), pp)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        if k != "raw":
            ax.axhline(0, color="red", lw=0.8)
    fig.suptitle("바른자세에서 사람별 분포 — 원본은 사람마다 제각각, 정규화하면 0 근처로 모여야 함")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_acc_bars(tab, label_col, out, title, plt, highlight=None):
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(tab)), 4.4))
    x = np.arange(len(tab))
    colors = ["#1e88e5" if (highlight is None or r[label_col] != highlight) else "#e53935" for _, r in tab.iterrows()]
    ax.bar(x, tab["acc_mean"] * 100, color=colors, alpha=0.85)
    pcols = [c for c in tab.columns if c.startswith("p") and c[1:].isdigit()]
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.scatter(np.full(len(pcols), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(pcols)),
                   r[pcols].astype(float) * 100, s=10, color="black", zorder=3)
        ax.text(i, r["acc_mean"] * 100 + 1.5, f"{r['acc_mean']:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x, tab[label_col], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("정확도 % (막대 = 평균, 점 = 사람별)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_paired(tab, out, plt, stats):
    pcols = [c for c in tab.columns if c.startswith("p") and c[1:].isdigit()]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tab))
    for p in pcols:
        ax.plot(x, tab[p] * 100, color="#b0bec5", lw=0.8, marker="o", ms=3)
    ax.plot(x, tab["acc_mean"] * 100, color="#e53935", lw=2.5, marker="s", label="평균")
    ax.set_xticks(x, tab["model_ko"], rotation=15)
    ax.set_ylabel("정확도 %")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("모델 비교 (회색 선 = 같은 사람)\n" + stats, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(y, yp, title, out, plt):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp, labels=LABELS)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ko = [LABEL_KO[l] for l in LABELS]
    ax.set_xticks(range(5), ko, rotation=30, ha="right")
    ax.set_yticks(range(5), ko)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)




# ---------------------------------------------------------------------------
# 공통 준비 (필요할 때 한 번만 계산)
# ---------------------------------------------------------------------------
def md_table(t: pd.DataFrame) -> str:
    """마크다운 표 (노션에 그대로 붙여 넣기 가능)"""
    cols = list(t.columns)
    import re as _re
    pct = {c for c in cols if c in ("acc_mean", "macro_f1") or c.startswith("train_") or _re.fullmatch(r"p\d\d", c)}

    def fmt(c, v):
        if isinstance(v, float) and np.isnan(v):
            return "-"
        return f"{v:.1%}" if c in pct and isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(cols, r)) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)


def movement_mask(df, channels, params):
    """사람 타임라인 전체(쉬는시간 포함)에서 움직임 감지 — 실시간과 같이 시간 순서대로"""
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby("person", sort=False):
        body, _ = _jump_events(g, channels, params)
        out.loc[g.index] = _spread(body, params)
    return out


class Ctx:
    """단계들이 함께 쓰는 데이터. 처음 쓸 때 계산하고 기억한다"""

    def __init__(self, args, out):
        self.args, self.out = args, out
        self.n_trees = 60 if args.fast else 200
        self.plt = _plt()
        self.cache_path = out / "results.json"
        self.cache = {} if args.rerun or not self.cache_path.exists() else \
            json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._m = {}

    def _get(self, key, fn):
        if key not in self._m:
            self._m[key] = fn()
        return self._m[key]

    # 데이터 ---------------------------------------------------------------
    @property
    def data(self):
        return self._get("data", lambda: load_dataset(self.args.folder))

    @property
    def df(self):
        return self.data[0]

    @property
    def ch7(self):
        return self.data[1]

    @property
    def people(self):
        return self.data[2]

    @property
    def qc(self):
        return self._get("qc", lambda: sensor_qc(self.df, self.ch7))

    @property
    def flags(self):
        return self._get("flags", lambda: session_flags(self.qc))

    @property
    def drop_ch(self):
        dead = self.qc[self.qc["status"] == "dead"].groupby("channel").size()
        return [c for c, n in dead.items() if n >= len(self.people) / 2]

    @property
    def ch6(self):
        return [c for c in self.ch7 if c not in self.drop_ch]

    @property
    def repaired(self):
        return self._get("repaired", lambda: repair_dropouts(self.df, self.ch6))

    @property
    def moving(self):
        return self._get("moving", lambda: movement_mask(self.repaired[0], self.ch6, MovementParams()))

    @property
    def W_all(self):
        return self._get("W_all", lambda: make_windows(self.df, self.ch7, ~self.df["rest"]))

    @property
    def W_fix(self):
        def make():
            fixed, unreliable, _ = self.repaired
            W = make_windows(fixed, self.ch7, ~fixed["rest"])
            assert (W["row0"].values == self.W_all["row0"].values).all()
            W["unreliable"] = [unreliable.loc[r:r + WIN - 1].any() for r in W["row0"]]
            W["moving"] = [self.moving.loc[r:r + WIN - 1].any() for r in W["row0"]]
            W["depth"] = W["row0"].map(self.df["depth"])
            return W
        return self._get("W_fix", make)

    @property
    def is_last(self):
        return self.W_all["row0"].map(self.df["last_attempt"])

    @property
    def test(self):
        return self.is_last

    @property
    def y(self):
        return self.W_all["y"]

    @property
    def person(self):
        return self.W_all["person"]

    @property
    def clean(self):
        W = self.W_fix
        return self.is_last & ~W["unreliable"] & ~W["moving"]

    @property
    def base(self):
        return self._get("base", lambda: baselines(self.repaired[0], self.ch6, self.moving))

    @property
    def base_by_depth(self):
        """앉은 위치(앞/중/뒤)마다 그 위치의 바른자세를 기준으로 (분석용: 위치를 라벨에서 가져옴)"""
        def make():
            fixed = self.repaired[0]
            b = {}
            use = fixed[(fixed["cls"] == "바른자세") & ~fixed["rest"] & fixed["last_attempt"] & ~self.moving]
            for (p, d), g in use.groupby(["person", "depth"]):
                b[(p, d)] = g[self.ch6].iloc[4:24].median()
            W = self.W_fix
            return pd.DataFrame([b[(p, d)].to_numpy() for p, d in zip(W["person"], W["depth"])], index=W.index,
                                columns=self.ch6)
        return self._get("base_depth", make)

    # 평가 (결과는 results.json에 저장해서 다음 단계가 다시 계산하지 않음) -------
    def evaluate(self, key, label, F, train, **kw):
        if key in self.cache and not kw.pop("force", False):
            log.info("  %-40s %.1f%% (저장된 결과)", label, self.cache[key]["mean"] * 100)
            return self.cache[key]
        kw.pop("force", None)
        t0 = time.time()
        res = loso(F, self.y, self.person, self.test, train, n_trees=self.n_trees, **kw)
        log.info("  %-40s %.1f%%  (%.0f초)", label, res["mean"] * 100, time.time() - t0)
        keep = {k: v for k, v in res.items() if k != "pred"}
        keep["label"] = label
        self.cache[key] = keep
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        res["label"] = label
        return res

    def acc_table(self, keys) -> pd.DataFrame:
        rows = []
        for k in keys:
            r = self.cache[k]
            d = {"step": r["label"], "acc_mean": round(r["mean"], 4), "macro_f1": round(r["f1_mean"], 4)}
            d.update({p: round(v, 4) for p, v in r["acc"].items()})
            rows.append(d)
        t = pd.DataFrame(rows)
        t.insert(2, "diff_pp", (t["acc_mean"].diff() * 100).round(1))
        return t

    def note(self, step, title, lines, figs):
        """단계 설명 파일 (노션 정리용)"""
        text = [f"# {step.upper()} {title}", ""] + lines + ["", "그래프: " + ", ".join(figs)]
        (self.out / f"{step}_result.md").write_text("\n".join(text), encoding="utf-8")
        print("\n" + "\n".join(text) + "\n")


def raw_F(ctx, W, channels):
    return features(W, channels, "raw")


# ---------------------------------------------------------------------------
# 단계
# ---------------------------------------------------------------------------
def p0(ctx: Ctx):
    """원본 그대로 학습 (+ 다시 찍은 구간 정리)"""
    out, plt, df = ctx.out, ctx.plt, ctx.df
    ctx.people.to_csv(out / "people.csv", index=False, encoding="utf-8-sig")
    plot_overview(df, ctx.people, out / "p0_data_overview.png", plt)
    (out / "p0_timelines").mkdir(exist_ok=True)
    if not ctx.args.no_timelines:
        empty = pd.DataFrame(columns=["channel", "start", "length"])
        for p, g in df.groupby("person", sort=False):
            plot_timeline(g, ctx.ch7, pd.Series(False, index=df.index), empty, out / "p0_timelines" / f"{p}.png", plt)
    all_rows = pd.Series(True, index=ctx.W_all.index)
    ctx.evaluate("s0", "S0 원본 (모든 시도·7채널)", raw_F(ctx, ctx.W_all, ctx.ch7), all_rows)
    ctx.evaluate("s1", "S1 + 다시 찍은 구간은 마지막 시도만", raw_F(ctx, ctx.W_all, ctx.ch7), ctx.is_last)
    t = ctx.acc_table(["s0", "s1"])
    plot_acc_bars(t, "step", out / "p0_acc.png", "P0 원본 학습 (RF, 한 사람 빼고 평가)", plt)
    n_retry = int((~ctx.is_last).sum())
    ctx.note("p0", "원본 그대로 학습", [
        f"- 데이터: {len(ctx.people)}명, {len(df)}줄, 2초 묶음 {len(ctx.W_all)}개 (다시 찍기 전 시도 {n_retry}개)",
        "- 평가: 한 사람 빼고 학습 → 그 사람으로 시험 (11번). 시험 데이터는 모든 단계에서 '마지막 시도' 전부로 같음",
        "", md_table(t[["step", "acc_mean", "diff_pp", "macro_f1"]])],
        ["p0_data_overview.png", "p0_timelines/pXX.png", "p0_acc.png"])


def p1(ctx: Ctx):
    """센서 상태 진단"""
    out, plt, qc, flags = ctx.out, ctx.plt, ctx.qc, ctx.flags
    qc.to_csv(out / "p1_sensor_qc.csv", index=False, encoding="utf-8-sig")
    flags.to_csv(out / "p1_session_flags.csv", index=False, encoding="utf-8-sig")
    plot_qc(qc, flags, ctx.ch7, out / "p1_sensor_qc.png", plt)
    plot_channel_health(qc, ctx.ch7, out / "p1_channel_health.png", plt)
    counts = qc["status"].value_counts()
    ctx.note("p1", "센서 상태 진단", [
        "- 기준: 고장 = 세션 내내 100 미만 / 접촉 끊김 = 자세 유지 중 높은 값→0→높은 값이 분당 0.5회 이상 / "
        "안 눌림 = 바른자세(중) 기준값이 최대치의 5% 미만",
        "- 결과: " + ", ".join(f"{STATUS_KO[k]} {counts.get(k, 0)}칸" for k in ("ok", "unloaded", "dropout", "dead")),
        "- 앞쪽 센서 문제 사람(P4에서 사용): " + (", ".join(f"{r.person}({r.reason})" for r in
                                                   flags[flags.flagged].itertuples()) or "없음"),
        "- 한계: 세션 중간에 고장 난 경우(예: 몇 분 뒤부터 0)는 세션 전체 기준이라 못 잡을 수 있음"],
        ["p1_sensor_qc.png", "p1_channel_health.png"])


def p2(ctx: Ctx):
    """고장이 많은 채널 제외"""
    ctx.evaluate("s1", "S1 + 다시 찍은 구간은 마지막 시도만", raw_F(ctx, ctx.W_all, ctx.ch7), ctx.is_last)
    names = "·".join(CH_KO.get(c, c) for c in ctx.drop_ch) or "없음"
    ctx.evaluate("s2", f"S2 + {names} 제외 ({len(ctx.ch6)}채널)", raw_F(ctx, ctx.W_all, ctx.ch6), ctx.is_last)
    t = ctx.acc_table(["s1", "s2"])
    plot_acc_bars(t, "step", ctx.out / "p2_acc.png", "P2 고장 채널 제외 전후", ctx.plt)
    dead = ctx.qc[ctx.qc["status"] == "dead"].groupby("channel").size()
    ctx.note("p2", "고장이 많은 채널 제외", [
        f"- 규칙: 11명 중 절반 이상에서 고장인 채널은 뺀다 → {names}",
        "- 채널별 고장 사람 수: " + ", ".join(f"{CH_KO.get(c, c)} {n}" for c, n in dead.items()),
        "", md_table(t[["step", "acc_mean", "diff_pp", "macro_f1"]])], ["p1_channel_health.png", "p2_acc.png"])


def p3(ctx: Ctx):
    """접촉 끊김 복구"""
    fixed, unreliable, rlog = ctx.repaired
    rlog.to_csv(ctx.out / "p3_dropout_log.csv", index=False, encoding="utf-8-sig")
    plot_repair(ctx.df, fixed, rlog, ctx.out / "p3_dropout_repair.png", ctx.plt)
    names = "·".join(CH_KO.get(c, c) for c in ctx.drop_ch) or "없음"
    ctx.evaluate("s2", f"S2 + {names} 제외 ({len(ctx.ch6)}채널)", raw_F(ctx, ctx.W_all, ctx.ch6), ctx.is_last)
    ctx.evaluate("s3", "S3 + 짧은 끊김 복구·긴 끊김 학습 제외", raw_F(ctx, ctx.W_fix, ctx.ch6),
                 ctx.is_last & ~ctx.W_fix["unreliable"])
    t = ctx.acc_table(["s2", "s3"])
    plot_acc_bars(t, "step", ctx.out / "p3_acc.png", "P3 접촉 끊김 복구 전후", ctx.plt)
    by_ch = rlog.groupby(["channel", "kind"]).size().unstack(fill_value=0)
    ctx.note("p3", "접촉 끊김 복구", [
        f"- 규칙: 자세 유지 중 0 근처로 떨어졌다 돌아온 구간 중 {SHORT_DROP * 0.5:g}초 이하는 앞뒤 값으로 직선 채움, "
        "더 길면 복구하지 않고 그 2초 묶음을 학습에서 제외",
        f"- 복구 {int((rlog.kind == 'repaired').sum())}번, 학습 제외 {int((rlog.kind == 'unreliable').sum())}번 "
        f"(제외된 2초 묶음 {int((ctx.is_last & ctx.W_fix['unreliable']).sum())}개)",
        "", md_table(by_ch.reset_index()), "", md_table(t[["step", "acc_mean", "diff_pp", "macro_f1"]])],
        ["p3_dropout_repair.png", "p3_acc.png"])


def p4(ctx: Ctx):
    """앞쪽 센서 문제 사람을 학습에서 뺐을 때 / 넣었을 때 (시험은 11명 전원)"""
    bad = ctx.flags.loc[ctx.flags["flagged"], "person"].tolist()
    train = ctx.is_last & ~ctx.W_fix["unreliable"]
    F = raw_F(ctx, ctx.W_fix, ctx.ch6)
    a = ctx.evaluate("s3", "S3 + 짧은 끊김 복구·긴 끊김 학습 제외", F, train)
    b = ctx.evaluate("p4_filtered", f"P4 문제 세션({', '.join(bad) or '없음'}) 학습 제외", F, train,
                     exclude_people=bad)
    pp = sorted(a["acc"])
    t = pd.DataFrame({"person": pp, "train_all": [a["acc"][p] for p in pp],
                      "train_filtered": [b["acc"][p] for p in pp], "flagged": [p in bad for p in pp]})
    t.to_csv(ctx.out / "p4_session_filter.csv", index=False, encoding="utf-8-sig")
    plt = ctx.plt
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(pp))
    ax.bar(x - 0.2, t["train_all"] * 100, 0.4, label=f"전원 학습 (평균 {a['mean']:.1%})", color="#90a4ae")
    ax.bar(x + 0.2, t["train_filtered"] * 100, 0.4, label=f"문제 세션 제외 학습 (평균 {b['mean']:.1%})", color="#1e88e5")
    ax.set_xticks(x, [p + (" ⚠" if f else "") for p, f in zip(pp, t["flagged"])])
    ax.set_ylim(0, 105)
    ax.set_ylabel("정확도 %")
    ax.legend(fontsize=8)
    ax.set_title(f"P4 앞쪽 센서 문제 세션({', '.join(bad) or '없음'})을 학습에서 뺐을 때 (시험은 11명 전원)")
    fig.tight_layout()
    fig.savefig(ctx.out / "p4_session_filter.png", dpi=130)
    plt.close(fig)
    good = [p for p in pp if p not in bad]
    ctx.note("p4", "문제 세션 학습 제외 실험", [
        "- 규칙(P1): 앞쪽 센서가 고장이거나, 바르게 앉아도 안 눌리고 자세 절반 이상에서 0인 사람",
        f"- 대상: {', '.join(f'{r.person}({r.reason})' for r in ctx.flags[ctx.flags.flagged].itertuples()) or '없음'}",
        "- 시험은 두 경우 모두 11명 전원 (제외한 사람도 시험함 → 결과를 좋게 보이려고 뺀 것이 아님)",
        f"- 전원 학습 {a['mean']:.1%} → 제외 학습 {b['mean']:.1%} "
        f"(정상 세션만 보면 {np.mean([a['acc'][p] for p in good]):.1%} → {np.mean([b['acc'][p] for p in good]):.1%})"],
        ["p4_session_filter.png"])


def p5(ctx: Ctx):
    """움직임 감지기 검증 (쉬는시간 = 정답) + 움직임 구간 학습 제외"""
    fixed = ctx.repaired[0]
    moving = ctx.moving
    rows = []
    for p, g in fixed.groupby("person", sort=False):
        m = moving.loc[g.index]
        caught = [m.loc[s.index].any() for _, s in g[g["rest"]].groupby("seg")]
        rows.append({"person": p, "rest_flagged_pct": round(m[g["rest"]].mean() * 100, 1),
                     "collect_flagged_pct": round(m[~g["rest"]].mean() * 100, 1),
                     "transitions": len(caught), "transitions_caught_pct": round(np.mean(caught) * 100, 1)})
    mv = pd.DataFrame(rows)
    mv.to_csv(ctx.out / "p5_movement_check.csv", index=False, encoding="utf-8-sig")
    plot_movement(mv, ctx.out / "p5_movement_check.png", ctx.plt)
    (ctx.out / "p5_timelines").mkdir(exist_ok=True)
    if not ctx.args.no_timelines:
        rlog = ctx.repaired[2]
        for p, g in ctx.df.groupby("person", sort=False):
            plot_timeline(g, ctx.ch7, moving, rlog[rlog["person"] == p], ctx.out / "p5_timelines" / f"{p}.png", ctx.plt)
    F = raw_F(ctx, ctx.W_fix, ctx.ch6)
    ctx.evaluate("s3", "S3 + 짧은 끊김 복구·긴 끊김 학습 제외", F, ctx.is_last & ~ctx.W_fix["unreliable"])
    ctx.evaluate("s4", "S4 + 움직임 구간 학습 제외", F, ctx.clean)
    t = ctx.acc_table(["s3", "s4"])
    plot_acc_bars(t, "step", ctx.out / "p5_acc.png", "P5 움직임 구간 학습 제외 전후", ctx.plt)
    ctx.note("p5", "움직임 감지기 검증", [
        "- 감지 규칙(1차와 같음): 0.5초 사이 800 이상 변한 채널이 2개 이상(또는 1개인데 0 근처가 아님) → 그때부터 2초 + 직전 1초",
        "- 쉬는시간(자세 바꾸는 10초)을 정답으로 처음 검증",
        f"- 쉬는시간 중 감지 {mv.rest_flagged_pct.mean():.1f}% (높을수록 좋음), 자세 유지 중 감지 "
        f"{mv.collect_flagged_pct.mean():.1f}% (낮을수록 좋음), 자세 바꾸기를 한 번이라도 잡은 비율 "
        f"{mv.transitions_caught_pct.mean():.1f}%",
        "", md_table(mv), "", md_table(t[["step", "acc_mean", "diff_pp", "macro_f1"]])],
        ["p5_movement_check.png", "p5_timelines/pXX.png", "p5_acc.png"])


def p6(ctx: Ctx):
    """정규화 비교 → 좌우 대칭 증강 → 모델 비교"""
    out, plt, W, ch6 = ctx.out, ctx.plt, ctx.W_fix, ctx.ch6
    clean = ctx.clean
    plot_norm_view(W, ch6, ctx.base, out / "p6_normalization_view.png", plt)
    ctx.evaluate("s4", "S4 + 움직임 구간 학습 제외", raw_F(ctx, W, ch6), clean)
    keys = ["s4"]
    for k, ko in NORMS.items():
        ctx.evaluate(f"norm_{k}", ko, features(W, ch6, k, ctx.base), clean)
        keys.append(f"norm_{k}")
    ctx.evaluate("norm_E", "E. 앉은 위치별 기준 (분석용)", features(W, ch6, "B", ctx.base_by_depth), clean)
    keys.append("norm_E")
    nt = ctx.acc_table(keys)
    nt.loc[0, "step"] = "원본 값 (S4)"
    nt.to_csv(out / "p6_normalization.csv", index=False, encoding="utf-8-sig")
    best = max(NORMS, key=lambda k: ctx.cache[f"norm_{k}"]["mean"])
    plot_acc_bars(nt, "step", out / "p6_normalization_acc.png", "P6 정규화 방법 비교 (RF)", plt, highlight=NORMS[best])
    F5 = features(W, ch6, best, ctx.base)
    ctx.evaluate("s6", "S6 + 좌우 대칭 증강", F5, clean, aug=True)

    mods = {}
    for m in ("rule", "logreg", "knn", "svm", "rf", "hgb"):
        Fm = F5 if (m != "rule" or "LR_front" in F5.columns) else features(W, ch6, "C", ctx.base)
        mods[m] = ctx.evaluate(f"model_{m}", MODEL_KO[m], Fm, clean, model=m, aug=True, force=True)
    mt = ctx.acc_table([f"model_{m}" for m in mods]).drop(columns="diff_pp")
    mt.insert(0, "model_ko", mt.pop("step"))
    mt.to_csv(out / "p6_models.csv", index=False, encoding="utf-8-sig")
    best_m = max(mods, key=lambda k: mods[k]["mean"])
    from scipy.stats import wilcoxon
    pp = sorted(mods[best_m]["acc"])
    wrows = []
    for m in mods:
        if m == best_m:
            continue
        a = np.array([mods[best_m]["acc"][p] for p in pp])
        b = np.array([mods[m]["acc"][p] for p in pp])
        try:
            pv = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
        except ValueError:
            pv = 1.0
        wrows.append({"best": MODEL_KO[best_m], "vs": MODEL_KO[m], "mean_diff_pp": round((a - b).mean() * 100, 1),
                      "wins": int((a > b).sum()), "losses": int((a < b).sum()), "p_value": round(pv, 4)})
    wt = pd.DataFrame(wrows)
    wt.to_csv(out / "p6_wilcoxon.csv", index=False, encoding="utf-8-sig")
    rr = wt[wt["vs"] == MODEL_KO["rule"]]
    stats = (f"{MODEL_KO[best_m]} vs 규칙: 평균 {rr.mean_diff_pp.iloc[0]:+.1f}%p, {rr.wins.iloc[0]}승 "
             f"{rr.losses.iloc[0]}패, Wilcoxon p = {rr.p_value.iloc[0]}") if len(rr) else ""
    plot_paired(mt, out / "p6_models_paired.png", plt, stats)
    te = ctx.test
    plot_confusion(ctx.y[te], mods[best_m]["pred"][te], f"{MODEL_KO[best_m]} 혼동 행렬 (한 사람 빼고 평가)",
                   out / f"p6_confusion_{best_m}.png", plt)

    lines = ["- 기준값 b = 각 사람 바른자세(중)에서 움직임 없는 10초 중앙값 (실제 서비스의 측정 시작 10초와 같음)",
             "- A~D는 실제 서비스에서 그대로 쓸 수 있음. E는 앉은 위치를 라벨에서 가져온 분석용 (실서비스는 거리센서로 위치 추정 필요)",
             "", md_table(nt[["step", "acc_mean", "macro_f1"]]), "", f"- 선택: {NORMS[best]} → 좌우 대칭 증강 "
             f"{ctx.cache['s6']['mean']:.1%}", "", "모델 비교 (선택한 정규화 + 증강)", "",
             md_table(mt[["model_ko", "acc_mean", "macro_f1"]]), "", md_table(wt)]
    if ctx.args.nested:
        lines += ["", nested(ctx)]
    ctx.note("p6", "정규화 → 증강 → 모델 비교", lines,
             ["p6_normalization_view.png", "p6_normalization_acc.png", "p6_models_paired.png",
              f"p6_confusion_{best_m}.png"])
    summary(ctx)


def nested(ctx: Ctx) -> str:
    """정규화 선택까지 한 사람씩 빼고 고르기 → 선택 과정이 시험 사람을 보지 않은 정확도"""
    W, ch6, y, person, clean, test = ctx.W_fix, ctx.ch6, ctx.y, ctx.person, ctx.clean, ctx.test
    Fs = {k: features(W, ch6, k, ctx.base) for k in NORMS}
    accs = {}
    for p in sorted(person.unique()):
        inner = person != p
        scores = {k: loso(F[inner], y[inner], person[inner], test[inner], clean[inner],
                          n_trees=max(60, ctx.n_trees // 2))["mean"] for k, F in Fs.items()}
        k = max(scores, key=scores.get)
        F = Fs[k]
        tr = clean & inner
        M, my = mirror(F[tr], y[tr])
        m = make_model("rf", ctx.n_trees).fit(pd.concat([F[tr], M]), pd.concat([y[tr], my]))
        te = test & (person == p)
        accs[p] = (k, float((m.predict(F[te]) == y[te]).mean()))
        log.info("  중첩 검증 %s: %s 선택 → %.1f%%", p, k, accs[p][1] * 100)
    mean = float(np.mean([v[1] for v in accs.values()]))
    (ctx.out / "p6_nested.json").write_text(json.dumps(accs, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"- 중첩 검증 (정규화 선택도 시험 사람 없이): {mean:.1%} — 사람별 선택 "
            + ", ".join(f"{p}:{k}" for p, (k, _) in accs.items()))


def summary(ctx: Ctx):
    keys = [k for k in ("s0", "s1", "s2", "s3", "s4") if k in ctx.cache]
    best = max(NORMS, key=lambda k: ctx.cache.get(f"norm_{k}", {"mean": -1})["mean"])
    if f"norm_{best}" in ctx.cache:
        ctx.cache["s5"] = dict(ctx.cache[f"norm_{best}"], label=f"S5 + 기준 대비 정규화 ({NORMS[best]})")
        keys.append("s5")
    if "s6" in ctx.cache:
        keys.append("s6")
    t = ctx.acc_table(keys)
    t.to_csv(ctx.out / "steps.csv", index=False, encoding="utf-8-sig")
    plot_acc_bars(t, "step", ctx.out / "steps_acc.png", "전처리 단계별 정확도 (RF, 시험 데이터는 모든 단계 동일)", ctx.plt)
    (ctx.out / "summary.md").write_text("# 전처리 단계별 정확도\n\n" + md_table(t[["step", "acc_mean", "diff_pp",
                                                                             "macro_f1"]]), encoding="utf-8")
    print(md_table(t[["step", "acc_mean", "diff_pp", "macro_f1"]]))


def p7(ctx: Ctx):
    """재생 테스트로 후처리 재검증 (아직 없음)"""
    print("P7(재생 테스트로 후처리 재검증)은 전처리 방식을 새로 정한 뒤 실시간 판정기(ai/engine/seat.py)를 "
          "새 데이터 형식에 맞춰 고치고 만든다. 아직 없음.")


STEPS = {"p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[report] %(message)s")
    ap = argparse.ArgumentParser(description="전처리 단계별 그래프 + 정확도 (2차 수집 11명)")
    ap.add_argument("--step", nargs="+", default=["all"], choices=["all", *STEPS],
                    help="실행할 단계 (예: --step p0, --step p3 p4). 기본: 전부 순서대로")
    ap.add_argument("--folder", default="data/dataset")
    ap.add_argument("--out", default="data/processed/report")
    ap.add_argument("--fast", action="store_true", help="나무 수를 줄여 빠르게 (흐름 확인용)")
    ap.add_argument("--rerun", action="store_true", help="저장된 정확도(results.json)를 무시하고 다시 계산")
    ap.add_argument("--nested", action="store_true", help="P6에서 중첩 검증까지 (+15분 정도)")
    ap.add_argument("--no-timelines", action="store_true", help="사람별 시계열 그림 생략 (빠르게)")
    args = ap.parse_args()
    from common.config import repo_path
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args, out)
    steps = list(STEPS) if "all" in args.step else args.step
    t0 = time.time()
    for s in steps:
        log.info("===== %s: %s =====", s.upper(), STEPS[s].__doc__.strip().splitlines()[0])
        STEPS[s](ctx)
    print(f"저장: {show_path(out)}/  ({(time.time() - t0) / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
