"""단계 판정 보고서 — 지금 장비(포화된 압력센서)로 무엇을 얼마나 판정할 수 있는지, 그래프·표로 (2차 수집 11명)

왜 단계로 나누나 (preprocess_report P0~P6 결과)
    5개 자세를 한 번에 구분하면 처음 보는 사람 기준 30~40%. 원인을 보니 압력센서가 '얼마나 눌렸나'가 아니라
    '닿았나/안 닿았나'만 재고 있어서(0 아니면 3500 근처, 포화), 세밀한 자세 구분 정보가 없다.
    그래서 판정을 쉬운 것부터 단계로 나누고, 단계마다 얼마나 되는지 따로 잰다.

실행 (저장소 최상위, .venv 켠 상태):
    python -m ai.analysis.staged_report                 # 전부 (약 10~15분)
    python -m ai.analysis.staged_report --step dev      # 기준 대비 그래프만 (학습 없음, 1분)
    python -m ai.analysis.staged_report --step tasks    # 단계별 학습·모델 비교
    python -m ai.analysis.staged_report --step depth    # 거리센서로 앉은 위치 판정
    python -m ai.analysis.staged_report --step app      # 단계를 이은 앱 최종 출력 정확도
    옵션: --fast (빠르게) / --aug (좌우 대칭 증강 켜기)

결과: data/processed/staged/
    dev_heatmap_people.png    사람마다 '자세 × 센서' 기준 대비 변화 (0 근처 = 자세를 바꿔도 센서가 안 변함)
    dev_heatmap_mean.png      문제 세션 제외 평균 + 사람 간 일치도
    dev_timelines/pXX.png     기준 대비 변화 시계열
    T*_models.png / .csv      단계마다 모델 5종 (막대 = 평균, 점 = 사람별)
    T*_confusion.png          그 단계 최고 모델 혼동 행렬
    conditions.csv / .png     조건 비교 (문제 세션 포함/제외 × 기준 1개 / 위치별 기준)
    depth.png / .csv          거리센서 앉은 위치 판정
    app.csv / app_confusion.png  최종 앱 출력 (바른자세 / 왼쪽에 체중 / 오른쪽에 체중) 정확도
    summary.md                전체 숫자 정리 (노션용 표)

평가 규칙 (preprocess_report와 같음)
    - 한 사람 빼고 학습 → 그 사람으로 시험 (11번). 정확도 = 사람별 정확도 평균
    - 시험은 항상 11명 전원. 문제 세션 제외는 '학습'에서만 (결과를 좋게 보이려고 시험에서 빼지 않음)
    - 기준값 구간(바른자세(중) 첫 10초가 들어 있는 구간)은 시험에서 뺀다 (기준으로 쓴 데이터를 맞히는 건 의미 없음)
    - 2초(4개) 묶음 하나 = 샘플 하나. 시계열 모델이 아니라 묶음마다 따로 분류한다
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from ai.analysis.dataset_v2 import ADC_MAX
from ai.analysis.preprocess_report import (
    CH_KO, LABEL_KO, LABELS, Ctx, features, make_model, md_table, mirror, _plt,
)
from ai.features.pressure_pipeline import show_path

log = logging.getLogger("sitsense.staged")

BAD_PEOPLE_RULE = "P1 센서 진단: 앞쪽 센서 고장 또는 바르게 앉아도 안 눌리고 자세 절반 이상에서 0"
DEPTH_STEP_CM = 10.0          # 바른자세 기준보다 머리가 10cm 넘게 가까우면 앞, 멀면 뒤 (데이터로 맞추지 않은 값)
MODELS = ("logreg", "knn", "svm", "rf", "hgb")
MODEL_KO = {"logreg": "로지스틱 회귀", "knn": "KNN", "svm": "SVM", "rf": "Random Forest", "hgb": "Gradient Boosting"}
# 체중이 실린 쪽. 다리 꼬기는 꼰 다리가 들려서 '반대쪽'에 체중이 실린다 (왼다리꼬기 → 오른쪽)
# 처음에 왼다리꼬기를 '왼쪽'으로 묶었더니 압력상 반대 방향끼리 섞여 57%였고, 체중 기준으로 묶으니 79% (RF)
SIDE = {"normal": "none", "lean_left": "left", "cross_right": "left", "lean_right": "right", "cross_left": "right"}
KIND = {"lean_left": "lean", "lean_right": "lean", "cross_left": "cross", "cross_right": "cross"}
NAME_KO = {"normal": "바른자세", "bad": "나쁜 자세", "lean": "편향", "cross": "다리 꼬기", "left": "왼쪽에 체중",
           "right": "오른쪽에 체중", "none": "바른자세", **LABEL_KO}
SWAP_ANY = {"left": "right", "right": "left", "lean_left": "lean_right", "lean_right": "lean_left",
            "cross_left": "cross_right", "cross_right": "cross_left"}


# ---------------------------------------------------------------------------
# 판정 단계 정의
# ---------------------------------------------------------------------------
def tasks(y: pd.Series) -> list[dict]:
    """(코드, 이름, 쓰는 데이터, 정답)"""
    bad = y != "normal"
    return [
        {"key": "T1", "name": "① 바른자세 vs 나쁜 자세", "mask": pd.Series(True, index=y.index),
         "target": y.where(y == "normal", "bad")},
        {"key": "T2", "name": "② (나쁜 자세 중) 편향 vs 다리 꼬기", "mask": bad, "target": y.map(KIND)},
        {"key": "T3", "name": "③ (나쁜 자세 중) 체중이 실린 쪽 왼쪽 vs 오른쪽", "mask": bad, "target": y.map(SIDE)},
        {"key": "T3a", "name": "③-1 (편향 중) 왼쪽 vs 오른쪽", "mask": y.str.startswith("lean"), "target": y},
        {"key": "T3b", "name": "③-2 (다리 꼬기 중) 왼쪽 vs 오른쪽", "mask": y.str.startswith("cross"), "target": y},
        {"key": "T4", "name": "참고: 다리 꼬기 빼고 3개 (바른/왼편향/오른편향)", "mask": ~y.str.startswith("cross"),
         "target": y},
        {"key": "T5", "name": "참고: 5개 자세 한 번에", "mask": pd.Series(True, index=y.index), "target": y},
    ]


# ---------------------------------------------------------------------------
# 머리 거리 · 앉은 위치 추정 · 위치별 기준
# ---------------------------------------------------------------------------
def window_head(ctx: Ctx) -> pd.Series:
    """2초 묶음마다 머리 거리 중앙값 (측정 실패는 사람 안에서 직전 값)"""
    df = ctx.repaired[0]
    dh = df["dist_head"].where(df["dist_head"].between(15, 150))
    dh = dh.groupby(df["person"]).ffill()
    W = ctx.W_fix
    return pd.Series([np.nanmedian(dh.loc[r:r + 3].to_numpy()) for r in W["row0"]], index=W.index)


def head_baseline(ctx: Ctx) -> pd.Series:
    """사람마다 바른자세(중) 첫 10초 머리 거리 (압력 기준값과 같은 구간)"""
    df = ctx.repaired[0]
    out = {}
    for p, g in df.groupby("person"):
        b = g[(g["cls"] == "바른자세") & (g["depth"] == "middle") & ~g["rest"] & g["last_attempt"]].iloc[4:]
        b = b[~ctx.moving.reindex(b.index, fill_value=False)]
        v = b["dist_head"].where(b["dist_head"].between(15, 150)).dropna()
        out[p] = float(v.head(20).median())
    return pd.Series(out)


def estimate_depth(ctx: Ctx) -> pd.Series:
    rel = window_head(ctx) - ctx.W_fix["person"].map(head_baseline(ctx))
    est = pd.Series("middle", index=rel.index)
    est[rel < -DEPTH_STEP_CM] = "front"
    est[rel > DEPTH_STEP_CM] = "back"
    return est


def depth_baselines(ctx: Ctx, depth: pd.Series) -> pd.DataFrame:
    """줄마다 '그 위치의 바른자세' 압력 기준. 위치는 depth(추정 또는 라벨)로 고른다"""
    fixed = ctx.repaired[0]
    use = fixed[(fixed["cls"] == "바른자세") & ~fixed["rest"] & fixed["last_attempt"] & ~ctx.moving]
    b = {(p, d): g[ctx.ch6].iloc[4:24].median() for (p, d), g in use.groupby(["person", "depth"])}
    W = ctx.W_fix
    return pd.DataFrame([b[(p, d)].to_numpy() for p, d in zip(W["person"], depth)], index=W.index, columns=ctx.ch6)


def feature_sets(ctx: Ctx) -> dict:
    W, ch = ctx.W_fix, ctx.ch6
    def make():
        return {
            "기준 1개": features(W, ch, "B", ctx.base),
            "위치별 기준 (거리센서 추정)": features(W, ch, "B", depth_baselines(ctx, estimate_depth(ctx))),
            "위치별 기준 (라벨, 분석용 상한)": features(W, ch, "B", depth_baselines(ctx, W["depth"])),
        }
    return ctx._get("feature_sets", make)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def loso_task(ctx: Ctx, F, target, mask, model, exclude=(), aug=False, n_trees=200):
    """한 사람씩 빼고. 학습 = mask & clean (& 제외 사람 아님), 시험 = mask & 마지막 시도 & 기준 구간 아님"""
    from sklearn.metrics import balanced_accuracy_score
    person = ctx.person
    base_seg = ctx.W_fix["seg"].isin(base_segments(ctx))
    train_ok = mask & ctx.clean & ~person.isin(exclude)
    test_ok = mask & ctx.test & ~base_seg
    pred = pd.Series(index=F.index, dtype=object)
    acc, bal = {}, {}
    for p in sorted(person.unique()):
        tr, te = train_ok & (person != p), test_ok & (person == p)
        if not te.any():
            continue
        Xtr, ytr = F[tr], target[tr]
        if aug:
            M, _ = mirror(Xtr, ytr)
            Xtr, ytr = pd.concat([Xtr, M]), pd.concat([ytr, ytr.map(lambda v: SWAP_ANY.get(v, v))])
        m = make_model(model, n_trees).fit(Xtr, ytr)
        yp = m.predict(F[te])
        pred[te] = yp
        acc[p] = float((yp == target[te]).mean())
        bal[p] = float(balanced_accuracy_score(target[te], yp))
    return {"acc": acc, "bal": bal, "mean": float(np.mean(list(acc.values()))),
            "bal_mean": float(np.mean(list(bal.values()))), "pred": pred, "test": test_ok}


def base_segments(ctx: Ctx) -> list:
    W = ctx.W_fix
    return list(W.loc[(W["y"] == "normal") & (W["depth"] == "middle") & ctx.is_last, "seg"].unique())


# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------
POSTURE_ORDER = [f"{c}({d})" for c in ("바른자세", "왼쪽편향", "오른쪽편향", "왼다리꼬기", "오른다리꼬기")
                 for d in ("앞", "중", "뒤")]


def dev_matrix(ctx: Ctx) -> dict[str, pd.DataFrame]:
    """사람마다 자세 × 센서 '기준 대비 변화' = (자세 구간 중앙값 − 바른자세(중) 기준) / 최대치 × 100 (%p)
    나눗셈 기준을 최대치로 둬서 기준이 0 근처인 센서도 폭발하지 않는다"""
    fixed = ctx.repaired[0]
    use = fixed[~fixed["rest"] & fixed["last_attempt"] & ~ctx.moving]
    out = {}
    for p, g in use.groupby("person"):
        med = g.groupby("label")[ctx.ch6].median()
        d = (med - ctx.base[p][ctx.ch6]) / ADC_MAX * 100
        out[p] = d.reindex([l for l in POSTURE_ORDER if l in d.index])
    return out


def plot_dev(ctx: Ctx, plt):
    D = dev_matrix(ctx)
    bad = ctx.flags.loc[ctx.flags["flagged"], "person"].tolist()
    pp = sorted(D)
    cols = 4
    rows = int(np.ceil(len(pp) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.3 * rows))
    for ax in axes.flat[len(pp):]:
        ax.axis("off")
    for ax, p in zip(axes.flat, pp):
        M = D[p]
        im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-80, vmax=80, aspect="auto")
        ax.set_xticks(range(len(M.columns)), [CH_KO[c] for c in M.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(M.index)), M.index, fontsize=7)
        flat = (M.abs() < 5).to_numpy().mean() * 100
        ax.set_title(f"{p}{' ⚠' if p in bad else ''}  (거의 안 변한 칸 {flat:.0f}%)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="기준 대비 변화 (최대치 대비 %p)  파랑 = 빠짐, 빨강 = 더 눌림")
    fig.suptitle("사람마다 자세별 센서 변화 (바른자세(중) 기준). 흰 칸 = 자세를 바꿔도 센서가 안 변함 (포화·미접촉)", y=1.0)
    fig.savefig(ctx.out / "dev_heatmap_people.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    good = [p for p in pp if p not in bad]
    stack = np.stack([D[p].reindex(D[good[0]].index).to_numpy() for p in good])
    mean = np.nanmean(stack, axis=0)
    agree = np.nanmean(np.sign(stack) == np.sign(mean), axis=0) * 100
    idx, colsn = D[good[0]].index, [CH_KO[c] for c in D[good[0]].columns]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im = axes[0].imshow(mean, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
    axes[1].imshow(agree, cmap="Greens", vmin=50, vmax=100, aspect="auto")
    for ax, M, fmt in ((axes[0], mean, "{:+.0f}"), (axes[1], agree, "{:.0f}%")):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(colsn)), colsn, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(idx)), idx, fontsize=8)
    axes[0].set_title(f"평균 기준 대비 변화 (문제 세션 제외 {len(good)}명, %p)")
    axes[1].set_title("변화 방향이 평균과 같은 사람 비율 (높을수록 사람끼리 일치)")
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    fig.tight_layout()
    fig.savefig(ctx.out / "dev_heatmap_mean.png", dpi=130)
    plt.close(fig)
    rows = []
    for p in pp:
        rows.append({"person": p, "flagged": p in bad, "flat_cells_pct": round((D[p].abs() < 5).to_numpy().mean() * 100, 1),
                     "mean_abs_change_pp": round(D[p].abs().to_numpy().mean(), 1)})
    return pd.DataFrame(rows), pd.DataFrame(mean, index=idx, columns=colsn), pd.DataFrame(agree, index=idx, columns=colsn)


def plot_dev_timeline(ctx: Ctx, plt):
    (ctx.out / "dev_timelines").mkdir(exist_ok=True)
    fixed = ctx.repaired[0]
    colors = plt.get_cmap("tab10")
    for p, g in fixed.groupby("person", sort=False):
        d = (g[ctx.ch6] - ctx.base[p][ctx.ch6]) / ADC_MAX * 100
        d = d.rolling(4, min_periods=1).median()
        t = np.arange(len(g)) * 0.5 / 60
        fig, ax = plt.subplots(figsize=(16, 4.5))
        for i, c in enumerate(ctx.ch6):
            ax.plot(t, d[c], lw=0.8, color=colors(i), label=CH_KO[c])
        ax.fill_between(t, -100, 100, where=g["rest"].to_numpy(), color="#bdbdbd", alpha=0.5, lw=0)
        ax.axhline(0, color="black", lw=0.6)
        starts = g.index[(g["seg"] != g["seg"].shift()) & ~g["rest"]]
        for s in starts:
            ax.text(t[g.index.get_loc(s)], 92, g.loc[s, "label"], fontsize=6, rotation=30)
        ax.set_ylim(-100, 100)
        ax.set_ylabel("기준 대비 변화 (%p)")
        ax.set_xlabel("분")
        ax.legend(fontsize=7, ncol=6, loc="lower right")
        ax.set_title(f"{p} 기준 대비 압력 변화 (0 = 바른자세(중)와 같음, 회색 = 쉬는시간)")
        fig.tight_layout()
        fig.savefig(ctx.out / "dev_timelines" / f"{p}.png", dpi=110)
        plt.close(fig)


def plot_models(tab: pd.DataFrame, title: str, chance: float, path, plt):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(tab))
    pcols = [c for c in tab.columns if c.startswith("p") and c[1:].isdigit()]
    ax.bar(x, tab["acc_mean"] * 100, color="#1e88e5", alpha=0.85)
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.scatter(np.full(len(pcols), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(pcols)),
                   r[pcols].astype(float) * 100, s=10, color="black", zorder=3)
        ax.text(i, r["acc_mean"] * 100 + 2, f"{r['acc_mean']:.1%}", ha="center", fontsize=9)
    ax.axhline(chance * 100, color="red", ls="--", lw=1, label=f"찍기 수준 {chance:.0%}")
    ax.set_xticks(x, tab["model_ko"], rotation=15)
    ax.set_ylim(0, 105)
    ax.set_ylabel("정확도 % (막대 = 평균, 점 = 사람별)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_conf(y, yp, labels, title, path, plt):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp, labels=labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2.5, 1.1 * len(labels) + 2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ko = [NAME_KO.get(l, l) for l in labels]
    ax.set_xticks(range(len(labels)), ko, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), ko)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 단계
# ---------------------------------------------------------------------------
def step_dev(ctx: Ctx, S: dict):
    """기준 대비 그래프"""
    plt = ctx.plt
    flat, mean, agree = plot_dev(ctx, plt)
    flat.to_csv(ctx.out / "dev_flat_cells.csv", index=False, encoding="utf-8-sig")
    mean.round(1).to_csv(ctx.out / "dev_mean.csv", encoding="utf-8-sig")
    agree.round(0).to_csv(ctx.out / "dev_agreement.csv", encoding="utf-8-sig")
    if not ctx.args.no_timelines:
        plot_dev_timeline(ctx, plt)
    S["dev"] = {"flat": flat}
    print(md_table(flat))


def step_tasks(ctx: Ctx, S: dict):
    """단계별 학습: 조건 비교(HGB) → 모델 5종 비교(가장 현실적인 조건)"""
    plt, y = ctx.plt, ctx.y
    bad = ctx.flags.loc[ctx.flags["flagged"], "person"].tolist()
    Fs = feature_sets(ctx)
    T = tasks(y)
    n_trees = ctx.n_trees
    # 1) 조건 비교 (Gradient Boosting)
    crows = []
    for t in T:
        for fname, F in Fs.items():
            for excl in (False, True):
                t0 = time.time()
                r = loso_task(ctx, F, t["target"], t["mask"], "hgb", bad if excl else (), ctx.args.aug, n_trees)
                crows.append({"task": t["key"], "task_name": t["name"], "features": fname,
                              "train": "문제 세션 제외" if excl else "전원", "acc_mean": round(r["mean"], 4),
                              "balanced": round(r["bal_mean"], 4)})
                log.info("  %-4s %-26s %-8s %.1f%% (%.0f초)", t["key"], fname, crows[-1]["train"], r["mean"] * 100,
                         time.time() - t0)
    cond = pd.DataFrame(crows)
    cond.to_csv(ctx.out / "conditions.csv", index=False, encoding="utf-8-sig")
    piv = cond.pivot_table(index="task_name", columns=["features", "train"], values="acc_mean", sort=False)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    piv.plot.bar(ax=ax, width=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("정확도 (Gradient Boosting)")
    ax.set_xticklabels([l.get_text() for l in ax.get_xticklabels()], rotation=15, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("조건 비교: 기준 방식 × 문제 세션 학습 포함/제외 (시험은 항상 11명 전원)")
    fig.tight_layout()
    fig.savefig(ctx.out / "conditions.png", dpi=130)
    plt.close(fig)

    # 2) 모델 비교: 실제 서비스에서 쓸 수 있는 조건 중 가장 좋은 것
    real = cond[cond["features"] != "위치별 기준 (라벨, 분석용 상한)"]
    best_cond = real.groupby(["features", "train"])["acc_mean"].mean().idxmax()
    log.info("모델 비교 조건: %s, %s", *best_cond)
    F = Fs[best_cond[0]]
    excl = bad if best_cond[1] == "문제 세션 제외" else ()
    rows, results = [], {}
    for t in T:
        mrows = []
        for m in MODELS:
            r = loso_task(ctx, F, t["target"], t["mask"], m, excl, ctx.args.aug, n_trees)
            results[(t["key"], m)] = r
            d = {"model": m, "model_ko": MODEL_KO[m], "acc_mean": round(r["mean"], 4), "balanced": round(r["bal_mean"], 4)}
            d.update({p: round(v, 4) for p, v in r["acc"].items()})
            mrows.append(d)
            log.info("  %-4s %-18s %.1f%%", t["key"], MODEL_KO[m], r["mean"] * 100)
        mt = pd.DataFrame(mrows)
        mt.to_csv(ctx.out / f"{t['key']}_models.csv", index=False, encoding="utf-8-sig")
        chance = 1 / t["target"][t["mask"]].nunique()
        plot_models(mt, f"{t['name']}  ({best_cond[0]}, 학습 {best_cond[1]})", chance,
                    ctx.out / f"{t['key']}_models.png", plt)
        bm = mt.loc[mt["acc_mean"].idxmax(), "model"]
        r = results[(t["key"], bm)]
        labels = sorted(t["target"][t["mask"]].dropna().unique(),
                        key=lambda v: (LABELS + ["bad", "lean", "cross", "left", "right"]).index(v)
                        if v in LABELS + ["bad", "lean", "cross", "left", "right"] else 99)
        te = r["test"]
        plot_conf(t["target"][te], r["pred"][te], labels, f"{t['name']} — {MODEL_KO[bm]}",
                  ctx.out / f"{t['key']}_confusion.png", plt)
        rows.append({"단계": t["name"], "찍기 수준": f"{chance:.0%}", "최고 모델": MODEL_KO[bm],
                     "정확도": float(mt["acc_mean"].max()), "균형 정확도": float(mt.loc[mt["acc_mean"].idxmax(), "balanced"]),
                     **{MODEL_KO[m]: float(mt.loc[mt["model"] == m, "acc_mean"].iloc[0]) for m in MODELS}})
    st = pd.DataFrame(rows)
    st.to_csv(ctx.out / "tasks_summary.csv", index=False, encoding="utf-8-sig")
    S["tasks"] = {"summary": st, "conditions": cond, "best_cond": best_cond, "results": results, "F": F, "excl": excl}
    print(md_table(pct(st)))


def step_depth(ctx: Ctx, S: dict):
    """거리센서로 앉은 위치(앞/중/뒤) 판정"""
    W = ctx.W_fix
    est = estimate_depth(ctx)
    m = ctx.test & ~W["seg"].isin(base_segments(ctx))
    per = (est[m] == W.loc[m, "depth"]).groupby(W.loc[m, "person"]).mean()
    rel = window_head(ctx) - W["person"].map(head_baseline(ctx))
    plt = ctx.plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for d, col in (("front", "#e53935"), ("middle", "#43a047"), ("back", "#1e88e5")):
        axes[0].hist(rel[m & (W["depth"] == d)].dropna(), bins=np.arange(-60, 61, 2), alpha=0.6, color=col,
                     label={"front": "앞", "middle": "중", "back": "뒤"}[d])
    for v in (-DEPTH_STEP_CM, DEPTH_STEP_CM):
        axes[0].axvline(v, color="black", ls="--")
    axes[0].set_xlabel("머리 거리 − 바른자세 기준 (cm)")
    axes[0].legend()
    axes[0].set_title(f"앉은 위치별 머리 거리 변화 (점선 = ±{DEPTH_STEP_CM:g}cm 기준)")
    axes[1].bar(per.index, per * 100, color="#43a047")
    axes[1].set_ylim(0, 105)
    axes[1].set_title(f"사람별 앉은 위치 정확도 (평균 {per.mean():.1%}, 찍기 33%)")
    fig.tight_layout()
    fig.savefig(ctx.out / "depth.png", dpi=130)
    plt.close(fig)
    pd.DataFrame({"person": per.index, "acc": per.round(3).values}).to_csv(ctx.out / "depth.csv", index=False,
                                                                          encoding="utf-8-sig")
    plot_conf(W.loc[m, "depth"], est[m], ["front", "middle", "back"], "앉은 위치 (머리 거리 기준 대비 ±10cm)",
              ctx.out / "depth_confusion.png", plt)
    S["depth"] = float(per.mean())
    print(f"앉은 위치 정확도: {per.mean():.1%} (사람별 {per.round(2).to_dict()})")


def step_app(ctx: Ctx, S: dict):
    """단계를 이어서 앱 최종 출력(바른자세 / 왼쪽에 체중 / 오른쪽에 체중) 정확도"""
    if "tasks" not in S:
        step_tasks(ctx, S)
    tk = S["tasks"]
    y = ctx.y
    F, excl = tk["F"], tk["excl"]
    t1 = [t for t in tasks(y) if t["key"] == "T1"][0]
    t3 = [t for t in tasks(y) if t["key"] == "T3"][0]
    best = lambda key: max(MODELS, key=lambda m: tk["results"][(key, m)]["mean"])   # noqa: E731
    m1, m3 = best("T1"), best("T3")
    r1 = tk["results"][("T1", m1)]
    # ③은 '나쁜 자세'로 판정된 묶음에 적용해야 하므로, 나쁜 자세 데이터로 학습한 모델을 전체 시험 묶음에 다시 적용
    person = ctx.person
    out = pd.Series(index=F.index, dtype=object)
    te = r1["test"]
    for p in sorted(person.unique()):
        tr = t3["mask"] & ctx.clean & ~person.isin(excl) & (person != p)
        mdl = make_model(m3, ctx.n_trees).fit(F[tr], t3["target"][tr])
        k = te & (person == p)
        side = pd.Series(mdl.predict(F[k]), index=F[k].index)
        out[k] = np.where(r1["pred"][k] == "normal", "none", side)
    truth = y.map(SIDE)
    acc = (out[te] == truth[te]).groupby(person[te]).mean()
    from sklearn.metrics import balanced_accuracy_score
    bal = np.mean([balanced_accuracy_score(truth[te & (person == p)], out[te & (person == p)]) for p in acc.index])
    plot_conf(truth[te], out[te], ["none", "left", "right"],
              f"앱 최종 출력 (①{MODEL_KO[m1]} → ③{MODEL_KO[m3]})  정확도 {acc.mean():.1%}", ctx.out / "app_confusion.png",
              ctx.plt)
    pd.DataFrame({"person": acc.index, "acc": acc.round(3).values}).to_csv(ctx.out / "app.csv", index=False,
                                                                          encoding="utf-8-sig")
    S["app"] = {"acc": float(acc.mean()), "bal": float(bal), "m1": m1, "m3": m3}
    print(f"앱 최종 출력 (바른자세/왼쪽에 체중/오른쪽에 체중): {acc.mean():.1%}, 균형 {bal:.1%} (찍기 33%)")


def pct(t: pd.DataFrame) -> pd.DataFrame:
    """0~1 숫자 열을 % 글자로 (노션 표용)"""
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f" and t[c].between(0, 1).all():
            t[c] = t[c].map(lambda v: f"{v:.1%}")
    return t


def write_summary(ctx: Ctx, S: dict):
    lines = [f"# 단계 판정 보고서 ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"- 데이터: {len(ctx.people)}명, 시험 = 11명 전원 (기준값 구간 제외)",
             f"- 문제 세션 규칙: {BAD_PEOPLE_RULE} → {', '.join(ctx.flags.loc[ctx.flags.flagged, 'person']) or '없음'}",
             f"- 좌우 대칭 증강: {'켬' if ctx.args.aug else '끔'}"]
    if "dev" in S:
        f = S["dev"]["flat"]
        lines += ["", "## 기준 대비 변화 (포화 확인)",
                  "자세를 바꿔도 거의 안 변한 칸(5%p 미만) 비율 — 높을수록 센서가 자세를 못 잡음", "", md_table(f)]
    if "tasks" in S:
        st = S["tasks"]["summary"].copy()
        lines += ["", "## 단계별 정확도 (모델 비교 조건: " + ", ".join(S["tasks"]["best_cond"]) + ")", "",
                  md_table(pct(st)), "", "## 조건 비교 (Gradient Boosting)", "",
                  md_table(pct(S["tasks"]["conditions"][["task_name", "features", "train", "acc_mean", "balanced"]]))]
    if "depth" in S:
        lines += ["", f"## 앉은 위치 (머리 거리 − 바른자세 기준, ±{DEPTH_STEP_CM:g}cm): {S['depth']:.1%} (찍기 33%)"]
    if "app" in S:
        a = S["app"]
        lines += ["", f"## 앱 최종 출력 (바른자세 / 왼쪽에 체중 / 오른쪽에 체중): {a['acc']:.1%}, 균형 {a['bal']:.1%} "
                      f"(① {MODEL_KO[a['m1']]} → ③ {MODEL_KO[a['m3']]}, 찍기 33%)"]
    (ctx.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


STEPS = {"dev": step_dev, "tasks": step_tasks, "depth": step_depth, "app": step_app}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[staged] %(message)s")
    ap = argparse.ArgumentParser(description="단계 판정 보고서 (2차 수집 11명)")
    ap.add_argument("--step", nargs="+", default=["all"], choices=["all", *STEPS])
    ap.add_argument("--folder", default="data/dataset")
    ap.add_argument("--out", default="data/processed/staged")
    ap.add_argument("--fast", action="store_true", help="나무 수를 줄여 빠르게")
    ap.add_argument("--aug", action="store_true", help="좌우 대칭 증강 켜기 (P6에서 이 데이터에는 효과 없었음)")
    ap.add_argument("--no-timelines", action="store_true", help="기준 대비 시계열 그림 생략")
    args = ap.parse_args()
    args.rerun, args.nested = True, False
    from common.config import repo_path
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args, out)
    S: dict = {}
    t0 = time.time()
    for s in (list(STEPS) if "all" in args.step else args.step):
        log.info("===== %s: %s =====", s, STEPS[s].__doc__.strip().splitlines()[0])
        STEPS[s](ctx, S)
    write_summary(ctx, S)
    print(f"저장: {show_path(out)}/  ({(time.time() - t0) / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
