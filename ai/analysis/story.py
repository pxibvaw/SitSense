"""발표 스토리 실험 — 단계마다 명령 하나, 그래프 한 장(전 → 후), 숫자 표 하나 (2차 수집 11명)

모든 그림은 실제 수집 데이터(data/dataset/)를 코드로 계산해 그린 그래프다 (예시·개념 그림 없음).

실행 (저장소 최상위, .venv 켠 상태). 번호 순서대로 하나씩:
    python -m ai.analysis.story --step 0      # 1차 3명 결과 (저장된 표를 읽음)
    python -m ai.analysis.story --step 1      # 11명 원본 학습
    …
    python -m ai.analysis.story --step 17     # 앱 최종 출력
    python -m ai.analysis.story --list        # 단계 목록
    옵션: --fast (나무 수 줄여 빠르게)

결과: data/processed/story/
    NN_제목.png      발표 한 장 (왼쪽 = 전, 오른쪽 = 후, 아래 띠 = 정확도 변화)
    NN_표.csv        그 단계 숫자
    results.json     단계별 정확도 (다음 단계 그림의 '전' 값으로 씀)

평가 규칙 (모든 단계 공통)
    - 한 사람 빼고 학습 → 그 사람으로 시험 (11번). 콘솔에 사람마다 학습·시험 개수와 정확도가 찍힌다
    - 정확도 = 사람별 정확도 평균. 시험은 항상 11명 전원
    - 1~7단계(5개 자세)는 마지막 시도 전부로 시험, 12단계부터(단계 판정)는 기준값 구간(바른자세(중))을 시험에서 뺌
    - 2초(4개) 묶음 하나 = 샘플 하나 (시계열 모델 아님)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from ai.analysis.dataset_v2 import ADC_MAX
from ai.analysis.preprocess_report import (
    CH_KO, LABEL_KO, LABELS, NORMS, STATUS_COLOR, STATUS_KO, Ctx, features, make_model, mirror, _plt,
)
from ai.analysis.staged_report import (
    DEPTH_STEP_CM, KIND, MODEL_KO, MODELS, NAME_KO, SIDE, SWAP_ANY, base_segments, depth_baselines, dev_matrix,
    estimate_depth, head_baseline, window_head,
)
from ai.features.pressure_pipeline import show_path

log = logging.getLogger("sitsense.story")
SIDE_WRONG = {"normal": "none", "lean_left": "w_left", "cross_left": "w_left", "lean_right": "w_right",
              "cross_right": "w_right"}
NAME_KO.update({"w_left": "왼쪽 (기운·꼰 쪽)", "w_right": "오른쪽 (기운·꼰 쪽)"})
DEPTH_KO = {"front": "앞", "middle": "중", "back": "뒤"}
BEFORE, AFTER = "#90a4ae", "#1e88e5"


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------
class Story:
    def __init__(self, args):
        from common.config import repo_path
        self.out = repo_path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        a = argparse.Namespace(folder=args.folder, fast=args.fast, rerun=True, nested=False, no_timelines=True)
        self.ctx = Ctx(a, self.out)
        self.plt = _plt()
        self.n_trees = 60 if args.fast else 200
        self.res_path = self.out / "results.json"
        self.res = json.loads(self.res_path.read_text(encoding="utf-8")) if self.res_path.exists() else {}

    def save(self, key, r):
        self.res[key] = {"mean": r["mean"], "acc": r["acc"]}
        self.res_path.write_text(json.dumps(self.res, ensure_ascii=False, indent=1), encoding="utf-8")

    def get(self, key, fn):
        """다른 단계 결과가 필요하면 저장된 값, 없으면 계산"""
        if key not in self.res:
            log.info("  (%s 결과가 없어 먼저 계산)", key)
            self.save(key, fn())
        return self.res[key]

    def loso(self, title, F, target, test, train, model="rf", exclude=(), aug=False, show=True):
        """한 사람 빼고 학습 → 시험, 사람마다 콘솔에 출력"""
        from sklearn.metrics import balanced_accuracy_score
        person = self.ctx.person
        if show:
            print(f"\n[학습] {title}  (모델: {MODEL_KO.get(model, model)}{', 좌우 대칭 증강' if aug else ''}"
                  f"{', 학습 제외 ' + ','.join(exclude) if exclude else ''})")
        pred = pd.Series(index=F.index, dtype=object)
        acc, bal = {}, {}
        for p in sorted(person.unique()):
            tr = train & (person != p) & ~person.isin(exclude)
            te = test & (person == p)
            if not te.any():
                continue
            Xtr, ytr = F[tr], target[tr]
            if aug:
                M, _ = mirror(Xtr, ytr)
                Xtr, ytr = pd.concat([Xtr, M]), pd.concat([ytr, ytr.map(lambda v: SWAP_ANY.get(v, v))])
            m = make_model(model, self.n_trees).fit(Xtr, ytr)
            yp = m.predict(F[te])
            pred[te] = yp
            acc[p] = float((yp == target[te]).mean())
            bal[p] = float(balanced_accuracy_score(target[te], yp))
            if show:
                print(f"  {p} 빼고 학습 {len(Xtr):>6,}개 → {p} 시험 {int(te.sum()):>5,}개: {acc[p]:.1%}")
        mean = float(np.mean(list(acc.values())))
        if show:
            print(f"  평균 {mean:.1%}  (균형 정확도 {np.mean(list(bal.values())):.1%})")
        return {"mean": mean, "acc": acc, "bal": float(np.mean(list(bal.values()))), "pred": pred, "test": test}

    # 그림 ------------------------------------------------------------------
    def figure(self, n_cols=2, width=15, height=5.2, ratios=None):
        fig = self.plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 0.1], width_ratios=ratios, hspace=0.5)
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]
        strip = fig.add_subplot(gs[1, :])
        strip.axis("off")
        return fig, axes, strip

    def strip(self, ax, text, color="#263238"):
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, fontweight="bold", color=color,
                transform=ax.transAxes)

    def finish(self, fig, n, name, title, table: pd.DataFrame | None = None):
        fig.suptitle(title, fontsize=14, fontweight="bold")
        path = self.out / f"{n:02d}_{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        self.plt.close(fig)
        if table is not None:
            table.to_csv(self.out / f"{n:02d}_표.csv", index=False, encoding="utf-8-sig")
            print("\n" + table.to_string(index=False))
        print(f"\n그림: {show_path(path)}")


def acc_bars(ax, names, values, per_person=None, colors=None, chance=None, title=None):
    x = np.arange(len(names))
    ax.bar(x, np.array(values) * 100, color=colors or AFTER, alpha=0.9)
    for i, v in enumerate(values):
        ax.text(i, v * 100 + 2, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
        if per_person is not None and per_person[i]:
            pts = np.array(list(per_person[i].values())) * 100
            ax.scatter(i + np.random.default_rng(i).uniform(-0.18, 0.18, len(pts)), pts, s=12, color="black", zorder=3)
    if chance:
        ax.axhline(chance * 100, color="red", ls="--", lw=1)
        ax.text(len(names) - 0.5, chance * 100 + 1, f"찍기 {chance:.0%}", color="red", fontsize=8, ha="right")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("정확도 %")
    if title:
        ax.set_title(title, fontsize=11)


def conf_ax(ax, y, yp, labels, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp, labels=labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.6 else "black")
    ko = [NAME_KO.get(l, DEPTH_KO.get(l, l)) for l in labels]
    ax.set_xticks(range(len(labels)), ko, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), ko)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title(title, fontsize=11)


def trace_ax(ax, t, series: dict, title, shade=None, ylim=(0, ADC_MAX)):
    for name, (v, kw) in series.items():
        ax.plot(t, v, label=name, **kw)
    if shade is not None:
        ax.fill_between(t, ylim[0], ylim[1], where=shade, color="#bdbdbd", alpha=0.5, lw=0)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")


# ---------------------------------------------------------------------------
# 1부. 데이터를 늘렸더니 안 됐다
# ---------------------------------------------------------------------------
def s0(S: Story):
    """1차 3명 결과 (기존)"""
    from common.config import repo_path
    st = pd.read_csv(repo_path("data/processed/train_pressure/stages.csv"), encoding="utf-8-sig")
    md = pd.read_csv(repo_path("data/processed/train_pressure/models.csv"), encoding="utf-8-sig")
    fig, (a, b), strip = S.figure()
    acc_bars(a, [s.split(" ", 1)[0] for s in st["condition"]], st["acc_mean"].tolist(),
             title="1차 3명: 전처리 단계별 (RF)")
    acc_bars(b, md["condition"].str.replace("(threshold)", "", regex=False).tolist(), md["acc_mean"].tolist(),
             per_person=[{p: r[f"acc_{p}"] for p in ("p1", "p2", "p3")} for _, r in md.iterrows()],
             title="1차 3명: 모델 비교")
    S.strip(strip, f"1차 3명 (같은 팀원·같은 날): 원본 {st['acc_mean'].iloc[0]:.1%} → 최종 RF {md.loc[md.model == 'rf', 'acc_mean'].iloc[0]:.1%}")
    S.res["old_raw"] = {"mean": float(st["acc_mean"].iloc[0])}
    S.res["old_final"] = {"mean": float(md.loc[md.model == "rf", "acc_mean"].iloc[0])}
    S.res_path.write_text(json.dumps(S.res, ensure_ascii=False, indent=1), encoding="utf-8")
    S.finish(fig, 0, "1차_3명", "0. 1차 데이터 (3명)", md[["condition", "acc_mean", "macro_f1"]])


def _s1_eval(S):
    c = S.ctx
    return S.loso("11명 원본 (모든 시도, 7채널, 원본 값)", features(c.W_all, c.ch7, "raw"), c.y, c.test,
                  pd.Series(True, index=c.W_all.index))


def s1(S: Story):
    """11명 원본 그대로 학습"""
    r = _s1_eval(S)
    S.save("s1", r)
    old_raw = S.res.get("old_raw", {}).get("mean")
    old_final = S.res.get("old_final", {}).get("mean")
    fig, (a, b), strip = S.figure(ratios=[1, 1.4])
    names, vals = [], []
    if old_raw is not None:
        names += ["3명 원본", "3명 최종"]
        vals += [old_raw, old_final]
    names.append("11명 원본")
    vals.append(r["mean"])
    acc_bars(a, names, vals, colors=[BEFORE] * (len(vals) - 1) + ["#e53935"], chance=0.2, title="사람 수를 늘린 뒤")
    pp = sorted(r["acc"])
    b.bar(pp, [r["acc"][p] * 100 for p in pp], color="#e53935", alpha=0.85)
    b.axhline(20, color="red", ls="--", lw=1)
    b.set_ylim(0, 105)
    b.set_ylabel("정확도 %")
    b.set_title("11명 사람별 (한 사람 빼고 학습 → 그 사람 시험)", fontsize=11)
    S.strip(strip, f"3명 최종 {old_final:.1%} → 11명 원본 {r['mean']:.1%}" if old_final else f"11명 원본 {r['mean']:.1%}",
            "#c62828")
    S.finish(fig, 1, "11명_원본", "1. 11명으로 늘려 원본 그대로 학습",
             pd.DataFrame({"사람": pp, "정확도": [round(r["acc"][p], 3) for p in pp]}))


# ---------------------------------------------------------------------------
# 2부. 전처리
# ---------------------------------------------------------------------------
def _example(S, status):
    qc = S.ctx.qc
    cand = qc[qc["status"] == status]
    if status == "dropout":
        cand = cand.sort_values("dropouts", ascending=False)
    elif status == "ok":                         # 정상 예시: 어느 자세에서도 0이 아니고 끊김 없는 센서
        cand = cand[(cand["zero_blocks"] == 0) & (cand["dropouts"] == 0)].sort_values("baseline", ascending=False)
    if cand.empty:
        return None
    r = cand.iloc[0]
    return r["person"], r["channel"]


def s2(S: Story):
    """센서 상태 진단"""
    c, plt = S.ctx, S.plt
    qc = c.qc
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.6)
    df = c.df
    for i, st in enumerate(("ok", "dropout", "unloaded", "dead")):
        ax = fig.add_subplot(gs[i, 0])
        ex = _example(S, st)
        if ex is None:
            ax.axis("off")
            continue
        p, ch = ex
        g = df[(df["person"] == p) & ~df["rest"] & df["last_attempt"]]
        g = g.iloc[: 360 * 4]
        t = np.arange(len(g)) * 0.5 / 60
        ax.plot(t, g[ch], color=STATUS_COLOR[st], lw=0.8)
        ax.set_ylim(0, ADC_MAX)
        ax.set_ylabel(STATUS_KO[st], rotation=0, ha="right", fontsize=10, fontweight="bold")
        ax.text(0.01, 0.85, f"{p} · {CH_KO[ch]}", transform=ax.transAxes, fontsize=8)
        ax.tick_params(labelsize=7)
    ax.set_xlabel("분 (자세 4개 구간)")
    ax = fig.add_subplot(gs[:, 1])
    code = {"ok": 0, "unloaded": 1, "dropout": 2, "dead": 3}
    pp = sorted(qc["person"].unique())
    M = qc.pivot(index="person", columns="channel", values="status").loc[pp, c.ch7]
    from matplotlib.colors import ListedColormap
    ax.imshow(M.replace(code).to_numpy(dtype=float), cmap=ListedColormap([STATUS_COLOR[k] for k in code]),
              vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(c.ch7)), [CH_KO[x] for x in c.ch7], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pp)), pp)
    ax.set_title("사람 × 센서 상태", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=STATUS_COLOR[k], label=STATUS_KO[k]) for k in code], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
    counts = qc["status"].value_counts()
    fig.suptitle("2. 센서 상태 진단 (실제 파형 예시 → 11명 × 7센서)", fontsize=14, fontweight="bold")
    path = S.out / "02_센서진단.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    t = pd.DataFrame([{"상태": STATUS_KO[k], "칸 수": int(counts.get(k, 0))} for k in code])
    t.to_csv(S.out / "02_표.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False), f"\n그림: {show_path(path)}")


def s3(S: Story):
    """다시 찍은 구간 정리 + 고장 채널 제외"""
    c = S.ctx
    r0 = S.get("s1", lambda: _s1_eval(S))
    r1 = S.loso("마지막 시도만 학습", features(c.W_all, c.ch7, "raw"), c.y, c.test, c.is_last)
    S.save("s3a", r1)
    r2 = S.loso(f"고장 채널({','.join(CH_KO[x] for x in c.drop_ch)}) 제외 → {len(c.ch6)}채널",
                features(c.W_all, c.ch6, "raw"), c.y, c.test, c.is_last)
    S.save("s3", r2)
    qc = c.qc
    fig, (a, b), strip = S.figure()
    for ax, chans, title in ((a, c.ch7, f"전: {len(c.ch7)}채널"), (b, c.ch6, f"후: {len(c.ch6)}채널")):
        t = qc[qc["channel"].isin(chans)].groupby(["channel", "status"]).size().unstack(fill_value=0).reindex(chans)
        bottom = np.zeros(len(chans))
        for k in ("ok", "unloaded", "dropout", "dead"):
            v = t[k].fillna(0).to_numpy() if k in t else np.zeros(len(chans))
            ax.bar([CH_KO[x] for x in chans], v, bottom=bottom, color=STATUS_COLOR[k], label=STATUS_KO[k])
            bottom += v
        ax.set_ylabel("사람 수")
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
    a.legend(fontsize=8)
    S.strip(strip, f"원본 {r0['mean']:.1%} → 마지막 시도만 {r1['mean']:.1%} → 고장 채널 제외 {r2['mean']:.1%}")
    S.finish(fig, 3, "재측정_고장채널", "3. 다시 찍은 구간 정리 · 고장 채널 제외",
             pd.DataFrame({"단계": ["원본", "마지막 시도만", f"{len(c.ch6)}채널"],
                           "정확도": [round(x, 3) for x in (r0["mean"], r1["mean"], r2["mean"])]}))


def s4(S: Story):
    """접촉 끊김 복구"""
    c = S.ctx
    fixed, unreliable, rlog = c.repaired
    before = S.get("s3", lambda: S.loso("6채널", features(c.W_all, c.ch6, "raw"), c.y, c.test, c.is_last, show=False))
    r = S.loso("짧은 끊김 복구 + 긴 끊김 학습 제외", features(c.W_fix, c.ch6, "raw"), c.y, c.test,
               c.is_last & ~c.W_fix["unreliable"])
    S.save("s4", r)
    ex = rlog[rlog["kind"] == "repaired"].sort_values("length", ascending=False)
    pick = None
    for _, cand in ex.iterrows():
        around = c.df.loc[cand["start"] - 30:cand["start"] + cand["length"] + 30]
        around = around[(around["person"] == cand["person"]) & (around["seg"] == cand["seg"])]
        outside = around.drop(index=range(cand["start"], cand["start"] + cand["length"]), errors="ignore")
        if len(around) >= 40 and (outside[cand["channel"]] >= 300).all():
            pick = cand
            break
    fig, (a, b), strip = S.figure()
    if pick is not None:
        ch, s0_ = pick["channel"], pick["start"]
        win = c.df.loc[s0_ - 30:s0_ + pick["length"] + 30]
        win = win[win["seg"] == pick["seg"]]
        t = (win.index - s0_) * 0.5
        trace_ax(a, t, {"원본": (win[ch], {"color": "#e53935"})}, f"전: {pick['person']} · {CH_KO[ch]} · {pick['label']}")
        trace_ax(b, t, {"복구 후": (fixed.loc[win.index, ch], {"color": AFTER})}, "후: 앞뒤 값으로 채움 (3초 이하 끊김)")
        for ax in (a, b):
            ax.set_xlabel("초")
    n_rep, n_bad = int((rlog.kind == "repaired").sum()), int((rlog.kind == "unreliable").sum())
    S.strip(strip, f"복구 {n_rep}번 · 긴 끊김 {n_bad}번 학습 제외   |   정확도 {before['mean']:.1%} → {r['mean']:.1%}")
    S.finish(fig, 4, "끊김복구", "4. 접촉 끊김 복구",
             pd.DataFrame({"구분": ["복구(3초 이하)", "학습 제외(3초 초과)"], "횟수": [n_rep, n_bad]}))


def s5(S: Story):
    """움직임 구간 학습 제외 (쉬는시간 = 정답으로 감지기 검증)"""
    c = S.ctx
    fixed, moving = c.repaired[0], c.moving
    before = S.get("s4", lambda: S.loso("끊김 복구", features(c.W_fix, c.ch6, "raw"), c.y, c.test,
                                         c.is_last & ~c.W_fix["unreliable"], show=False))
    r = S.loso("움직임 구간 학습 제외", features(c.W_fix, c.ch6, "raw"), c.y, c.test, c.clean)
    S.save("s5", r)
    rest_rate = moving[fixed["rest"]].mean()
    hold_rate = moving[~fixed["rest"]].mean()
    p = fixed["person"].iloc[0]
    g = fixed[fixed["person"] == p].iloc[: 380 * 3]
    t = np.arange(len(g)) * 0.5 / 60
    ch = max(c.ch6, key=lambda x: g[x].std())
    fig, (a, b), strip = S.figure()
    trace_ax(a, t, {CH_KO[ch]: (g[ch], {"color": "#e53935", "lw": 0.8})}, f"전: {p} · 회색 = 쉬는시간(자세 바꾸는 중)",
             shade=g["rest"].to_numpy())
    kept = g[ch].where(~moving.loc[g.index])
    trace_ax(b, t, {"학습에 쓰는 값": (kept, {"color": AFTER, "lw": 0.8})}, "후: 움직임 감지 구간을 뺌",
             shade=g["rest"].to_numpy())
    for ax in (a, b):
        ax.set_xlabel("분")
    S.strip(strip, f"쉬는시간 중 감지 {rest_rate:.0%} · 자세 유지 중 감지 {hold_rate:.0%}   |   정확도 {before['mean']:.1%} → {r['mean']:.1%}")
    S.finish(fig, 5, "움직임", "5. 움직임 구간 학습 제외",
             pd.DataFrame({"구분": ["쉬는시간 중 감지", "자세 유지 중 감지"], "비율": [round(rest_rate, 3), round(hold_rate, 3)]}))


def s6(S: Story):
    """기준 대비 정규화"""
    c = S.ctx
    W = c.W_fix
    before = S.get("s5", lambda: S.loso("움직임 제외", features(W, c.ch6, "raw"), c.y, c.test, c.clean, show=False))
    res = {}
    for k in ("A", "B", "C", "D"):
        res[k] = S.loso(f"정규화 {NORMS[k]}", features(W, c.ch6, k, c.base), c.y, c.test, c.clean)
    best = max(res, key=lambda k: res[k]["mean"])
    S.save("s6", res[best])
    S.res["s6_norm"] = {"mean": res[best]["mean"], "best": best}
    nm = W[W["y"] == "normal"]
    pp = sorted(W["person"].unique())
    # 보여줄 센서: 11명 모두 정상인 센서 중 첫 번째 (고장 센서는 정규화와 상관없이 망가져 보이므로)
    ok = c.qc.groupby("channel")["status"].apply(lambda s: (s == "ok").all())
    show = next((x for x in c.ch6 if ok.get(x, False)), c.ch6[0])
    raw = features(nm, c.ch6, "raw")[f"v_{show}"]
    nrm = features(nm, c.ch6, best, c.base)
    col = f"n_{show}" if f"n_{show}" in nrm else nrm.columns[0]
    fig, (a, b, d), strip = S.figure(n_cols=3, width=18, ratios=[1, 1, 1.1])
    a.boxplot([raw[nm["person"] == p] for p in pp], showfliers=False)
    a.set_xticks(range(1, len(pp) + 1), pp, fontsize=8)
    a.set_title(f"전: 바른자세 원본 값 ({CH_KO[show]}) — 사람마다 제각각", fontsize=10)
    b.boxplot([nrm.loc[nm["person"] == p, col] for p in pp], showfliers=False)
    b.axhline(0, color="red", lw=0.8)
    b.set_xticks(range(1, len(pp) + 1), pp, fontsize=8)
    b.set_title(f"후: 기준 대비 ({NORMS[best]}, {CH_KO[show]})", fontsize=10)
    acc_bars(d, ["원본"] + list(res), [before["mean"]] + [res[k]["mean"] for k in res],
             colors=[BEFORE] + [AFTER if k != best else "#e53935" for k in res], title="정규화 방법별 정확도")
    S.strip(strip, f"정확도 {before['mean']:.1%} → {res[best]['mean']:.1%} ({NORMS[best]})")
    S.finish(fig, 6, "정규화", "6. 기준 대비 정규화",
             pd.DataFrame({"방법": ["원본"] + [NORMS[k] for k in res],
                           "정확도": [round(before["mean"], 3)] + [round(res[k]["mean"], 3) for k in res]}))


def s7(S: Story):
    """증강 + 모델 비교 (5개 자세) → 전처리로는 한계"""
    c = S.ctx
    best = S.res.get("s6_norm", {}).get("best", "A")
    F = features(c.W_fix, c.ch6, best, c.base)
    r_aug = S.loso("좌우 대칭 증강", F, c.y, c.test, c.clean, aug=True)
    S.save("s7_aug", r_aug)
    mods = {}
    for m in ("rule",) + MODELS:
        Fm = F if (m != "rule" or "LR_front" in F) else features(c.W_fix, c.ch6, "C", c.base)
        mods[m] = S.loso(f"모델 {MODEL_KO.get(m, '규칙')}", Fm, c.y, c.test, c.clean, model=m)
    bm = max(mods, key=lambda k: mods[k]["mean"])
    S.save("s7", mods[bm])
    chain = [("1 원본", "s1"), ("3 정리", "s3"), ("4 복구", "s4"), ("5 움직임", "s5"), ("6 정규화", "s6"), ("7 증강", "s7_aug")]
    chain = [(n, S.res[k]["mean"]) for n, k in chain if k in S.res]
    fig, (a, b), strip = S.figure(ratios=[1.2, 1])
    acc_bars(a, [n for n, _ in chain], [v for _, v in chain], chance=0.2, title="전처리 단계별 (RF)")
    ko = {**MODEL_KO, "rule": "규칙"}
    acc_bars(b, [ko[m] for m in mods], [mods[m]["mean"] for m in mods], per_person=[mods[m]["acc"] for m in mods],
             chance=0.2, title="모델 비교 (5개 자세)")
    b.tick_params(axis="x", rotation=20)
    S.strip(strip, f"전처리를 다 해도 5개 자세 최고 {mods[bm]['mean']:.1%} ({ko[bm]}) — 전처리로는 한계", "#c62828")
    S.finish(fig, 7, "전처리_한계", "7. 전처리 전체 → 모델 비교",
             pd.DataFrame({"모델": [ko[m] for m in mods], "정확도": [round(mods[m]["mean"], 3) for m in mods]}))


# ---------------------------------------------------------------------------
# 3부. 원인 찾기
# ---------------------------------------------------------------------------
def s8(S: Story):
    """자세를 바꾸면 센서가 실제로 변하나 (기준 대비 변화)"""
    D = dev_matrix(S.ctx)
    flat = {p: (D[p].abs() < 5).to_numpy().mean() for p in D}
    lo, hi = min(flat, key=flat.get), max(flat, key=flat.get)
    fig, (a, b, d), strip = S.figure(n_cols=3, width=19, height=6.5, ratios=[1, 1, 1])
    for ax, p in ((a, lo), (b, hi)):
        M = D[p]
        ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-80, vmax=80, aspect="auto")
        ax.set_xticks(range(len(M.columns)), [CH_KO[x] for x in M.columns], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(M.index)), M.index, fontsize=8)
        ax.set_title(f"{p}: 안 변한 칸 {flat[p]:.0%}", fontsize=11)
    pp = sorted(flat)
    d.bar(pp, [flat[p] * 100 for p in pp], color="#78909c")
    d.set_ylim(0, 100)
    d.set_ylabel("자세를 바꿔도 거의 안 변한 칸 %")
    d.set_title("사람별", fontsize=11)
    d.tick_params(axis="x", rotation=45)
    S.strip(strip, f"자세를 바꿔도 센서 값이 거의 안 변하는 칸이 평균 {np.mean(list(flat.values())):.0%} (포화·미접촉)", "#c62828")
    S.finish(fig, 8, "센서변화", "8. 자세를 바꾸면 센서가 변하나 (바른자세(중) 기준 대비, 파랑 = 빠짐 · 빨강 = 더 눌림)",
             pd.DataFrame({"사람": pp, "안변한칸": [round(flat[p], 3) for p in pp]}))


def s9(S: Story):
    """틀린 구간은 무엇으로 보이나"""
    c = S.ctx
    best = S.res.get("s6_norm", {}).get("best", "A")
    r = S.loso("5개 자세 (구간 분석용)", features(c.W_fix, c.ch6, best, c.base), c.y, c.test, c.clean, show=False)
    te = c.test
    d = pd.DataFrame({"seg": c.W_fix["seg"][te], "y": c.y[te], "p": r["pred"][te]})
    blk = d.groupby("seg").agg(y=("y", "first"), p=("p", lambda s: s.mode().iloc[0]))
    fig, (a, b), strip = S.figure(ratios=[1, 1])
    conf_ax(a, blk["y"], blk["p"], LABELS, f"3분 구간 다수결 (구간 {len(blk)}개)")
    wrong = blk[blk["y"] != blk["p"]]["p"].value_counts()
    b.barh([LABEL_KO[x] for x in wrong.index[::-1]], wrong.values[::-1], color="#e53935")
    b.set_xlabel("틀린 구간 수")
    b.set_title("틀린 구간은 무엇으로 판정됐나", fontsize=11)
    top2 = wrong.head(2)
    S.strip(strip, f"틀린 구간 {len(wrong) and int(wrong.sum())}개 중 {int(top2.sum())}개가 "
                   f"'{LABEL_KO[top2.index[0]]}' 또는 '{LABEL_KO[top2.index[1]]}'", "#c62828")
    S.finish(fig, 9, "틀린구간", "9. 틀린 구간은 무엇으로 보이나",
             pd.DataFrame({"판정": [LABEL_KO[x] for x in wrong.index], "틀린구간수": wrong.values}))


def s10(S: Story):
    """압력은 자세보다 앉은 위치를 더 반영한다"""
    c = S.ctx
    W = c.W_fix
    F = features(W, c.ch6, "raw")
    r_pos = S.loso("압력으로 앉은 위치(앞/중/뒤) 맞히기", F, W["depth"], c.test, c.clean)
    r_cls = S.get("s5", lambda: S.loso("압력으로 자세 맞히기", F, c.y, c.test, c.clean, show=False))
    D = dev_matrix(c)
    p = max(D, key=lambda q: D[q].loc[[l for l in D[q].index if l.startswith("바른자세")]].abs().to_numpy().mean())
    M = D[p].loc[[l for l in D[p].index if l.startswith("바른자세")]]
    fig, (a, b), strip = S.figure(ratios=[1.3, 1])
    x = np.arange(len(M.columns))
    for i, (lab, row) in enumerate(M.iterrows()):
        a.bar(x + (i - 1) * 0.27, row.to_numpy(), 0.27, label=lab)
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(x, [CH_KO[x_] for x_ in M.columns], rotation=20)
    a.set_ylabel("기준 대비 변화 (%p)")
    a.legend(fontsize=8)
    a.set_title(f"{p}: 같은 바른자세인데 앉은 위치만 바꿨을 때", fontsize=11)
    acc_bars(b, ["앉은 위치\n(3개)", "자세\n(5개)"], [r_pos["mean"], r_cls["mean"]],
             per_person=[r_pos["acc"], r_cls["acc"]], colors=["#43a047", BEFORE], title="압력만으로 맞히기")
    S.strip(strip, f"압력으로 앉은 위치 {r_pos['mean']:.1%} (찍기 33%) vs 자세 {r_cls['mean']:.1%} (찍기 20%)")
    S.finish(fig, 10, "앉은위치", "10. 압력은 자세보다 앉은 위치를 더 반영한다",
             pd.DataFrame({"맞힐 것": ["앉은 위치", "자세"], "정확도": [round(r_pos["mean"], 3), round(r_cls["mean"], 3)]}))


# ---------------------------------------------------------------------------
# 4부. 단계 판정
# ---------------------------------------------------------------------------
def staged_test(c):
    return c.test & ~c.W_fix["seg"].isin(base_segments(c))


def s11(S: Story):
    """거리센서로 앉은 위치 알기"""
    c = S.ctx
    W = c.W_fix
    m = staged_test(c)
    head = window_head(c)
    rel = head - W["person"].map(head_baseline(c))
    est = estimate_depth(c)
    from sklearn.tree import DecisionTreeClassifier
    accs = {}
    for p in sorted(W["person"].unique()):          # 절대 거리: 다른 사람으로 기준선을 배워서 적용
        tr = m & (W["person"] != p) & head.notna()
        te = m & (W["person"] == p) & head.notna()
        dt = DecisionTreeClassifier(max_depth=2, random_state=0).fit(head[tr].to_frame(), W.loc[tr, "depth"])
        accs[p] = float((dt.predict(head[te].to_frame()) == W.loc[te, "depth"]).mean())
    abs_acc = float(np.mean(list(accs.values())))
    rel_acc = (est[m] == W.loc[m, "depth"]).groupby(W.loc[m, "person"]).mean()
    S.res["s11"] = {"mean": float(rel_acc.mean()), "acc": rel_acc.to_dict()}
    S.res_path.write_text(json.dumps(S.res, ensure_ascii=False, indent=1), encoding="utf-8")
    fig, (a, b), strip = S.figure()
    cols = {"front": "#e53935", "middle": "#43a047", "back": "#1e88e5"}
    for d_, col in cols.items():
        a.hist(head[m & (W["depth"] == d_)].dropna(), bins=np.arange(10, 121, 2), alpha=0.55, color=col, label=DEPTH_KO[d_])
        b.hist(rel[m & (W["depth"] == d_)].dropna(), bins=np.arange(-60, 61, 2), alpha=0.55, color=col, label=DEPTH_KO[d_])
    for v in (-DEPTH_STEP_CM, DEPTH_STEP_CM):
        b.axvline(v, color="black", ls="--")
    a.set_title("전: 머리 거리 그대로 (모니터와의 거리에 따라 달라짐)", fontsize=11)
    b.set_title(f"후: 머리 거리 − 바른자세 기준 (점선 ±{DEPTH_STEP_CM:g}cm)", fontsize=11)
    a.set_xlabel("cm")
    b.set_xlabel("cm")
    a.legend()
    b.legend()
    S.strip(strip, f"앉은 위치: 거리 그대로(다른 사람으로 기준선 학습) {abs_acc:.1%} · 바른자세 기준 대비 ±10cm {rel_acc.mean():.1%} (찍기 33%)")
    S.finish(fig, 11, "거리센서_위치", "11. 거리센서로 앉은 위치(앞/중/뒤) 알기",
             pd.DataFrame({"방법": ["머리 거리 그대로", "기준 대비 ±10cm"], "정확도": [round(abs_acc, 3), round(rel_acc.mean(), 3)]}))


def s12(S: Story):
    """① 바른자세 vs 나쁜 자세: 기준 1개 → 위치별 기준"""
    c = S.ctx
    y = c.y
    t = y.where(y == "normal", "bad")
    m = staged_test(c)
    r1 = S.loso("① 기준 1개", features(c.W_fix, c.ch6, "B", c.base), t, m, c.clean)
    r2 = S.loso("① 위치별 기준 (거리센서 추정)", features(c.W_fix, c.ch6, "B", depth_baselines(c, estimate_depth(c))),
                t, m, c.clean)
    S.save("s12", r2)
    fig, (a, b), strip = S.figure()
    conf_ax(a, t[m], r1["pred"][m], ["normal", "bad"], f"전: 바른자세(중) 기준 1개  {r1['mean']:.1%}")
    conf_ax(b, t[m], r2["pred"][m], ["normal", "bad"], f"후: 앉은 위치별 기준  {r2['mean']:.1%}")
    S.strip(strip, f"① 바른자세 vs 나쁜 자세  {r1['mean']:.1%} → {r2['mean']:.1%} (찍기 50%)")
    S.finish(fig, 12, "바른vs나쁜", "12. ① 바른자세 vs 나쁜 자세",
             pd.DataFrame({"기준": ["1개", "위치별"], "정확도": [round(r1["mean"], 3), round(r2["mean"], 3)],
                           "균형정확도": [round(r1["bal"], 3), round(r2["bal"], 3)]}))


def _staged_F(S):
    c = S.ctx
    return S.ctx._get("staged_F", lambda: features(c.W_fix, c.ch6, "B", depth_baselines(c, estimate_depth(c))))


def s13(S: Story):
    """② 편향 vs 다리 꼬기"""
    c = S.ctx
    y = c.y
    mask = y != "normal"
    t = y.map(KIND)
    F = _staged_F(S)
    res = {m: S.loso("② 편향 vs 다리 꼬기", F, t, staged_test(c) & mask, c.clean & mask, model=m) for m in MODELS}
    bm = max(res, key=lambda k: res[k]["mean"])
    S.save("s13", res[bm])
    fig, (a, b), strip = S.figure()
    te = staged_test(c) & mask
    conf_ax(a, t[te], res[bm]["pred"][te], ["lean", "cross"], f"{MODEL_KO[bm]}  {res[bm]['mean']:.1%}")
    acc_bars(b, [MODEL_KO[m] for m in res], [res[m]["mean"] for m in res], per_person=[res[m]["acc"] for m in res],
             chance=0.5, title="모델 5종")
    b.tick_params(axis="x", rotation=20)
    S.strip(strip, f"② 편향 vs 다리 꼬기 최고 {res[bm]['mean']:.1%} (찍기 50%) → 앱에서는 '추정'으로만 표시", "#c62828")
    S.finish(fig, 13, "편향vs다리꼬기", "13. ② 편향 vs 다리 꼬기",
             pd.DataFrame({"모델": [MODEL_KO[m] for m in res], "정확도": [round(res[m]["mean"], 3) for m in res]}))


def s14(S: Story):
    """③ 체중이 실린 쪽 — 실수와 수정"""
    c = S.ctx
    y = c.y
    mask = y != "normal"
    F = _staged_F(S)
    te = staged_test(c) & mask
    wrong = S.loso("③ 처음: 왼다리꼬기를 '왼쪽'으로 묶음", F, y.map(SIDE_WRONG), te, c.clean & mask)
    right = S.loso("③ 수정: 체중이 실린 쪽 (왼다리꼬기 → 오른쪽)", F, y.map(SIDE), te, c.clean & mask)
    S.save("s14", right)
    D = dev_matrix(c)
    good = [p for p in D if p not in c.flags.loc[c.flags.flagged, "person"].tolist()]
    fig, (a, b, d), strip = S.figure(n_cols=3, width=19, ratios=[1, 1, 1.1])
    conf_ax(a, y.map(SIDE_WRONG)[te], wrong["pred"][te], ["w_left", "w_right"], f"전: 꼰 다리 쪽으로 묶음  {wrong['mean']:.1%}")
    conf_ax(b, y.map(SIDE)[te], right["pred"][te], ["left", "right"], f"후: 체중 실린 쪽으로 묶음  {right['mean']:.1%}")
    labels = ["왼쪽편향", "왼다리꼬기", "오른쪽편향", "오른다리꼬기"]
    lr = []
    for l in labels:                              # 앉은 위치(앞/중/뒤) 전부 평균
        v = [D[p].loc[r, ["L_front", "L_mid", "L_back"]].mean() - D[p].loc[r, ["R_front", "R_mid", "R_back"]].mean()
             for p in good for r in D[p].index if r.startswith(l + "(")]
        lr.append(np.mean(v))
    d.bar(labels, lr, color=["#1e88e5", "#e53935", "#1e88e5", "#e53935"])
    d.axhline(0, color="black", lw=0.6)
    d.set_ylabel("왼쪽 − 오른쪽 압력 변화 (%p)  + = 왼쪽에 체중")
    d.set_title(f"자세별 좌우 압력 차 (문제 세션 제외 {len(good)}명, 모든 위치 평균)", fontsize=10)
    d.tick_params(axis="x", rotation=20)
    S.strip(strip, f"③ 체중이 실린 쪽  {wrong['mean']:.1%} → {right['mean']:.1%}  (다리 꼬기 방향을 압력 기준으로 바로잡음)")
    S.finish(fig, 14, "쏠림방향_실수와수정", "14. ③ 어느 쪽에 체중이 실렸나 — 실수와 수정",
             pd.DataFrame({"묶는 방식": ["꼰 다리 쪽", "체중 실린 쪽"], "정확도": [round(wrong["mean"], 3), round(right["mean"], 3)]}))


def s15(S: Story):
    """단계마다 모델 비교 → 모델 선정"""
    c = S.ctx
    y = c.y
    F = _staged_F(S)
    tasks = {"① 바른 vs 나쁜": (pd.Series(True, index=y.index), y.where(y == "normal", "bad")),
             "③ 체중 실린 쪽": (y != "normal", y.map(SIDE))}
    rows, res = [], {}
    for tn, (mask, t) in tasks.items():
        for m in MODELS:
            res[(tn, m)] = S.loso(f"{tn}", F, t, staged_test(c) & mask, c.clean & mask, model=m)
            rows.append({"단계": tn, "모델": MODEL_KO[m], "정확도": round(res[(tn, m)]["mean"], 3)})
    tab = pd.DataFrame(rows)
    fig, axes, strip = S.figure()
    for ax, tn in zip(axes, tasks):
        acc_bars(ax, [MODEL_KO[m] for m in MODELS], [res[(tn, m)]["mean"] for m in MODELS],
                 per_person=[res[(tn, m)]["acc"] for m in MODELS], chance=0.5, title=tn)
        ax.tick_params(axis="x", rotation=20)
    wins = pd.Series([max(MODELS, key=lambda m: res[(tn, m)]["mean"]) for tn in tasks]).value_counts()
    S.strip(strip, f"선정: {MODEL_KO[wins.index[0]]} (두 단계 중 {wins.iloc[0]}개에서 최고)")
    S.finish(fig, 15, "모델선정", "15. 단계마다 모델 5종 비교", tab)


def s16(S: Story):
    """문제 세션을 학습에서 빼야 하나"""
    c = S.ctx
    y = c.y
    F = _staged_F(S)
    bad = c.flags.loc[c.flags.flagged, "person"].tolist()
    t1 = y.where(y == "normal", "bad")
    a1 = S.loso("① 전원 학습", F, t1, staged_test(c), c.clean, show=False)
    b1 = S.loso("① 문제 세션 제외 학습", F, t1, staged_test(c), c.clean, exclude=bad)
    mask = y != "normal"
    a3 = S.loso("③ 전원 학습", F, y.map(SIDE), staged_test(c) & mask, c.clean & mask, show=False)
    b3 = S.loso("③ 문제 세션 제외 학습", F, y.map(SIDE), staged_test(c) & mask, c.clean & mask, exclude=bad)
    pp = sorted(a1["acc"])
    fig, axes, strip = S.figure()
    for ax, (A, B, tn) in zip(axes, ((a1, b1, "① 바른 vs 나쁜"), (a3, b3, "③ 체중 실린 쪽"))):
        x = np.arange(len(pp))
        ax.bar(x - 0.2, [A["acc"][p] * 100 for p in pp], 0.4, color=BEFORE, label=f"전원 학습 {A['mean']:.1%}")
        ax.bar(x + 0.2, [B["acc"][p] * 100 for p in pp], 0.4, color=AFTER, label=f"문제 세션 제외 {B['mean']:.1%}")
        ax.set_xticks(x, [p + (" ⚠" if p in bad else "") for p in pp], fontsize=8)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.set_title(tn, fontsize=11)
    S.strip(strip, f"문제 세션({', '.join(bad)}) 제외: ① {a1['mean']:.1%} → {b1['mean']:.1%}, ③ {a3['mean']:.1%} → {b3['mean']:.1%}")
    S.finish(fig, 16, "문제세션", "16. 문제 세션을 학습에서 빼야 하나 (시험은 11명 전원)",
             pd.DataFrame({"단계": ["①", "③"], "전원": [round(a1["mean"], 3), round(a3["mean"], 3)],
                           "제외": [round(b1["mean"], 3), round(b3["mean"], 3)]}))


def s17(S: Story):
    """앱 최종 출력 (① → ③ 연결)"""
    c = S.ctx
    y = c.y
    F = _staged_F(S)
    te = staged_test(c)
    person = c.person
    r1 = S.loso("① 바른 vs 나쁜", F, y.where(y == "normal", "bad"), te, c.clean)
    out = pd.Series(index=F.index, dtype=object)
    mask = y != "normal"
    print("\n[연결] ①이 '나쁜 자세'라고 한 묶음에 ③ 적용")
    for p in sorted(person.unique()):
        tr = mask & c.clean & (person != p)
        mdl = make_model("rf", S.n_trees).fit(F[tr], y.map(SIDE)[tr])
        k = te & (person == p)
        out[k] = np.where(r1["pred"][k] == "normal", "none", mdl.predict(F[k]))
    truth = y.map(SIDE)
    acc = (out[te] == truth[te]).groupby(person[te]).mean()
    print(acc.round(3).to_string())
    S.res["s17"] = {"mean": float(acc.mean()), "acc": acc.to_dict()}
    S.res_path.write_text(json.dumps(S.res, ensure_ascii=False, indent=1), encoding="utf-8")
    five = S.res.get("s7", {}).get("mean")
    fig, (a, b), strip = S.figure()
    conf_ax(a, truth[te], out[te], ["none", "left", "right"], f"앱 출력 (바른자세 / 왼쪽에 체중 / 오른쪽에 체중)  {acc.mean():.1%}")
    b.bar(acc.index, acc.values * 100, color=AFTER)
    b.axhline(100 / 3, color="red", ls="--", lw=1)
    b.set_ylim(0, 105)
    b.set_title("사람별", fontsize=11)
    S.strip(strip, (f"5개 자세 한 번에 {five:.1%} → 단계 판정 앱 출력 {acc.mean():.1%} (찍기 33%)" if five
                    else f"단계 판정 앱 출력 {acc.mean():.1%} (찍기 33%)"))
    S.finish(fig, 17, "앱최종", "17. 앱 최종 출력 (① 바른/나쁜 → ③ 체중 실린 쪽)",
             pd.DataFrame({"사람": acc.index, "정확도": acc.round(3).values}))


STEPS = {i: f for i, f in enumerate([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17])}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[story] %(message)s")
    ap = argparse.ArgumentParser(description="발표 스토리 실험 (단계마다 그래프 한 장)")
    ap.add_argument("--step", nargs="+", type=int, help="단계 번호 (예: --step 1, --step 2 3 4)")
    ap.add_argument("--all", action="store_true", help="0~17 전부")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--folder", default="data/dataset")
    ap.add_argument("--out", default="data/processed/story")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.list or not (args.step or args.all):
        for i, f in STEPS.items():
            print(f"{i:>2}  {f.__doc__.strip().splitlines()[0]}")
        return 0
    S = Story(args)
    for i in (list(STEPS) if args.all else args.step):
        t0 = time.time()
        print(f"\n===== {i}. {STEPS[i].__doc__.strip().splitlines()[0]} =====")
        STEPS[i](S)
        log.info("%d단계 끝 (%.0f초)", i, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
