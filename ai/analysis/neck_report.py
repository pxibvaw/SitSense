"""거북목·좌우 기울기 기준값 보고서 — collect_neck으로 모은 CSV로 카메라 기준값(threshold)을 데이터로 정하고 정확도를 잰다

거북목은 학습 모델이 아니라 '각자의 바른자세 기준 대비 변화량' 규칙이다 (ai/engine/upper_body.judge).
그래서 할 일은 기준값을 고르는 것과, 고른 기준이 처음 보는 사람에게도 맞는지 확인하는 것.

    단서 ① 목 높이 감소 = (기준 목높이 − 지금) / 기준          config upper_rules.neck_drop_ratio (지금 0.12)
    단서 ② 얼굴 커짐   = 지금 얼굴비율 / 기준 − 1               config upper_rules.face_grow_ratio (지금 0.08)
    판정: 둘 중 하나라도 넘으면 거북목. 거리센서 2개는 확신도만 올림 (A 머리만 가까움 / B 가슴만 뒤로)

    좌우 기울기 (같은 수집의 왼쪽기울기·오른쪽기울기 자세)
    어깨 선 각도 변화 ≥ shoulder_angle_deg(5°) · 눈 선 각도 변화 ≥ head_angle_deg(7°) · 코 옆이동 ≥ head_offset(10%)
    중 하나라도 넘으면 기울기, 방향은 넘은 단서들의 부호 합 (+ 사람 왼쪽)

정답
    기울기 = 왼쪽기울기(left), 오른쪽기울기(right) / 나머지 자세 = none
    거북목 = 머리내밀기, 어깨말림 / 거북목 아님 = 바른자세, 바른자세2, 상체숙임, 등받이기댐
    data/dataset/neck/cva.csv (subject,label,cva)가 있으면 CVA < 48°를 거북목으로 (옆모습 사진으로 잰 임상 기준)

평가 (압력과 같은 방식)
    - 한 사람 빼고 나머지로 기준값을 고른 뒤(균형 정확도 최대) 뺀 사람으로 시험, 사람마다 반복
    - 자세마다 처음 3초는 자세를 바꾸는 중이라 뺌, 특징은 엔진처럼 최근 2초 중앙값

실행 (저장소 최상위, .venv 켠 상태):
    python -m ai.analysis.neck_report                    # data/dataset/neck/*.csv (test_로 시작하는 파일 제외)
    python -m ai.analysis.neck_report --folder 다른폴더

결과: data/processed/neck/
    01_단서분포.png        자세별 ①②·머리·가슴 변화 (가로선 = 지금 기준값)
    02_ROC.png             단서별 ROC + 데이터로 고른 기준값
    03_자세별판정.png/.csv  지금 기준 / 데이터로 고른 기준으로 자세마다 '거북목'으로 판정한 비율
    04_거리보조.png/.csv    거북목 확신도: 카메라만 vs 거리 보조 포함, 보조 A/B가 켜진 비율
    05_기울기단서.png       자세별 어깨선·눈선 각도, 코 옆이동 변화 (+ = 사람 왼쪽)
    06_기울기판정.png/.csv  지금 기준 / 데이터 기준으로 자세마다 left·right로 판정한 비율
    loso.csv, loso_tilt.csv 한 사람씩 빼고 시험한 결과 (사람별)
    summary.md             숫자 정리 + config에 넣을 값
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("sitsense.neck_report")

POSITIVE = ("머리내밀기", "어깨말림")
ORDER = ["바른자세", "머리내밀기", "어깨말림", "상체숙임", "등받이기댐", "왼쪽기울기", "오른쪽기울기", "바른자세2"]
TILT_TRUTH = {"왼쪽기울기": "left", "오른쪽기울기": "right"}
SKIP_SEC = 3.0          # 자세마다 처음 3초 제외 (바꾸는 중)
BASE_FROM, BASE_TO = 2.0, 12.0     # 첫 바른자세의 2~12초 = 기준 10초 (엔진의 기준 측정과 같은 길이)
SMOOTH = "2s"           # 엔진의 smooth_sec
CVA_CUT = 48.0


def load(folder: Path, cfg: dict) -> pd.DataFrame:
    from ai.features.pose_features import upper_features
    from common.schema import LANDMARK_NAMES
    from common.sensors_base import Landmark, PoseReading
    files = sorted(p for p in folder.glob("*.csv") if not p.name.startswith("test_") and p.name != "cva.csv")
    if not files:
        raise SystemExit(f"{folder}에 수집 CSV가 없음 (python -m ai.camera.collect_neck으로 수집)")
    min_vis = cfg["upper_rules"]["min_visibility"]
    parts = []
    for f in files:
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["file"] = f.name
        d["ts"] = pd.to_datetime(d["ts"])
        feats = []
        for _, r in d.iterrows():
            pts = {}
            for n in LANDMARK_NAMES:
                v = [r.get(f"{n}_{k}") for k in ("x", "y", "z", "vis")]
                if all(pd.notna(x) for x in v):
                    pts[n] = Landmark(*map(float, v))
            pose = PoseReading(ts=0.0, detected=bool(r["pose_detected"]) and bool(pts), points=pts)
            feats.append(upper_features(pose, int(r["img_w"]), int(r["img_h"]), min_vis))
        for k in ("neck_height", "face_ratio", "shoulder_angle", "head_angle", "head_offset", "shoulder_width_px"):
            d[k] = [np.nan if x is None else getattr(x, k) for x in feats]
        parts.append(d)
        log.info("  %s: %d줄, 카메라 판단 가능 %.0f%%", f.name, len(d), 100 * d["neck_height"].notna().mean())
    df = pd.concat(parts, ignore_index=True)
    df["person"] = df["subject"].astype(str) + np.where(df["clothes"] == "loose", "_loose", "")
    return df


def prepare(df: pd.DataFrame, rules) -> tuple[pd.DataFrame, dict]:
    """세션마다 기준 → 2초 중앙값 → 단서·엔진 판정"""
    from ai.engine.upper_body import Baseline, judge
    from ai.features.pose_features import UpperFeatures
    feat_cols = ["neck_height", "face_ratio", "shoulder_angle", "head_angle", "head_offset", "shoulder_width_px"]
    out, bases = [], {}
    for file, g in df.groupby("file", sort=False):
        g = g[g["phase"] == "record"].copy()
        b = g[(g["label"] == "바른자세") & g["t_seg"].between(BASE_FROM, BASE_TO)]
        if b["neck_height"].notna().sum() < 5:
            log.warning("  %s: 기준(첫 바른자세)에 카메라 판단 가능 프레임이 부족 → 건너뜀", file)
            continue
        base = Baseline()
        base.value = UpperFeatures(**{k: float(b[k].median()) for k in feat_cols})
        hb, cb = b["head_cm"].median(), b["chest_cm"].where(b["chest_cm"] * 10 >= rules.chest_min_mm).median()
        base.distance_mm = None if pd.isna(hb) else float(hb) * 10
        base.chest_mm = None if pd.isna(cb) else float(cb) * 10
        bases[file] = base
        # 2초 중앙값 (자세 구간 안에서만)
        sm = []
        for _, s in g.groupby("seg", sort=False):
            s = s.set_index("ts").sort_index()
            r = s[feat_cols + ["head_cm"]].rolling(SMOOTH).median()
            r["chest_cm"] = s["chest_cm"].where(s["chest_cm"] * 10 >= rules.chest_min_mm).rolling(SMOOTH).median()
            r[["file", "person", "label", "seg", "t_seg", "clothes"]] = s[["file", "person", "label", "seg", "t_seg",
                                                                          "clothes"]]
            sm.append(r.reset_index())
        g = pd.concat(sm, ignore_index=True)
        g = g[(g["t_seg"] >= SKIP_SEC) & g["neck_height"].notna()]
        bv = base.value
        g["neck_drop"] = (bv.neck_height - g["neck_height"]) / bv.neck_height
        g["face_grow"] = g["face_ratio"] / bv.face_ratio - 1
        g["closer_cm"] = (base.distance_mm / 10 - g["head_cm"]) if base.distance_mm else np.nan
        g["chest_back_cm"] = (g["chest_cm"] - base.chest_mm / 10) if base.chest_mm else np.nan
        heads, confs, cam_confs, assists, tilts = [], [], [], [], []
        for _, r in g.iterrows():
            f = UpperFeatures(**{k: float(r[k]) for k in feat_cols})
            dm = None if pd.isna(r["head_cm"]) else float(r["head_cm"]) * 10
            cm = None if pd.isna(r["chest_cm"]) else float(r["chest_cm"]) * 10
            j = judge(f, base, rules, distance_mm=dm, chest_mm=cm)
            j0 = judge(f, base, rules)
            heads.append(j.head.value)
            tilts.append(j.tilt.value)
            confs.append(j.head_confidence)
            cam_confs.append(j0.head_confidence)
            assists.append(next((k for k in ("head_only", "chest_back", "closer") if j.cues.get(k)), ""))
        g["engine"], g["conf"], g["conf_cam"], g["assist"] = heads, confs, cam_confs, assists
        g["tilt_engine"] = tilts
        g["d_shoulder"] = g["shoulder_angle"] - bv.shoulder_angle      # 도, + = 사람 왼쪽
        g["d_eye"] = g["head_angle"] - bv.head_angle
        g["d_offset"] = g["head_offset"] - bv.head_offset              # 어깨 너비 비율
        out.append(g)
    if not out:
        raise SystemExit("기준을 만들 수 있는 세션이 없음")
    return pd.concat(out, ignore_index=True), bases


def truth(d: pd.DataFrame, folder: Path) -> pd.Series:
    y = d["label"].isin(POSITIVE)
    cva = folder / "cva.csv"
    if cva.exists():
        c = pd.read_csv(cva, encoding="utf-8-sig")
        m = d[["person", "label"]].merge(c.rename(columns={"subject": "person"}), on=["person", "label"], how="left")
        has = m["cva"].notna().to_numpy()
        y = y.where(~has, m["cva"].to_numpy() < CVA_CUT)
        log.info("cva.csv: %d개 자세에 옆모습 각도 정답 적용 (CVA < %.0f° = 거북목)", int(c.shape[0]), CVA_CUT)
    return y.astype(bool)


def bal_acc(y, p) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tpr = (p & y).sum() / max(y.sum(), 1)
    tnr = (~p & ~y).sum() / max((~y).sum(), 1)
    return float((tpr + tnr) / 2)


GRID1 = np.round(np.arange(0.02, 0.41, 0.01), 3)      # 목 높이 감소 2~40%
GRID2 = np.round(np.arange(0.02, 0.41, 0.01), 3)      # 얼굴 커짐 2~40%


def best_pair(d: pd.DataFrame, y: pd.Series, now: tuple, tol: float = 0.005) -> tuple[float, float, float]:
    """'①≥t1 또는 ②≥t2' 규칙에서 균형 정확도가 가장 높은 (t1, t2).
    최고점과 tol(0.5%p) 안으로 비슷한 후보가 여럿이면 지금 config 값에 가장 가까운 것
    (= 데이터가 구분하지 못하면 팀 테스트 값을 유지)"""
    nd, fg = d["neck_drop"].to_numpy(), d["face_grow"].to_numpy()
    yy = y.to_numpy()
    a = nd[:, None, None] >= GRID1[None, :, None]
    b = fg[:, None, None] >= GRID2[None, None, :]
    p = a | b                                           # 줄 × t1 × t2
    tpr = (p & yy[:, None, None]).sum(0) / max(yy.sum(), 1)
    tnr = (~p & ~yy[:, None, None]).sum(0) / max((~yy).sum(), 1)
    ba = (tpr + tnr) / 2
    cand = np.argwhere(ba >= ba.max() - tol)
    i, j = min(cand, key=lambda ij: abs(GRID1[ij[0]] - now[0]) / now[0] + abs(GRID2[ij[1]] - now[1]) / now[1])
    return float(GRID1[i]), float(GRID2[j]), float(ba[i, j])


def roc(score: np.ndarray, y: np.ndarray):
    from sklearn.metrics import roc_auc_score, roc_curve
    m = np.isfinite(score)
    fpr, tpr, th = roc_curve(y[m], score[m])
    k = int(np.argmax(tpr - fpr))
    return fpr, tpr, float(th[k]), float(roc_auc_score(y[m], score[m])), (fpr[k], tpr[k])


TG_S = np.arange(1.0, 15.5, 1.0)          # 어깨 선 1~15°
TG_E = np.arange(1.0, 15.5, 1.0)          # 눈 선 1~15°
TG_O = np.round(np.arange(0.02, 0.31, 0.01), 2)   # 코 옆이동 2~30%


def tilt_predict(d: pd.DataFrame, ts: float, te: float, to: float) -> np.ndarray:
    """엔진 규칙과 같게: 넘은 단서들의 (변화/기준) 합의 부호로 방향"""
    sc = np.stack([(d["d_shoulder"] / ts).to_numpy(), (d["d_eye"] / te).to_numpy(), (d["d_offset"] / to).to_numpy()], 1)
    over = np.abs(sc) >= 1
    s = np.where(over, sc, 0).sum(1)
    return np.where(~over.any(1), "none", np.where(s > 0, "left", "right"))


def tilt_score(y: np.ndarray, p: np.ndarray) -> float:
    """3개 클래스(none/left/right) 균형 정확도"""
    return float(np.mean([(p[y == c] == c).mean() for c in ("none", "left", "right") if (y == c).any()]))


def best_tilt(d: pd.DataFrame, now: tuple, tol: float = 0.005) -> tuple[float, float, float, float]:
    """균형 정확도가 가장 높은 (어깨, 눈, 코) 기준. 비슷하면 지금 config 값에 가장 가까운 것"""
    y = d["tilt_y"].to_numpy()
    rows = []
    for a in TG_S:
        for b in TG_E:
            for c in TG_O:
                rows.append((tilt_score(y, tilt_predict(d, a, b, c)), a, b, c))
    top = max(r[0] for r in rows)
    cand = [r for r in rows if r[0] >= top - tol]
    best = min(cand, key=lambda r: sum(abs(v - n) / n for v, n in zip(r[1:], now)))
    return float(best[1]), float(best[2]), float(best[3]), float(best[0])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[neck] %(message)s")
    ap = argparse.ArgumentParser(description="거북목 기준값 보고서")
    ap.add_argument("--folder", default="data/dataset/neck")
    ap.add_argument("--out", default="data/processed/neck")
    args = ap.parse_args()
    from ai.analysis.preprocess_report import _plt
    from ai.engine.upper_body import UpperRules
    from common.config import load_config, repo_path
    cfg = load_config()
    rules = UpperRules.from_config(cfg)
    folder, out = repo_path(args.folder), repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()

    df = load(folder, cfg)
    d, _ = prepare(df, rules)
    y = truth(d, folder)
    d["y"] = y
    people = sorted(d["person"].unique())
    order = [l for l in ORDER if l in set(d["label"])]
    t_now = (rules.neck_drop_ratio, rules.face_grow_ratio)
    log.info("사람 %d명 (%s), 판단 가능 프레임 %d개", len(people), ", ".join(people), len(d))

    # 1) 단서 분포
    cues = [("neck_drop", "① 목 높이 감소 (%)", 100, t_now[0] * 100), ("face_grow", "② 얼굴 커짐 (%)", 100, t_now[1] * 100),
            ("closer_cm", "머리가 가까워진 거리 (cm)", 1, rules.assist_head_mm / 10),
            ("chest_back_cm", "가슴이 멀어진 거리 (cm)", 1, rules.assist_chest_mm / 10)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (c, title, k, line) in zip(axes.flat, cues):
        data = [d.loc[d["label"] == l, c].dropna().to_numpy() * k for l in order]
        if all(len(x) == 0 for x in data):
            ax.set_title(title + " (데이터 없음)")
            ax.axis("off")
            continue
        bp = ax.boxplot([x if len(x) else [np.nan] for x in data], showfliers=False, patch_artist=True)
        ax.set_xticks(range(1, len(order) + 1), order)
        for patch, l in zip(bp["boxes"], order):
            patch.set_facecolor("#e45756" if l in POSITIVE else "#9fb8d6")
        ax.axhline(line, color="k", ls="--", lw=1, label=f"지금 기준 {line:g}")
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.suptitle(f"자세별 기준 대비 변화 (빨강 = 거북목 자세, {len(people)}명)")
    fig.tight_layout()
    fig.savefig(out / "01_단서분포.png", dpi=130)
    plt.close(fig)

    # 2) ROC (전체 데이터) + 데이터로 고른 기준
    yy = d["y"].to_numpy()
    t1, t2, ba_all = best_pair(d, d["y"], t_now)
    combo = np.maximum(d["neck_drop"] / t1, d["face_grow"] / t2).to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    roc_rows = []
    for name, score, col in (("① 목 높이 감소", d["neck_drop"].to_numpy(), "#4c78a8"),
                             ("② 얼굴 커짐", d["face_grow"].to_numpy(), "#f58518"),
                             ("① 또는 ② (고른 기준)", combo, "#e45756")):
        fpr, tpr, th, auc, pt = roc(score, yy)
        ax.plot(fpr, tpr, color=col, label=f"{name}  AUC {auc:.2f}")
        ax.plot(*pt, "o", color=col)
        roc_rows.append((name, auc, th))
    p_now = ((d["neck_drop"] >= t_now[0]) | (d["face_grow"] >= t_now[1])).to_numpy()
    ax.plot(p_now[~yy].mean(), p_now[yy].mean(), "k*", ms=12, label=f"지금 기준 ({t_now[0]:.0%}, {t_now[1]:.0%})")
    ax.plot([0, 1], [0, 1], color="#ccc", ls=":")
    ax.set_xlabel("거북목이 아닌데 거북목으로 판정 (오탐률)")
    ax.set_ylabel("거북목을 거북목으로 판정 (민감도)")
    ax.set_title("단서별 ROC (점 = 가장 잘 나누는 기준)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "02_ROC.png", dpi=130)
    plt.close(fig)

    # 3) 한 사람씩 빼고: 나머지로 기준 고르기 → 뺀 사람 시험
    lrows = []
    d["pred_loso"] = False
    for p in people:
        tr, te = d["person"] != p, d["person"] == p
        if tr.sum() == 0 or d.loc[tr, "y"].nunique() < 2:
            continue
        a, b, _ = best_pair(d[tr], d.loc[tr, "y"], t_now)
        pred = (d.loc[te, "neck_drop"] >= a) | (d.loc[te, "face_grow"] >= b)
        d.loc[te, "pred_loso"] = pred
        yt = d.loc[te, "y"]
        neg_move = d.loc[te, "label"].isin(["상체숙임", "등받이기댐"]) & ~yt
        lrows.append({"person": p, "t1": a, "t2": b, "정확도": float((pred == yt).mean()), "균형": bal_acc(yt, pred),
                      "민감도": float(pred[yt].mean()) if yt.any() else np.nan,
                      "숙임·기댐 오탐": float(pred[neg_move].mean()) if neg_move.any() else np.nan})
    loso = pd.DataFrame(lrows)
    loso.to_csv(out / "loso.csv", index=False, encoding="utf-8-sig")

    # 4) 자세별 판정 비율: 지금 기준 / 데이터 기준(한 사람씩 빼고)
    d["pred_now"] = p_now
    rate = d.groupby("label")[["pred_now", "pred_loso"]].mean().reindex(order) * 100
    rate.columns = [f"지금 기준 ({t_now[0]:.0%}·{t_now[1]:.0%})", "데이터 기준 (한 사람씩 빼고)"]
    rate.round(1).to_csv(out / "03_자세별판정.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rate.plot.bar(ax=ax, color=["#9fb8d6", "#e45756"], width=0.75)
    ax.set_ylabel("거북목으로 판정한 비율 (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    for i, l in enumerate(rate.index):
        ax.text(i, 96, "맞혀야 함" if l in POSITIVE else "오탐", ha="center", fontsize=8,
                color="#c00" if l in POSITIVE else "#555")
    ax.set_title("자세별 '거북목' 판정 비율 (거북목 자세는 높을수록, 나머지는 낮을수록 좋음)")
    fig.tight_layout()
    fig.savefig(out / "03_자세별판정.png", dpi=130)
    plt.close(fig)

    # 5) 거리 보조: 확신도와 보조가 켜진 비율
    fw = d[d["engine"] == "forward"]
    asst = pd.DataFrame({
        "카메라만 확신도": fw.groupby("label")["conf_cam"].mean(),
        "거리 보조 포함 확신도": fw.groupby("label")["conf"].mean(),
        "A 머리만 가까움 (%)": d.groupby("label")["assist"].apply(lambda s: (s == "head_only").mean() * 100),
        "B 가슴만 뒤로 (%)": d.groupby("label")["assist"].apply(lambda s: (s == "chest_back").mean() * 100),
    }).reindex(order)
    asst.round(2).to_csv(out / "04_거리보조.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    asst[["카메라만 확신도", "거리 보조 포함 확신도"]].plot.bar(ax=axes[0], color=["#bbb", "#4c78a8"], width=0.75)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("거북목으로 판정됐을 때 확신도")
    asst[["A 머리만 가까움 (%)", "B 가슴만 뒤로 (%)"]].plot.bar(ax=axes[1], color=["#f58518", "#54a24b"], width=0.75)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("거리 보조가 켜진 비율 (거북목 자세에서 높아야 함)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "04_거리보조.png", dpi=130)
    plt.close(fig)

    # 6) 좌우 기울기
    d["tilt_y"] = d["label"].map(TILT_TRUTH).fillna("none")
    has_tilt = d["tilt_y"].ne("none").any()
    t_tilt_now = (rules.shoulder_angle_deg, rules.head_angle_deg, rules.head_offset)
    tilt_lines = ["", "## 좌우 기울기", ""]
    if has_tilt:
        tcues = [("d_shoulder", "어깨 선 각도 변화 (°)", 1, t_tilt_now[0]), ("d_eye", "눈 선 각도 변화 (°)", 1, t_tilt_now[1]),
                 ("d_offset", "코 옆이동 (어깨 너비 %)", 100, t_tilt_now[2] * 100)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (c, title, k, line) in zip(axes, tcues):
            data = [d.loc[d["label"] == l, c].dropna().to_numpy() * k for l in order]
            bp = ax.boxplot([x if len(x) else [np.nan] for x in data], showfliers=False, patch_artist=True)
            ax.set_xticks(range(1, len(order) + 1), order)
            for patch, l in zip(bp["boxes"], order):
                patch.set_facecolor({"left": "#54a24b", "right": "#b279a2"}.get(TILT_TRUTH.get(l), "#9fb8d6"))
            for sgn in (1, -1):
                ax.axhline(sgn * line, color="k", ls="--", lw=1)
            ax.axhline(0, color="#bbb", lw=0.8)
            ax.set_title(f"{title}  (점선 = 지금 기준 ±{line:g})")
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("자세별 기울기 단서 (+ = 사람 왼쪽, 초록 = 왼쪽기울기, 보라 = 오른쪽기울기)")
        fig.tight_layout()
        fig.savefig(out / "05_기울기단서.png", dpi=130)
        plt.close(fig)

        ty = d["tilt_y"].to_numpy()
        p_tnow = tilt_predict(d, *t_tilt_now)
        s1, s2, s3, sc_all = best_tilt(d, t_tilt_now)
        trows = []
        d["tilt_loso"] = "none"
        for p in people:
            tr, te = d["person"] != p, d["person"] == p
            if d.loc[tr, "tilt_y"].nunique() < 2:
                continue
            a, b, c, _ = best_tilt(d[tr], t_tilt_now)
            pr = tilt_predict(d[te], a, b, c)
            d.loc[te, "tilt_loso"] = pr
            yt = d.loc[te, "tilt_y"].to_numpy()
            neg = d.loc[te, "label"].isin(["머리내밀기", "어깨말림", "상체숙임", "등받이기댐"]).to_numpy()
            trows.append({"person": p, "어깨": a, "눈": b, "코": c, "정확도": float((pr == yt).mean()),
                          "균형": tilt_score(yt, pr),
                          "방향 맞춤": float((pr[yt != "none"] == yt[yt != "none"]).mean()),
                          "다른 자세 오탐": float((pr[neg] != "none").mean()) if neg.any() else np.nan})
        tl = pd.DataFrame(trows)
        tl.to_csv(out / "loso_tilt.csv", index=False, encoding="utf-8-sig")
        d["tilt_now"] = p_tnow
        rate_t = pd.DataFrame({
            "지금 기준 left": d.groupby("label")["tilt_now"].apply(lambda s: (s == "left").mean() * 100),
            "지금 기준 right": d.groupby("label")["tilt_now"].apply(lambda s: (s == "right").mean() * 100),
            "데이터 기준 left": d.groupby("label")["tilt_loso"].apply(lambda s: (s == "left").mean() * 100),
            "데이터 기준 right": d.groupby("label")["tilt_loso"].apply(lambda s: (s == "right").mean() * 100),
        }).reindex(order)
        rate_t.round(1).to_csv(out / "06_기울기판정.csv", encoding="utf-8-sig")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
        for ax, pre, title in ((axes[0], "지금 기준", f"지금 기준 ({t_tilt_now[0]:g}°·{t_tilt_now[1]:g}°·{t_tilt_now[2]:.0%})"),
                               (axes[1], "데이터 기준", "데이터 기준 (한 사람씩 빼고)")):
            rate_t[[f"{pre} left", f"{pre} right"]].plot.bar(ax=ax, stacked=True, color=["#54a24b", "#b279a2"], width=0.75)
            ax.set_ylim(0, 100)
            ax.set_title(title)
            ax.set_ylabel("기울기로 판정한 비율 (%)")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(["left", "right"], fontsize=8)
        fig.suptitle("자세별 기울기 판정 (왼쪽기울기 = 초록, 오른쪽기울기 = 보라만 높아야 함)")
        fig.tight_layout()
        fig.savefig(out / "06_기울기판정.png", dpi=130)
        plt.close(fig)

        def pc(v):
            return "-" if pd.isna(v) else f"{v:.1%}"
        tilt_lines += [
            "| | 어깨 선 | 눈 선 | 코 옆이동 | 균형 정확도 (같은 데이터) |", "|---|---|---|---|---|",
            f"| 지금 config | {t_tilt_now[0]:g}° | {t_tilt_now[1]:g}° | {t_tilt_now[2]:.0%} | {tilt_score(ty, p_tnow):.1%} |",
            f"| 데이터로 고른 값 (전원) | {s1:g}° | {s2:g}° | {s3:.0%} | {sc_all:.1%} |", "",
            "한 사람씩 빼고 시험:", "",
            "| 사람 | 어깨 | 눈 | 코 | 정확도 | 균형 정확도 | 방향 맞춤 | 다른 자세 오탐 |", "|---|---|---|---|---|---|---|---|",
        ]
        tilt_lines += [f"| {r['person']} | {r['어깨']:g}° | {r['눈']:g}° | {r['코']:.0%} | {pc(r['정확도'])} | {pc(r['균형'])} | "
                       f"{pc(r['방향 맞춤'])} | {pc(r['다른 자세 오탐'])} |" for _, r in tl.iterrows()]
        if len(tl):
            tilt_lines.append(f"| **평균** | | | | {pc(tl['정확도'].mean())} | {pc(tl['균형'].mean())} | "
                              f"{pc(tl['방향 맞춤'].mean())} | {pc(tl['다른 자세 오탐'].mean())} |")
        tilt_lines += ["", "```yaml", f"  shoulder_angle_deg: {s1:g}", f"  head_angle_deg: {s2:g}",
                       f"  head_offset: {s3:.2f}", "```"]
    else:
        tilt_lines.append("(왼쪽기울기·오른쪽기울기 자세 데이터 없음)")

    # 7) 정리
    def pct(v):
        return "-" if pd.isna(v) else f"{v:.1%}"
    lines = [
        "# 거북목·좌우 기울기 기준값 보고서", "",
        f"- 사람 {len(people)}명 ({', '.join(people)}), 판단 가능 프레임 {len(d)}개",
        f"- 정답: {'cva.csv(CVA < 48°) + ' if (folder / 'cva.csv').exists() else ''}수행 지시 (거북목 = {', '.join(POSITIVE)})",
        "",
        "## 거북목 기준값",
        "",
        "| | ① 목 높이 감소 | ② 얼굴 커짐 | 균형 정확도 (같은 데이터) |",
        "|---|---|---|---|",
        f"| 지금 config | {t_now[0]:.0%} | {t_now[1]:.0%} | {bal_acc(yy, p_now):.1%} |",
        f"| 데이터로 고른 값 (전원) | {t1:.0%} | {t2:.0%} | {ba_all:.1%} |",
        "",
        "## 한 사람씩 빼고 시험 (처음 보는 사람 기준, 발표용 숫자)",
        "",
    ]
    if len(loso):
        lines += ["| 사람 | 고른 ① | 고른 ② | 정확도 | 균형 정확도 | 민감도 | 숙임·기댐 오탐 |", "|---|---|---|---|---|---|---|"]
        for _, r in loso.iterrows():
            lines.append(f"| {r['person']} | {r['t1']:.0%} | {r['t2']:.0%} | {pct(r['정확도'])} | {pct(r['균형'])} | "
                         f"{pct(r['민감도'])} | {pct(r['숙임·기댐 오탐'])} |")
        lines.append(f"| **평균** | | | {pct(loso['정확도'].mean())} | {pct(loso['균형'].mean())} | "
                     f"{pct(loso['민감도'].mean())} | {pct(loso['숙임·기댐 오탐'].mean())} |")
    else:
        lines.append("(사람이 2명 이상이어야 계산)")
    lines += ["", "## ROC", "", "| 단서 | AUC | 가장 잘 나누는 값 |", "|---|---|---|"]
    lines += [f"| {n} | {a:.2f} | {t:.3f} |" for n, a, t in roc_rows]
    lines += ["", "## config.yaml에 넣을 값 (데이터로 정할 때)", "", "```yaml",
              f"  neck_drop_ratio: {t1:.2f}", f"  face_grow_ratio: {t2:.2f}", "```", "",
              "자세별 판정 비율·거리 보조: 03_자세별판정.csv, 04_거리보조.csv"] + tilt_lines
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
