"""단계별 좌면 모델 학습 — 2차 수집 11명 전원으로 ①②③ Random Forest를 학습해 실시간 엔진용 파일로 저장

왜 단계로 나누나 (ai/analysis/story 1~17단계)
    압력센서가 포화돼서(0 아니면 3500 근처) 5개 자세를 한 번에 구분하면 처음 보는 사람 기준 31~45%.
    쉬운 것부터 나눠서 판정한다:
        ① 바른자세 vs 나쁜 자세
        ② (나쁜 자세 중) 편향 vs 다리 꼬기        ← 60%, 앱에서는 "추정"으로 표시
        ③ (나쁜 자세 중) 체중이 실린 쪽 왼쪽 vs 오른쪽 (다리 꼬기는 꼰 다리 반대쪽에 체중)

모델 두 벌 (실시간 엔진이 기준 측정 상태를 보고 고른다)
    single  기준 1개   = 바른자세(중) 10초 기준만 있을 때 (지금 서버 흐름)
    depth   위치별 기준 = 바른자세를 앞·중·뒤에서 각 10초 잰 기준이 있을 때.
            머리 거리로 지금 앉은 위치를 추정(바른자세(중)보다 10cm 넘게 가까우면 앞, 멀면 뒤)하고 그 위치의 기준 사용

학습 규칙 (분석과 같음)
    - 2초(4개) 중앙값 묶음 하나 = 샘플 하나, 짧은 접촉 끊김 복구, 긴 끊김·움직임·마지막 시도가 아닌 구간은 학습에서 뺌
    - 특징: ai/features/pressure_pipeline.staged_features (실시간 엔진과 같은 함수).
      저장 전에 분석 코드(preprocess_report.features)와 값이 같은지 확인한다
    - 모델: Random Forest (story 15단계에서 단계마다 5종 비교로 선정), 11명 전원 학습
      (story 16단계: 문제 세션을 빼도 나아지지 않음)
    - --cv면 한 사람씩 빼고 시험한 정확도도 계산해 파일에 같이 저장 (약 3~5분)

실행 (저장소 최상위, .venv 켠 상태):
    python -m ai.classify.train_staged            # 학습 + 저장 (약 1분)
    python -m ai.classify.train_staged --cv       # + 한 사람씩 빼고 시험한 정확도 (발표 숫자 확인용)

저장: ai/models/final_staged_seat_model.joblib (이름이 final_로 시작 → git에 올라감, Pi에서 그대로 사용)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time

import numpy as np
import pandas as pd

log = logging.getLogger("sitsense.train_staged")

MODEL_OUT = "ai/models/final_staged_seat_model.joblib"
N_TREES = 300
POST = {"hold_sec": 4.0, "avg_sec": 2.0, "min_conf": 0.5}      # 후처리 (기존 모델과 같은 4초 유지)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[train_staged] %(message)s")
    ap = argparse.ArgumentParser(description="단계별 좌면 모델 학습 (2차 수집 11명)")
    ap.add_argument("--folder", default="data/dataset")
    ap.add_argument("--out", default=MODEL_OUT)
    ap.add_argument("--cv", action="store_true", help="한 사람씩 빼고 시험한 정확도도 계산")
    args = ap.parse_args()

    import joblib
    import sklearn
    from ai.analysis.preprocess_report import Ctx, features, make_model
    from ai.analysis.staged_report import (
        DEPTH_STEP_CM, depth_baselines, estimate_depth, head_baseline, loso_task, tasks,
    )
    from ai.engine.sitsense_engine import channel_names_from_positions
    from ai.features.pressure_pipeline import MovementParams, staged_features, show_path
    from common.config import load_config, repo_path

    t0 = time.time()
    cfg = load_config()
    channels = channel_names_from_positions(cfg["pressure"]["positions"])   # 서버가 넣는 순서 그대로
    ns = argparse.Namespace(folder=args.folder, fast=False, aug=False, no_timelines=True, rerun=True,
                            nested=False, step=[])
    out_dir = repo_path("data/processed/train_staged")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(ns, out_dir)
    if sorted(channels) != sorted(ctx.ch6):
        raise SystemExit(f"config.yaml 채널 {channels}과 데이터 채널 {ctx.ch6}이 다름")
    W, y, clean = ctx.W_fix, ctx.y, ctx.clean
    X = W[channels].to_numpy(dtype=float)

    # 기준: 기준 1개(사람마다 바른자세(중)) / 위치별(머리 거리로 추정한 위치의 바른자세)
    B1 = np.vstack([ctx.base[p][channels].to_numpy(dtype=float) for p in W["person"]])
    depth = estimate_depth(ctx)
    B2 = depth_baselines(ctx, depth)[channels].to_numpy(dtype=float)
    feats = {"single": staged_features(X, B1, channels), "depth": staged_features(X, B2, channels)}
    for k, F in feats.items():
        F.index = W.index
    # 분석 코드와 같은 값인지 확인 (학습-실시간 불일치 방지)
    ref = {"single": features(W, channels, "B", ctx.base),
           "depth": features(W, channels, "B", depth_baselines(ctx, depth))}
    for k in feats:
        if list(feats[k].columns) != list(ref[k].columns) or not np.allclose(feats[k].to_numpy(), ref[k].to_numpy()):
            raise SystemExit(f"{k}: 실시간 특징과 분석 특징이 다름 — staged_features 확인")
    log.info("특징 확인: 실시간 함수 == 분석 함수 (%d묶음 × %d특징)", len(W), feats["single"].shape[1])

    T = {t["key"]: t for t in tasks(y)}
    models, cv = {}, {}
    for kind, F in feats.items():
        models[kind] = {}
        for key in ("T1", "T2", "T3"):
            t = T[key]
            m = t["mask"] & clean
            mdl = make_model("rf", N_TREES).fit(F[m], t["target"][m])
            mdl.n_jobs = 1
            models[kind][key] = mdl
            log.info("  %-6s %s: %d묶음 학습, 클래스 %s", kind, t["name"], int(m.sum()), list(mdl.classes_))
            if args.cv:
                r = loso_task(ctx, F, t["target"], t["mask"], "rf", (), False, 200)
                cv[f"{kind}_{key}"] = round(r["mean"], 4)
                log.info("         한 사람씩 빼고 시험: %.1f%%", r["mean"] * 100)

    bundle = {
        "kind": "staged",
        "version": 1,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "sklearn": sklearn.__version__,
        "channels": channels,
        "feature_names": list(feats["single"].columns),
        "models": models,                              # {"single"|"depth": {"T1"|"T2"|"T3": RF}}
        "depth_step_cm": DEPTH_STEP_CM,
        "min_loaded": 3,                               # 기준 측정: 25% 이상 눌린 센서 3개 이상 (11명 전원 통과)
        "window": 4,
        "movement": vars(MovementParams()),
        "post": POST,
        "trained_on": sorted(W["person"].unique()),
        "n_train": int(clean.sum()),
        "head_baseline_cm": {p: round(float(v), 1) for p, v in head_baseline(ctx).items()},
        "cv_acc": cv,
        "notes": ["① 바른자세 vs 나쁜 자세 → ② 편향 vs 다리 꼬기(추정) → ③ 체중 쪽",
                  "다리 꼬기 체중 쪽: cross_left(왼다리 꼼) → right, cross_right → left"],
    }
    path = repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"저장: {show_path(path)}  ({path.stat().st_size / 1e6:.1f}MB, {time.time() - t0:.0f}초)")
    if cv:
        print(pd.Series(cv).map(lambda v: f"{v:.1%}").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
