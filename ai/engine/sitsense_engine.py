"""SitSense AI 엔진 — 센서 한 순간(Sample)을 받아 좌면·목·기울기·화면 거리를 한 번에 판정한다

서버(hardware/server/mock_engine.py)의 가짜 판정 judge_seat()·judge_upper()를 이걸로 바꾸면 된다.
결과 이름은 서버 /current 필드와 같다 (seat_state, head_state, tilt_state, confidence, deltas, cues, pending).

    from ai.engine.sitsense_engine import SitSenseEngine
    engine = SitSenseEngine(cfg)                      # cfg = common.config.load_config()
    res = engine.calibrate(samples)                   # 기준 자세 10초 동안 모은 Sample 목록
    if not res.ok: ...                                # res.reason: pressure_low / no_person (서버: 직전 기준 유지)
    r = engine.update(sample)                         # 매 순간 (0.1초든 0.5초든 상관없음)
    r.seat_state, r.head_state, r.tilt_state          # 이미 유지 시간을 거친 확정 상태
    r.to_dict()                                       # JSON에 바로 넣을 수 있는 dict
    engine.restore(pressure, distance_mm, pose)       # 서버에 저장된 기준(BaselineValue)으로 복원

    앉은 위치별 기준 (선택, 좌면 정확도 ↑ — docs/judgement_rules.md 5절)
    engine.calibrate_position("front", samples)       # calibrate() 뒤, 의자 앞쪽에 바르게 앉은 10초
    engine.calibrate_position("back", samples)        # 등받이까지 깊게 바르게 앉은 10초
    engine.position_base                              # {"front": [...], "back": [...]} → 서버가 저장
    engine.restore(..., positions=engine.position_base)   # 복원할 때 같이 넘기면 위치별 기준까지 복원
    (앞·뒤가 다 있으면 위치별 모델, 없으면 기준 1개 모델 — 서버를 안 바꿔도 동작)

    가슴 거리센서(거북목 확신도 보조): Sample에 아직 가슴 거리 칸이 없어서 따로 넘긴다
    engine.calibrate(samples, chest_mm=[…])           # 기준 측정 동안 가슴 거리 목록 (samples와 같은 순서, 실패는 None)
    engine.update(sample, chest_mm=…)                 # 이번 순간 가슴 거리 (없으면 생략 → 거북목 보조는 예전 방식)
    (sample.chest_mm 속성이 있으면 그것도 읽음 — common/sensors_base.Sample에 칸이 생기면 인자 없이 동작)

판정
    좌면    ai/engine/seat.py StagedSeatClassifier — 단계별 모델(ai/models/final_staged_seat_model.joblib, 2차 11명).
            ① 바른/나쁜 → ② 편향/다리꼬기(추정) → ③ 체중 쪽. 결과 값은 기존 5개 자세 그대로 (+ seat_side 등)
            예전 3명 모델 파일을 넘기면 SeatClassifier로 동작 (load_seat_classifier가 고름)
            모델이 0.5초 간격 데이터로 학습돼서, 서버가 더 자주 불러도 0.5초마다 한 번만 판정하고 사이에는 직전 결과.
            유지 시간·확신도는 모델 파일에 저장된 값(재생 테스트로 고른 값)으로 이미 적용됨
            → 서버가 좌면에 시간 필터를 또 걸면 안 됨 (늦어짐)
    목·기울기 ai/engine/upper_body.py 규칙 — 기준 자세 대비 변화량 (threshold는 config.yaml upper_rules,
            전체 기준표는 docs/upper_body_rules.md). 거북목 = 카메라(목 높이↓·얼굴 커짐) 1개 이상, 거리센서 2개는 확신도 보조
            최근 smooth_sec(2초) 특징 중앙값, 카메라 결과가 오래됐으면(POSE_MAX_AGE_SEC) 판단하지 않음,
            같은 상태가 short_filter_sec(4초) 이어져야 바꿈
    화면 거리 config.yaml thresholds (40cm 미만 warning / 40~50 caution / 50 이상 ok), 최근 2초 중앙값,
            short_filter_sec(4초) 유지 (seat.py)
    시간 기준 통일: 세 센서 모두 0.5초마다 판정 · 최근 2초로 떨림 줄이기 · 4초 유지해야 상태 변경
            (4초 = 압력 재생 테스트로 고른 값 + ISO 11226 '4초 넘게 유지 = 정적 자세')
    자리 비움이면 목·기울기는 unknown (서버 mock과 같은 규칙)

기준 자세 (서버 설계: 세션 시작·온보딩·다시 측정 때 10초)
    calibrate(samples)  압력: 모든 센서가 최대치의 25% 이상 눌려야 성공 (아니면 pressure_low)
                        카메라: 판단 가능한 프레임이 절반 이상이어야 성공 (아니면 no_person)
                        둘 다 성공해야 바뀐다. 실패하면 이전 기준 그대로 (서버는 using_previous)
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from ai.engine.seat import MODEL_PATH, SeatResult, StagedSeatClassifier, load_seat_classifier
from ai.engine.upper_body import Baseline, FeatureSmoother, StateFilter, UpperRules, judge
from ai.features.pose_features import UpperFeatures, upper_features
from common.schema import HeadState, SeatState, TiltState
from common.sensors_base import Landmark, PoseReading, Sample

log = logging.getLogger("sitsense.engine")

SEAT_PERIOD_SEC = 0.5           # 좌면 모델 판정 간격 (학습 데이터 간격)
POSE_MAX_AGE_SEC = 1.5          # 카메라 결과가 이보다 오래되면 판단하지 않음 (카메라 멈춤·꺼짐)
BASELINE_MIN_POSE_RATIO = 0.5   # 기준 측정 중 판단 가능한 카메라 프레임 비율 (서버와 같은 값)


@dataclass
class CalibrationResult:
    ok: bool
    reason: str | None              # None / pressure_low / no_person
    message: str
    pressure: list[float] | None = None
    distance_mm: float | None = None
    upper: dict | None = None       # 기준 상체 특징 (UpperFeatures)


@dataclass
class EngineResult:
    ts: float
    baseline_ready: bool
    seat_state: str
    head_state: str
    tilt_state: str
    confidence: dict                # {"seat", "head", "tilt"} 0~1
    pending: list                   # [{"kind", "state", "remaining_sec"}] 바뀌기를 기다리는 상태
    deltas: dict                    # 상체 기준 대비 변화량 (카메라 미인식이면 빈 dict)
    cues: dict                      # 어떤 단서가 기준값을 넘었는지
    distance_mm: float | None       # 최근 2초 거리 중앙값
    distance_level: str             # ok / caution / warning / unknown (short_filter_sec 유지)
    moving: bool                    # 자세 고쳐 앉는 중 (좌면 판정 멈춤)
    empty: bool                     # 이 순간 빈 의자로 보임
    pose_fresh: bool                # 최근 카메라 결과로 판단했는지
    seat_probs: dict = field(default_factory=dict)   # 좌면 모델 확률 (화면 표시·디버깅용)
    seat_side: str | None = None    # 체중이 실린 쪽 left / right (앱 확정 표시, 바른자세면 None)
    seat_kind_confidence: float = 0.0   # 편향 vs 다리 꼬기 확신도 (앱은 "추정"으로 표시)
    sit_position: str = "unknown"   # 머리 거리로 추정한 앉은 위치 front / middle / back
    seat_baseline_mode: str = ""    # single(기준 1개) / depth(앉은 위치별 기준)

    def to_dict(self) -> dict:
        return asdict(self)


def channel_names_from_positions(positions) -> list[str]:
    """config.yaml pressure.positions (x: 왼 -1 ~ 오 +1, y: 뒤 -1 ~ 앞 +1) → R_back 같은 위치 이름"""
    names = []
    for x, y in np.asarray(positions, dtype=float):
        side = "R" if x > 0 else "L" if x < 0 else "C"
        row = "front" if y > 0.4 else "back" if y < -0.4 else "mid"
        names.append(f"{side}_{row}")
    return names


class SitSenseEngine:
    def __init__(self, cfg: dict, model=MODEL_PATH, *, require_camera: bool = True):
        """require_camera=False면 카메라 없이도 기준 측정 성공 (목·기울기는 unknown)"""
        th = cfg["thresholds"]
        self.seat = load_seat_classifier(model, overrides={      # 거리 기준·유지 시간은 config.yaml 하나에서
            "dist_warn_cm": th["distance_warning_mm"] / 10, "dist_caution_cm": th["distance_ok_mm"] / 10,
            "dist_hold_sec": float(th["short_filter_sec"])})
        names = channel_names_from_positions(cfg["pressure"]["positions"])
        if sorted(names) != sorted(self.seat.channels):
            raise ValueError(f"config.yaml 압력 센서 위치 {names}와 모델 센서 위치 {self.seat.channels}가 다름 "
                             f"— 센서 구성이 바뀌었으면 모델을 다시 학습해야 함")
        # 서버는 config.yaml 순서(p0, p1, …)로 값을 준다. 모델은 위치 이름으로 학습했으므로 이름으로 맞춰 넣는다
        # (순서만 다르면 다시 학습할 필요 없음). 서버에 돌려주는 기준값은 다시 config 순서로
        self.channel_names = names
        self._to_model = [names.index(c) for c in self.seat.channels]
        self._to_cfg = [self.seat.channels.index(c) for c in names]
        self.staged = isinstance(self.seat, StagedSeatClassifier)
        self.position_base: dict[str, list[float]] = {}
        self.rules = UpperRules.from_config(cfg)
        self.hold = float(th["short_filter_sec"])
        self.width, self.height = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])
        self.require_camera = require_camera
        self.upper_base = Baseline(seconds=float(th["baseline_seconds"]))
        self.pressure_base: list[float] | None = None
        self._reset_state()

    def _m(self, values) -> list:
        """config 순서 → 모델 순서"""
        v = list(values)
        return [v[i] for i in self._to_model]

    def _c(self, values) -> list:
        """모델 순서 → config 순서"""
        v = list(values)
        return [v[i] for i in self._to_cfg]

    # ------------------------------------------------------------------ 상태
    @property
    def baseline_ready(self) -> bool:
        return self.pressure_base is not None

    def _reset_state(self) -> None:
        self.smoother = FeatureSmoother(self.rules.smooth_sec)
        self.head_filter = StateFilter(self.hold)
        self.tilt_filter = StateFilter(self.hold)
        self._seat_r: SeatResult | None = None
        self._seat_ts: float | None = None
        self._pose_ts: float | None = None
        self._feat: UpperFeatures | None = None
        self._dists: deque = deque()          # (ts, 머리 mm, 가슴 mm) 최근 smooth_sec초 → 거북목 거리 단서는 중앙값으로

    # ------------------------------------------------------------------ 기준 자세
    def calibrate(self, samples: list[Sample], chest_mm: list | None = None) -> CalibrationResult:
        """기준 자세 동안 모은 Sample → 기준 설정. 실패하면 이전 기준을 그대로 둔다
        chest_mm: samples와 같은 순서의 가슴 거리 목록 (없으면 sample.chest_mm 속성, 그것도 없으면 가슴 기준 없음)"""
        if not samples:
            return CalibrationResult(False, "pressure_low", "기준 측정 값이 없음")
        feats = [f for f in (self._features(s.pose) for s in samples) if f is not None]
        if self.require_camera and len(feats) < BASELINE_MIN_POSE_RATIO * len(samples):
            return CalibrationResult(False, "no_person",
                                     f"카메라로 상체를 찾은 프레임이 {len(feats)}/{len(samples)} — 정면에 앉았는지 확인")
        rows = [self._m(s.pressure.values) for s in samples]
        dists = [s.distance.distance_mm for s in samples if s.distance.valid]
        dist = float(np.median(dists)) if dists else None
        try:                                          # 실패하면 ValueError, 좌면 기준은 그대로
            if self.staged:
                base = self.seat.calibrate(rows, head_cm=None if dist is None else dist / 10)
            else:
                base = self.seat.calibrate(rows)
        except ValueError as e:
            return CalibrationResult(False, "pressure_low", str(e))
        self.position_base = {}
        chests = chest_mm if chest_mm is not None else [getattr(s, "chest_mm", None) for s in samples]
        chests = [float(v) for v in chests if v is not None and math.isfinite(v) and v >= self.rules.chest_min_mm]
        chest = float(np.median(chests)) if len(chests) >= BASELINE_MIN_POSE_RATIO * len(samples) else None
        upper = None
        if feats:
            keys = [f.name for f in fields(UpperFeatures)]
            upper = UpperFeatures(**{k: float(np.median([getattr(f, k) for f in feats])) for k in keys})
        self._commit(self._c(base.round(1).tolist()), dist, upper, chest)
        return CalibrationResult(True, None, "기준 측정 성공", self.pressure_base, dist,
                                 None if upper is None else upper.as_dict())

    def calibrate_position(self, position: str, samples: list[Sample]) -> CalibrationResult:
        """(선택) 앞(front)·뒤(back)에 바르게 앉은 10초 → 앉은 위치별 좌면 기준. calibrate() 성공 뒤에만.
        실패(덜 눌린 센서)하면 pressure_low, 그 위치 기준은 그대로"""
        if not self.staged:
            return CalibrationResult(False, "unsupported", "예전 5개 자세 모델은 위치별 기준을 쓰지 않음")
        if not self.baseline_ready:
            return CalibrationResult(False, "no_baseline", "바른자세(중) 기준 측정을 먼저 해야 함")
        try:
            b = self.seat.calibrate_position(position, [self._m(s.pressure.values) for s in samples])
        except ValueError as e:
            return CalibrationResult(False, "pressure_low", str(e))
        self.position_base[position] = self._c(b.round(1).tolist())
        self._seat_r, self._seat_ts = None, None
        return CalibrationResult(True, None, f"{position} 기준 측정 성공", self.position_base[position])

    def restore(self, pressure, distance_mm: float | None = None, pose: dict | None = None,
                chest_mm: float | None = None, positions: dict | None = None) -> None:
        """서버에 저장된 기준(BaselineValue: pressure, distance_mm, pose {점: [x, y, z, visibility]})으로 복원.
        positions = {"front": [...], "back": [...]} (calibrate_position 결과, 없으면 기준 1개).
        이미 검사를 통과한 값이라 다시 검사하지 않는다"""
        if self.staged:
            self.seat.set_baseline(self._m(pressure), head_cm=None if distance_mm is None else distance_mm / 10,
                                   positions={k: self._m(v) for k, v in (positions or {}).items()})
            self.position_base = {k: list(v) for k, v in (positions or {}).items()}
        else:
            self.seat.set_baseline(self._m(pressure))
        upper = None
        if pose:
            pr = PoseReading(ts=0.0, detected=True,
                             points={n: Landmark(*map(float, v[:4])) for n, v in pose.items()})
            upper = self._features(pr)
        self._commit([round(float(v), 1) for v in pressure], distance_mm, upper, chest_mm)

    def _commit(self, pressure, distance_mm, upper, chest_mm=None) -> None:
        self.pressure_base = pressure
        self.upper_base.reset()
        self.upper_base.value = upper
        self.upper_base.distance_mm = distance_mm
        self.upper_base.chest_mm = chest_mm
        self._reset_state()

    # ------------------------------------------------------------------ 판정
    def update(self, sample: Sample, chest_mm: float | None = None) -> EngineResult:
        ts = sample.ts
        d = sample.distance
        dist_mm = float(d.distance_mm) if d.valid else None
        if chest_mm is None:
            chest_mm = getattr(sample, "chest_mm", None)
        head_med, chest_med = self._distance_medians(ts, dist_mm, chest_mm)
        if not self.baseline_ready:
            return self._unknown(ts, dist_mm)

        # 좌면: 0.5초마다 한 번 (사이에는 직전 결과)
        if self._seat_ts is None or ts - self._seat_ts >= SEAT_PERIOD_SEC - 1e-6:
            try:
                self._seat_r = self.seat.update(ts, self._m(sample.pressure.values),
                                                None if dist_mm is None else dist_mm / 10)
                self._seat_ts = ts
            except ValueError as e:                   # 이상한 압력 값 → 이번 순간 건너뜀 (상태 그대로)
                log.warning("압력 값 건너뜀: %s", e)
        r = self._seat_r
        if r is None:
            return self._unknown(ts, dist_mm)

        # 목·기울기: 새 카메라 결과가 왔을 때만 특징을 갱신
        pose = sample.pose
        fresh = pose is not None and ts - pose.ts <= POSE_MAX_AGE_SEC
        if not fresh:
            self.smoother.reset()
            self._feat, self._pose_ts = None, None
        elif pose.ts != self._pose_ts:
            self._pose_ts = pose.ts
            self._feat = self.smoother.update(self._features(pose), pose.ts)
        seated = r.state != SeatState.EMPTY.value and not r.empty
        if seated:
            up = judge(self._feat, self.upper_base, self.rules, distance_mm=head_med, chest_mm=chest_med)
            head_raw, tilt_raw = up.head, up.tilt
            head_conf, tilt_conf, deltas, cues = up.head_confidence, up.tilt_confidence, up.deltas, up.cues
        else:                                         # 자리 비움이면 목·기울기는 판단하지 않음
            head_raw, tilt_raw, head_conf, tilt_conf, deltas, cues = (
                HeadState.UNKNOWN, TiltState.UNKNOWN, 0.0, 0.0, {}, {})
        head = self.head_filter.update(head_raw, ts)
        tilt = self.tilt_filter.update(tilt_raw, ts)

        return EngineResult(
            ts=ts, baseline_ready=True, seat_state=r.state, head_state=head.value, tilt_state=tilt.value,
            confidence={"seat": r.confidence, "head": round(float(head_conf), 3), "tilt": round(float(tilt_conf), 3)},
            pending=self._pending(ts),
            deltas={k: round(float(v), 4) for k, v in deltas.items()},
            cues={k: bool(v) for k, v in cues.items()},
            distance_mm=None if r.distance_cm is None else round(r.distance_cm * 10, 1),
            distance_level=r.distance_state, moving=r.moving, empty=r.empty, pose_fresh=bool(fresh),
            seat_probs=r.probs, seat_side=r.weight_side, seat_kind_confidence=r.kind_confidence,
            sit_position=r.sit_position, seat_baseline_mode=r.baseline_mode)

    def _distance_medians(self, ts: float, head: float | None, chest: float | None):
        """최근 smooth_sec초 머리·가슴 거리 중앙값 (튀는 값 완화).
        가슴 0·막힘(chest_min_mm 미만)은 결측으로 빼고 2초 안의 정상값 중앙값으로 채움, 정상값이 없으면 None"""
        ok_chest = chest is not None and math.isfinite(chest) and chest >= self.rules.chest_min_mm
        self._dists.append((ts, head, chest if ok_chest else None))
        while self._dists and self._dists[0][0] <= ts - self.rules.smooth_sec:
            self._dists.popleft()
        hs = [h for _, h, _ in self._dists if h is not None]
        cs = [c for _, _, c in self._dists if c is not None]
        return (float(np.median(hs)) if hs else None), (float(np.median(cs)) if cs else None)

    def _features(self, pose: PoseReading | None) -> UpperFeatures | None:
        f = upper_features(pose, self.width, self.height, self.rules.min_visibility)
        if f is not None and not all(math.isfinite(v) for v in f.as_dict().values()):
            return None
        return f

    def _pending(self, ts: float) -> list[dict]:
        out = []
        post = self.seat.post
        if post.cand is not None and post.cand != post.state and post.cand_since is not None:
            out.append({"kind": "seat", "state": post.cand,
                        "remaining_sec": round(max(0.0, self.seat.params.hold_sec - (ts - post.cand_since)), 1)})
        for kind, f in (("head", self.head_filter), ("tilt", self.tilt_filter)):
            p = f.pending(ts)
            if p is not None:
                out.append({"kind": kind, "state": p[0].value, "remaining_sec": round(p[1], 1)})
        return out

    def _unknown(self, ts: float, dist_mm: float | None) -> EngineResult:
        u = "unknown"
        return EngineResult(ts=ts, baseline_ready=self.baseline_ready, seat_state=u, head_state=u, tilt_state=u,
                            confidence={"seat": 0.0, "head": 0.0, "tilt": 0.0}, pending=[], deltas={}, cues={},
                            distance_mm=dist_mm, distance_level=u, moving=False, empty=False, pose_fresh=False)
