"""거북목 데이터 수집 — 카메라(상체 점) + 거리센서 2개(머리·가슴)를 같이 기록 (Raspberry Pi에서 실행)

한 번 실행 = 한 사람 = 자세 6개 (각 30초, 사이 5초 쉬기) ≈ 3분 30초
    1 바른자세        기준 (분석에서 처음 10초를 기준값으로 씀)
    2 머리내밀기      거북목 A: 몸통은 그대로, 머리만 화면 쪽으로
    3 어깨말림        거북목 B: 머리는 그대로, 어깨를 말고 등을 구부정하게 (가슴이 뒤로)
    4 상체숙임        거북목 아님: 허리부터 몸 전체를 앞으로
    5 등받이기댐      거북목 아님: 등받이에 기대기
    6 바른자세2       마무리
쉬는 5초 동안 다음 자세로 바꾼다 (phase=rest로 저장, 분석에서는 안 씀).

옆모습 정답(CVA): 시작 전에 휴대폰으로 옆에서 영상 녹화를 켜 두고, 자세가 바뀔 때마다 화면 안내를 소리 내어 읽는다.
나중에 영상에서 자세마다 한 장씩 골라 귀–목(C7)–수평선 각도를 잰다 (data/dataset/neck/cva.csv, 선택).

사용 예 (저장소 최상위, .venv 켠 상태, Pi에서)
    python -m ai.camera.collect_neck --check                     # 카메라·거리 2개 확인 (저장 안 함)
    python -m ai.camera.collect_neck --subject n1
    python -m ai.camera.collect_neck --subject n1 --clothes loose --only 머리내밀기 어깨말림
    python -m ai.camera.collect_neck --source mock --subject test --seconds 3 --rest 1     # 장비 없이 코드만 확인
    ※ 거리센서 핀 = 압력 수집 프로그램(hardware/sensors/collect_all1117.py)과 같음: 머리 23/24(config.yaml), 가슴 5/6.
      다르면 --chest-trig/--chest-echo. 압력 수집 프로그램과 동시에 실행하지 말 것 (같은 핀).

저장: data/dataset/neck/{subject}_{clothes}_{YYYYMMDD_HHMMSS}.csv   한 줄 = 카메라 한 프레임
    ts, subject, clothes, phase(rest/record), label, seg, t_seg, head_cm, chest_cm, img_w, img_h, pose_detected,
    {부위}_x/_y/_z/_vis (common.schema.LANDMARK_NAMES 7개), neck_height, face_ratio (판단 불가면 빈칸)
    거리는 측정 안 되면 빈칸. 이미지는 저장하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import warnings
from datetime import datetime

from ai.camera.collect import open_stream, pose_to_row, say
from common.config import load_config, repo_path
from common.recorder import SUBJECT_RE
from common.schema import LANDMARK_NAMES
from common.sensors_base import SensorError

log = logging.getLogger("sitsense.collect_neck")

POSTURES = [
    ("바른자세", "바르게 앉아 화면을 보세요 (기준)"),
    ("머리내밀기", "몸통은 그대로, 머리만 화면 쪽으로 쭉 내미세요"),
    ("어깨말림", "머리 위치는 그대로, 어깨를 앞으로 말고 등을 구부정하게"),
    ("상체숙임", "허리부터 몸 전체를 앞으로 숙이세요 (목은 곧게)"),
    ("등받이기댐", "등받이에 편하게 기대세요"),
    ("바른자세2", "다시 바르게 앉으세요 (마무리)"),
]
LABELS = [p for p, _ in POSTURES]
OUT_SUBDIR = "data/dataset/neck"


class Distances:
    """HC-SR04 두 개 (gpiozero가 배경에서 계속 재고 최근 5개 중앙값을 줌). 없으면 None"""

    def __init__(self, cfg: dict, chest_pins: tuple[int, int] | None, mock: bool = False):
        self.mock, self.t0 = mock, time.time()
        self.head = self.chest = None
        self.max_m = cfg["distance"]["max_mm"] / 1000.0
        if mock:
            return
        try:
            from gpiozero import DistanceSensor, DistanceSensorNoEcho
        except ImportError as e:
            raise SensorError("gpiozero가 없음 — Pi에서 실행하거나 --no-distance") from e
        warnings.simplefilter("ignore", DistanceSensorNoEcho)
        d = cfg["distance"]
        self.head = DistanceSensor(echo=d["echo_pin"], trigger=d["trigger_pin"], max_distance=self.max_m,
                                   queue_len=d["median_window"])
        if chest_pins is not None:
            self.chest = DistanceSensor(echo=chest_pins[1], trigger=chest_pins[0], max_distance=self.max_m,
                                        queue_len=d["median_window"])

    def _cm(self, s):
        if s is None:
            return None
        m = s.distance
        return None if m >= self.max_m - 1e-6 else round(m * 100, 1)

    def read(self) -> tuple[float | None, float | None]:
        if self.mock:
            t = time.time() - self.t0
            return round(60 + 2 * math.sin(t), 1), round(30 + math.sin(t / 2), 1)
        return self._cm(self.head), self._cm(self.chest)

    def close(self):
        for s in (self.head, self.chest):
            if s is not None:
                s.close()


def columns() -> list[str]:
    cols = ["ts", "subject", "clothes", "phase", "label", "seg", "t_seg", "head_cm", "chest_cm",
            "img_w", "img_h", "pose_detected"]
    cols += [f"{n}_{f}" for n in LANDMARK_NAMES for f in ("x", "y", "z", "vis")]
    return cols + ["neck_height", "face_ratio"]


def fmt(v, nd=1):
    return "" if v is None else round(v, nd)


def run_check(stream, dist, seconds: float, min_vis: float) -> None:
    from ai.features.pose_features import upper_features
    say(f"확인 {seconds:g}초 (저장 안 함). 바르게 앉아 주세요.")
    t0 = last = time.time()
    n = ok = 0
    while time.time() - t0 < seconds:
        pose, _ = stream.read()
        n += 1
        f = upper_features(pose, stream.width, stream.height, min_vis)
        ok += f is not None
        if time.time() - last >= 1.0:
            h, c = dist.read()
            say(f"  카메라 판단 가능 {ok}/{n}  머리 {h if h is not None else '측정 안 됨'} cm  "
                f"가슴 {c if c is not None else '측정 안 됨'} cm")
            last = time.time()
    h, c = dist.read()
    if c is None:
        say("⚠ 가슴 거리가 측정되지 않아요 — 가슴 센서 배선(5/6번 핀)과 방향 확인")
    if n and ok / n < 0.7:
        say("⚠ 카메라 판단 가능 비율이 낮아요 — 조명, 어깨까지 화면에 들어오는지 확인")


def run_collect(stream, dist, args, cfg) -> None:
    from ai.features.pose_features import upper_features
    min_vis = cfg["upper_rules"]["min_visibility"]
    plan = [(k, p, d) for k, (p, d) in enumerate(POSTURES, 1) if not args.only or p in args.only]
    out = repo_path(OUT_SUBDIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.subject}_{args.clothes}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    say(f"\n자세 {len(plan)}개 × {args.seconds:g}초 (사이 {args.rest:g}초). 저장: {path.relative_to(repo_path('.'))}")
    say("휴대폰 옆모습 녹화를 켜고, 자세가 바뀔 때마다 안내를 소리 내어 읽어 주세요.")
    input("준비되면 Enter… ")
    counts = {}
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(columns())
        for k, label, desc in plan:
            for phase, seconds in (("rest", args.rest), ("record", args.seconds)):
                if phase == "rest":
                    say(f"\a\n[{k}/{len(POSTURES)}] 다음: {label} — {desc}  ({seconds:g}초 뒤 시작)")
                else:
                    say(f"  ● 기록 중: {label} ({seconds:g}초)")
                t0 = last = time.time()
                while (el := time.time() - t0) < seconds:
                    pose, _ = stream.read()
                    head, chest = dist.read()
                    f = upper_features(pose, stream.width, stream.height, min_vis)
                    det = int(pose is not None and pose.detected)
                    lm = pose_to_row(pose) if pose is not None else [""] * (4 * len(LANDMARK_NAMES))
                    w.writerow([datetime.now().isoformat(timespec="milliseconds"), args.subject, args.clothes,
                                phase, label, k, round(el, 2), fmt(head), fmt(chest), stream.width,
                                stream.height, det, *lm,
                                "" if f is None else round(f.neck_height, 5),
                                "" if f is None else round(f.face_ratio, 5)])
                    if phase == "record":
                        counts[label] = counts.get(label, 0) + (f is not None)
                    if time.time() - last >= 5:
                        say(f"    {el:4.0f}s  머리 {head} cm  가슴 {chest} cm  "
                            f"{'카메라 OK' if f is not None else '⚠ 카메라 판단 불가'}")
                        last = time.time()
                fh.flush()
    say("\n끝. 자세별 카메라 판단 가능 프레임 수:")
    for _, label, _ in plan:
        say(f"  {label}: {counts.get(label, 0)}")
    say(f"저장: {path.relative_to(repo_path('.'))}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="거북목 수집: 카메라 + 머리·가슴 거리")
    ap.add_argument("--subject", help="사람 ID (n1, n2 …, 실명 금지)")
    ap.add_argument("--clothes", choices=["tight", "loose"], default="tight", help="옷: tight(붙는 옷, 기본) / loose")
    ap.add_argument("--seconds", type=float, default=30, help="자세당 기록 시간(초)")
    ap.add_argument("--rest", type=float, default=5, help="자세 사이 쉬는 시간(초)")
    ap.add_argument("--only", nargs="+", choices=LABELS, help="일부 자세만 (예: 헐렁한 옷으로 머리내밀기 어깨말림)")
    ap.add_argument("--chest-trig", type=int, default=5, help="가슴 HC-SR04 trigger 핀 (BCM, 기본 5 = collect_all1117과 같음)")
    ap.add_argument("--chest-echo", type=int, default=6, help="가슴 HC-SR04 echo 핀 (BCM, 기본 6)")
    ap.add_argument("--no-distance", action="store_true", help="거리센서 없이 (카메라만)")
    ap.add_argument("--source", choices=["picam", "webcam", "mock"], default="picam")
    ap.add_argument("--index", type=int, default=None, help="webcam 번호")
    ap.add_argument("--model", type=int, choices=[0, 1, 2], default=None, help="MediaPipe 모델 (기본 config)")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--rotate180", action="store_true")
    ap.add_argument("--check", action="store_true", help="저장 없이 카메라·거리 확인")
    ap.add_argument("--check-seconds", type=float, default=15)
    a = ap.parse_args()
    if not a.check:
        if not a.subject or not SUBJECT_RE.match(a.subject):
            ap.error("--subject를 영문·숫자로 지정하세요 (예: n1)")
        if a.source == "mock" and a.subject not in ("test", "mock"):
            ap.error("--source mock은 --subject test로만")
    if (a.chest_trig is None) != (a.chest_echo is None):
        ap.error("--chest-trig와 --chest-echo는 같이 지정하세요")
    if a.chest_trig is None and not a.no_distance and a.source != "mock":
        say("⚠ 가슴 센서 핀을 지정하지 않아 머리 거리만 기록합니다 (--chest-trig/--chest-echo)")
    return a


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[collect_neck] %(message)s")
    stream = dist = None
    try:
        cfg = load_config()
        args = parse_args()
        if args.model is None:
            args.model = cfg["camera"].get("model_complexity", 1)
        chest = None if args.chest_trig is None else (args.chest_trig, args.chest_echo)
        dist = Distances(cfg, chest, mock=args.source == "mock" or args.no_distance)
        if args.no_distance:
            dist.read = lambda: (None, None)
        say("카메라를 여는 중… (MediaPipe 준비에 몇 초)")
        stream = open_stream(args, cfg)
        if args.check:
            run_check(stream, dist, args.check_seconds, cfg["upper_rules"]["min_visibility"])
        else:
            run_collect(stream, dist, args, cfg)
    except KeyboardInterrupt:
        say("\nCtrl+C — 그때까지 기록한 줄은 파일에 남아 있어요")
        return 130
    except SensorError as e:
        log.error("%s", e)
        return 2
    finally:
        for obj in (stream, dist):
            if obj is not None:
                obj.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
