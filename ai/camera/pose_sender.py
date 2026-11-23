"""노트북 웹캠으로 MediaPipe 좌표를 뽑아 Pi로 보낸다 — Pi에 USB 웹캠을 못 꽂을 때 쓰는 대체 경로

Pi 본체에 꽂은 카메라(Pi 카메라 모듈/USB 웹캠) 대신, 노트북 웹캠 + 이 노트북의 CPU로
MediaPipe를 돌리고, 결과 좌표(숫자 몇 개)만 같은 Wi-Fi로 Pi에 보낸다. 영상 자체는 보내지 않는다.
Pi 쪽은 hardware/sensors/network_pose.py의 NetworkPoseSource가 그 좌표를 받아서, 실제 카메라가
붙어있을 때와 똑같이 서버에 공급한다.

사용 예 (저장소 최상위, 노트북에서 .venv 켠 상태)
    python -m ai.camera.pose_sender --host 192.168.0.23               # Pi IP, 기본 webcam index 0
    python -m ai.camera.pose_sender --host 192.168.0.23 --index 1 --model 0
    python -m ai.camera.pose_sender --host 192.168.0.23 --no-preview  # 미리보기 창 없이

먼저 Pi에서 이걸 돌려 포트를 열어둬야 함:
    python -m hardware.server.mock_server --mode real --camera network --pose-port 8765 --autostart

미리보기 창(노트북 화면)에 초록/빨강 점으로 visibility 표시. q 또는 Ctrl+C로 종료.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger("sitsense.pose_sender")


def send(url: str, pose, timeout: float = 0.5) -> bool:
    points = {}
    if pose is not None and pose.detected:
        for name, lm in pose.points.items():
            points[name] = {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
    payload = json.dumps({
        "ts": time.time(),
        "detected": bool(pose is not None and pose.detected),
        "points": points,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("Pi로 전송 실패 (%s) — Pi가 켜져있고 같은 Wi-Fi인지 확인", e)
        return False


def draw_preview(cv2, img, pose, min_vis: float):
    frame = img.copy()
    h, w = frame.shape[:2]
    if pose is not None and pose.detected:
        for _, lm in pose.points.items():
            if lm.is_finite():
                c = (0, 200, 0) if lm.visibility >= min_vis else (0, 0, 255)
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, c, -1)
    else:
        cv2.putText(frame, "NO POSE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="노트북 웹캠 좌표를 Pi로 전송")
    ap.add_argument("--host", required=True, help="Pi IP (같은 Wi-Fi), 예: 192.168.0.23")
    ap.add_argument("--port", type=int, default=8765, help="Pi의 NetworkPoseSource 포트 (기본 8765)")
    ap.add_argument("--index", type=int, default=0, help="노트북 웹캠 번호 (기본 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=5, help="Pi로 보내는 속도 (초당 번, 기본 5)")
    ap.add_argument("--model", type=int, choices=[0, 1, 2], default=0,
                    help="MediaPipe 모델 0=lite(기본, 빠름)/1=full/2=heavy")
    ap.add_argument("--min-vis", type=float, default=0.5, help="미리보기에서 점 색을 가르는 기준")
    ap.add_argument("--no-preview", action="store_true", help="미리보기 창 띄우지 않기")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[pose_sender] %(message)s")

    from ai.camera.pose import PoseExtractor, Webcam
    cam = Webcam(index=args.index, width=args.width, height=args.height)
    extractor = PoseExtractor(model_complexity=args.model)
    url = f"http://{args.host}:{args.port}/pose"
    period = 1.0 / args.fps

    cv2 = None
    if not args.no_preview:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            log.warning("cv2 못 불러옴 — 미리보기 없이 계속")

    print(f"Pi로 전송 시작: {url}  (웹캠 index={args.index}, {cam.width}x{cam.height}, "
          f"model={args.model}, {args.fps:g}fps)")
    print("Ctrl+C로 종료" + (" (또는 미리보기 창에서 q)" if cv2 else ""))

    ok_count = fail_count = 0
    try:
        while True:
            t0 = time.time()
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            pose = extractor.extract(frame)
            if send(url, pose):
                ok_count += 1
            else:
                fail_count += 1
            if cv2 is not None:
                prev = draw_preview(cv2, frame.image, pose, args.min_vis)
                cv2.imshow("pose_sender (q=종료)", prev)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if (ok_count + fail_count) % 20 == 0:
                print(f"  보냄 {ok_count} / 실패 {fail_count}")
            rest = period - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료")
    finally:
        extractor.close()
        cam.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
