"""노트북 웹캠에서 뽑은 좌표를 네트워크로 받아 쓰는 PoseSource — Pi에 USB 웹캠을 못 꽂을 때용

노트북에서 ai/camera/pose_sender.py를 돌리면 좌표(JSON, 숫자 몇 개뿐이라 가벼움)를 이 포트로
0.2~0.5초마다 POST한다. Pi 쪽은 그 값을 "가장 최근 결과"로 들고 있다가, CameraPoseSource와
똑같은 모양(PoseSource)으로 read()에 돌려준다. 그래서 factory.build_hub(pose=...)에 이 객체를
끼워 넣기만 하면 나머지(엔진·서버)는 카메라가 실제로 Pi에 붙어있을 때와 똑같이 동작한다.

영상 자체는 네트워크로 안 보낸다 — 좌표(숫자 7개 점 × 4개 값)만 오가서 가볍고,
이미지를 저장·전송하지 않는다는 원래 규칙도 그대로 지켜진다.

쓰는 법 (Pi, 저장소 최상위):
    python main.py --mode real --camera network --pose-port 8765
    python -m hardware.server.mock_server --mode real --camera network --pose-port 8765 --autostart

노트북 쪽은 ai/camera/pose_sender.py 참고.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common.sensors_base import Landmark, PoseReading, PoseSource

log = logging.getLogger("sitsense.network_pose")

ON, OFF, WARMING = "on", "off", "warming"      # hardware/sensors/power.py와 같은 값
STALE_SEC = 1.0                                # 이보다 오래된 값이면 '연결 끊김'으로 보고 None


class NetworkPoseSource(PoseSource):
    name = "network_pose"

    def __init__(self, port: int = 8765):
        self.port = port
        self._lock = threading.Lock()
        self._latest: PoseReading | None = None
        self._received_at = 0.0
        self._server: ThreadingHTTPServer | None = None
        self._on = False

    # ------------------------------------------------------------------ 전원 (power.py의 Switchable 약속)
    @property
    def is_on(self) -> bool:
        return self._on

    def power_on(self, ts: float | None = None) -> None:
        if self._on:
            return
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), self._make_handler())
        threading.Thread(target=self._server.serve_forever, daemon=True, name="network-pose").start()
        self._on = True
        log.info("좌표 수신 서버 켬 (포트 %d) — 노트북에서 ai/camera/pose_sender.py를 돌려야 값이 들어옴",
                 self.port)

    def power_off(self, ts: float | None = None) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        self._on = False
        with self._lock:
            self._latest = None

    def state(self, ts: float | None = None) -> str:
        if not self._on:
            return OFF
        with self._lock:
            return WARMING if self._latest is None else ON

    # ------------------------------------------------------------------ 읽기
    def read(self) -> PoseReading | None:
        with self._lock:
            latest, received_at = self._latest, self._received_at
        if latest is None:
            return None
        if time.time() - received_at > STALE_SEC:
            return None          # 노트북 쪽이 멈췄거나 네트워크가 끊겼을 때 오래된 값을 계속 쓰지 않음
        return latest

    def close(self) -> None:
        self.power_off()

    # ------------------------------------------------------------------ 수신 서버
    def _make_handler(self):
        src = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/pose":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    data = json.loads(body)
                    points = {
                        name: Landmark(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]),
                                       visibility=float(p["visibility"]))
                        for name, p in data.get("points", {}).items()
                    }
                    pose = PoseReading(ts=float(data["ts"]), detected=bool(data["detected"]), points=points)
                except Exception as e:  # noqa: BLE001 — 형식이 잘못된 요청은 그냥 버림
                    log.warning("좌표 수신 실패 (형식 오류): %s", e)
                    self.send_error(400)
                    return
                with src._lock:
                    src._latest = pose
                    src._received_at = time.time()
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):   # 접속 로그로 화면이 어지러워지지 않게
                pass

        return Handler
