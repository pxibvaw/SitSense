"""FSR + HC-SR04 2개(머리/가슴) 동시 데이터 수집 (Raspberry Pi에서 실행)

사용법:
    python3 collect_all.py            # 자세당 180초, 쉬는 시간 10초
    python3 collect_all.py 10 3       # 시험용: 자세당 10초, 쉬는 시간 3초
    python3 collect_all.py --plot posture_data_X.csv   # 센서 없이 이미 있는 CSV로 그래프 PNG만 만들기 (노트북 가능)

필요: spidev, gpiozero (Raspberry Pi OS에 보통 설치돼 있음. 없으면 sudo apt install python3-spidev python3-gpiozero)
      SPI 활성화(raspi-config), HC-SR04 Echo는 분압 회로로 3.3V로 낮춰서 연결.

배치 (앞 -> 뒤):  화면에 이 모양으로 FSR 값이 표시됩니다 (CH5도 표시되지만 CSV에는 저장 안 함)
    CH0   CH5  CH7        (맨 앞)
    CH3        CH6        (중간)
    CH1   CH2  CH4        (맨 뒤)

거리센서 (BCM 번호):
    머리: TRIG 23, ECHO 24
    가슴: TRIG 5,  ECHO 6

구조와 동기화:
    메인(시계)이 0.5초마다 "틱"을 만들어 두 스레드에 똑같이 보냅니다.
    각 틱에는 같은 timestamp / 자세 / 샘플 번호가 들어 있어서, 두 스레드가 같은 순간에 각자 센서를 읽고
    자기 CSV에 한 줄씩 저장합니다. 그래서 두 CSV는 줄 수와 timestamp가 정확히 같습니다.
    FSR 스레드      -> posture_data_<실행시각>.csv
    HC-SR04 스레드  -> distance_data_<실행시각>.csv   (머리, 가슴 거리를 한 줄에 같이 저장)
    자세마다 [쉬는 시간 10초 -> 수집 180초]를 반복합니다.
    쉬는 시간에도 측정해서 저장합니다. phase 열로 구분합니다 (쉬는시간 / 수집).
    쉬는 시간의 label은 "다음에 잡을 자세"입니다. 학습에 쓸 때는 phase가 "수집"인 행만 쓰세요.

CSV 열:
    posture_data_*.csv : timestamp, label, phase, attempt, sample, elapsed_s,
                         CH0(맨 앞 왼쪽), CH1(맨 뒤 왼쪽), CH2(맨 뒤 가운데), CH3(중간 왼쪽),
                         CH4(맨 뒤 오른쪽), CH6(중간 오른쪽), CH7(맨 앞 오른쪽)      (CH5는 저장 안 함)
    distance_data_*.csv: timestamp, phase, attempt, sample, distance_cm(머리), distance_cm(가슴)
        attempt = 같은 자세를 몇 번째 시도했는지 (보통 1, 일시정지 후 다시 하면 2, 3, ...)
        sample = 그 phase 안에서 몇 번째 측정인지 (0부터 시작, phase가 바뀌면 다시 0). 전체 행 번호가 아닙니다.
        distance_cm이 비어 있으면 측정 범위(MAX_DIST_M) 밖이거나 반사가 없어 측정 안 된 것입니다.

두 CSV 합치기 (pandas). timestamp가 정확히 같으므로 그대로 붙습니다:
    fsr  = pd.read_csv("posture_data_X.csv",  encoding="utf-8-sig")
    dist = pd.read_csv("distance_data_X.csv", encoding="utf-8-sig")
    m = pd.merge(fsr, dist, on=["timestamp", "phase", "attempt", "sample"])

일시정지: Ctrl+Z를 누르면 측정이 멈추고, Enter = 지금 자세를 처음(쉬는 시간)부터 다시, q + Enter = 종료.
    멈추기 전까지 잰 데이터도 attempt=1로 CSV에 남습니다. 다시 잰 자세는 attempt가 2, 3...이므로
    학습에는 자세마다 attempt가 가장 큰 행만 쓰세요. (그래프의 평균은 자동으로 마지막 시도만 씁니다)

실시간 화면 (이 파일 하나로 터미널 + 그래프가 같이 나옵니다):
    1) 터미널: 상태 한 줄 + FSR 배치 모양 + 거리(머리/가슴)가 제자리에서 갱신됩니다.
    2) 그래프: 시작하면 "http://<Pi주소>:8000" 이 터미널에 나옵니다. 노트북/폰 브라우저에서 열면 0.5초마다
       FSR 배치(색), FSR 선 그래프, 거리(머리/가슴) 그래프, 자세별 평균 배치(칸 안에 숫자)가 실시간으로 갱신됩니다.
       그래프는 "최근 2분 / 전체"로 바꿔 볼 수 있고, 회색 배경 = 쉬는 시간, 세로선 = 자세 바뀜입니다.
       Pi에 모니터가 연결돼 있어 데스크톱이 떠 있으면 Pi 브라우저에도 자동으로 열립니다.
       (WEB_PORT = 0 이면 끔, AUTO_OPEN_BROWSER = False 면 자동으로 열지 않음)
    3) 끝나면 같은 그래프를 PNG 2장(_timeline.png, _heatmap.png)으로도 저장합니다 (pandas, matplotlib가 있을 때).
       없으면 건너뜁니다: sudo apt install python3-pandas python3-matplotlib   (한글이 깨지면 fonts-nanum)

소리: 쉬는 시간이 시작될 때 "삑" 1번(낮은 소리), 쉬는 시간이 끝나고 수집이 시작될 때 "삑삑" 2번(높은 소리)이 납니다.
    - 터미널 벨(\a): 터미널 프로그램이 소리를 지원하면 납니다 (SSH면 노트북 터미널에서 남). BEEP_TERMINAL = False 로 끔.
    - 브라우저: 그래프 페이지의 "소리 켜기" 버튼을 한 번 누르면 노트북/폰에서도 같은 소리가 납니다 (브라우저 규칙상 클릭 필요).
    Pi에 스피커/부저를 따로 연결했다면 알려주세요 (그쪽으로도 울리게 바꿀 수 있습니다).

그래프 저장: 자세 하나가 끝날 때마다(그리고 일시정지할 때) 그래프 PNG를 같은 이름으로 덮어써서 저장하고,
    끝날 때 한 번 더 저장합니다. 그래서 도중에 Pi가 꺼져도 마지막 자세까지의 그래프가 남습니다.
    (AUTOSAVE_GRAPH = False 면 끝날 때만 저장)

timestamp는 Pi의 시계를 따릅니다. 실행 전에 Pi의 날짜·시간이 맞는지 확인하세요 (date 명령).
Ctrl+C로 끝내도 그때까지 저장된 데이터는 파일에 남아 있습니다 (한 줄씩 바로 저장).
"""
import csv
import errno
import json
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
import warnings
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# spidev, gpiozero는 센서를 읽는 main()에서 불러옵니다 (--plot은 센서 없는 PC에서도 되게)

# ---------------- 설정 ----------------
DURATION_SEC = 180    # 자세당 수집 시간(초)
REST_SEC = 10         # 자세가 바뀔 때 쉬는 시간(초)
INTERVAL = 0.5        # FSR, HC-SR04 공통 수집 간격(초)
SPEED_HZ = 500_000    # SPI 속도 (3.3V에서 안전한 속도)

# HC-SR04 핀 (BCM 번호). 실제로 연결한 핀에 맞게 고치세요.
HEAD_TRIG_PIN = 23    # 머리: 물리 핀 16
HEAD_ECHO_PIN = 24    # 머리: 물리 핀 18 (분압 회로를 거쳐서 연결)
CHEST_TRIG_PIN = 5    # 가슴: 물리 핀 29  (5=TRIG, 6=ECHO로 가정. 반대면 서로 바꾸세요)
CHEST_ECHO_PIN = 6    # 가슴: 물리 핀 31  (분압 회로를 거쳐서 연결)
MAX_DIST_M = 2.0      # 이 거리(m) 이상이면 "측정 안 됨"(빈 칸)
QUEUE_LEN = 5         # gpiozero가 최근 N번 측정값의 중앙값을 냄 (노이즈 줄이기)

# 실시간 웹 그래프
WEB_PORT = 8000       # 0이면 실시간 웹 그래프를 끔. 이 포트가 사용 중이면 8001, 8002... 순서로 빈 포트를 씀
AUTO_OPEN_BROWSER = True   # Pi에 데스크톱이 떠 있으면 Pi 브라우저에도 자동으로 열기

BEEP_TERMINAL = True       # 쉬는 시간 시작/끝에 터미널 벨(\a)을 울림
AUTOSAVE_GRAPH = True      # 자세 하나가 끝날 때마다 그래프 PNG를 덮어써서 저장

REST = "쉬는시간"
COLLECT = "수집"

PLOT_ONLY = len(sys.argv) >= 2 and sys.argv[1] == "--plot"
if not PLOT_ONLY:
    if len(sys.argv) >= 2:
        DURATION_SEC = float(sys.argv[1])
    if len(sys.argv) >= 3:
        REST_SEC = float(sys.argv[2])

LABELS = [
    "바른자세(앞)",
    "바른자세(중)",
    "바른자세(뒤)",
    "왼쪽편향(앞)",
    "왼쪽편향(뒤)",
    "오른쪽편향(앞)",
    "오른쪽편향(뒤)",
    "왼다리꼬기(앞)",
    "왼다리꼬기(중)",
    "왼다리꼬기(뒤)",
    "오른다리꼬기(앞)",
    "오른다리꼬기(중)",
    "오른다리꼬기(뒤)",
]

# 읽을 FSR 채널과 위치 이름
POSITIONS = {
    0: "맨 앞 왼쪽",
    5: "맨 앞 가운데",
    7: "맨 앞 오른쪽",
    3: "중간 왼쪽",
    6: "중간 오른쪽",
    1: "맨 뒤 왼쪽",
    2: "맨 뒤 가운데",
    4: "맨 뒤 오른쪽",
}
CHANNELS = list(POSITIONS)                                   # 화면에 표시할 채널 (8개)
CSV_CHANNELS = sorted(ch for ch in CHANNELS if ch != 5)      # CSV에 저장할 채널 (CH5 제외)


class Shared:
    """메인과 스레드들이 같이 쓰는 상태."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.phase = REST
        self.fsr = None          # 가장 최근 FSR 값 (채널 -> 값)
        self.head_cm = None      # 가장 최근 머리 거리(cm), 없으면 None
        self.chest_cm = None     # 가장 최근 가슴 거리(cm), 없으면 None
        self.errors = []
        self.points = []         # 실시간 그래프용 측정값 전체 (계속 쌓음)
        self.status = ""          # 실시간 그래프에 보여줄 상태 문장
        self.pause_req = False    # Ctrl+Z를 누르면 True (신호 처리기에서는 락을 쓰지 않음)
        self.t0 = time.monotonic()


class Tick:
    """0.5초마다 두 스레드에 똑같이 전달되는 한 번의 측정 지시."""

    def __init__(self, ts, label, phase, attempt, sample, elapsed):
        self.ts = ts                # 두 CSV에 똑같이 들어가는 timestamp
        self.label = label
        self.phase = phase          # REST(쉬는시간) 또는 COLLECT(수집)
        self.attempt = attempt      # 같은 자세를 몇 번째 시도하는지 (일시정지 후 다시 하면 2, 3, ...)
        self.sample = sample        # 이 구간 안에서 몇 번째인지 (0부터)
        self.elapsed = elapsed      # 구간 시작 후 경과 시간(초)


def read_adc(spi, ch):
    """MCP3208 한 채널을 읽어 0~4095를 돌려준다."""
    r = spi.xfer2([0x06 | (ch >> 2), (ch & 3) << 6, 0])
    return ((r[1] & 0x0F) << 8) | r[2]


def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def to_cm(d):
    """gpiozero 거리(m) -> cm. 범위 밖이면 None."""
    return None if d >= MAX_DIST_M - 1e-6 else round(d * 100, 1)


def fsr_worker(sh, q, spi, path, counts):
    """틱이 올 때마다 FSR를 읽고 CSV에 한 줄 저장한다."""
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "label", "phase", "attempt", "sample", "elapsed_s"]
                       + [f"CH{ch}({POSITIONS[ch]})" for ch in CSV_CHANNELS])
            f.flush()
            while True:
                tick = q.get()
                if tick is None:
                    break
                vals = {ch: read_adc(spi, ch) for ch in CHANNELS}
                with sh.lock:
                    sh.fsr = vals
                w.writerow([tick.ts, tick.label, tick.phase, tick.attempt, tick.sample, tick.elapsed]
                           + [vals[ch] for ch in CSV_CHANNELS])
                f.flush()
                key = (tick.label, tick.phase, tick.attempt)
                counts[key] = counts.get(key, 0) + 1
    except Exception as e:
        sh.errors.append(f"FSR 스레드 오류: {e!r}")
        sh.stop.set()


def dist_worker(sh, q, head, chest, path, counts):
    """틱이 올 때마다 HC-SR04 두 개(머리, 가슴)의 거리를 읽고 CSV에 한 줄 저장한다."""
    try:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "phase", "attempt", "sample", "distance_cm(머리)", "distance_cm(가슴)"])
            f.flush()
            while True:
                tick = q.get()
                if tick is None:
                    break
                head_cm = to_cm(head.distance)       # 미터 단위
                chest_cm = to_cm(chest.distance)
                with sh.lock:
                    sh.head_cm = head_cm
                    sh.chest_cm = chest_cm
                w.writerow([tick.ts, tick.phase, tick.attempt, tick.sample, head_cm, chest_cm])
                f.flush()
                key = (tick.label, tick.phase, tick.attempt)
                counts[key] = counts.get(key, 0) + 1
    except Exception as e:
        sh.errors.append(f"HC-SR04 스레드 오류: {e!r}")
        sh.stop.set()


drawn = False


def show(sh, status):
    """상태 + FSR 배치 모양 3줄 + 거리 두 줄을 제자리에서 갱신한다."""
    global drawn
    with sh.lock:
        z = sh.fsr
        head_cm = sh.head_cm
        chest_cm = sh.chest_cm
    if z is None:
        z = {ch: 0 for ch in CHANNELS}

    def fmt(name, cm):
        return f"{name}: 측정 안 됨 (범위 밖/반사 없음)" if cm is None else f"{name}: {cm:6.1f} cm"

    rows = [
        status,
        f"{z[0]:4d}  {z[5]:4d}  {z[7]:4d}",
        f"{z[3]:4d}        {z[6]:4d}",
        f"{z[1]:4d}  {z[2]:4d}  {z[4]:4d}",
        fmt("거리(머리)", head_cm),
        fmt("거리(가슴)", chest_cm),
    ]
    if drawn:
        print(f"\033[{len(rows)}A", end="")
    for r in rows:
        print("\033[K" + r)
    drawn = True


LAYOUT = [[0, 5, 7], [3, None, 6], [1, 2, 4]]    # 웹/그래프에 보여줄 FSR 배치 (앞 -> 뒤)

LIVE_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>실시간 센서</title>
<style>
body{font-family:system-ui,"Noto Sans KR","Malgun Gothic",sans-serif;margin:16px;background:#fafafa;color:#222}
#status{font-size:18px;font-weight:600;margin-bottom:12px}
#status.rest{color:#888}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
#grid{display:grid;grid-template-columns:repeat(3,92px);gap:6px}
.cell{height:72px;border:1px solid #ccc;border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:11px;background:#fff}
.cell b{font-size:18px}
.cell.skip{border-style:dashed}
.cell.empty{border:none;background:none}
.panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}
canvas{width:100%;display:block}
#cf{height:280px} #cd{height:200px}
.legend span{margin-right:10px;font-size:12px;white-space:nowrap}
.legend i{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}
.note{font-size:12px;color:#888}
.bar{display:flex;justify-content:space-between;align-items:center}
h3{margin:18px 0 6px;font-size:15px}
#cards{display:flex;flex-wrap:wrap;gap:10px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:6px 8px;width:168px}
.card.cur{border:2px solid #2a7ae2}
.ct{font-size:12px;font-weight:600;margin-bottom:4px}
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:2px}
.mc{height:34px;display:flex;align-items:center;justify-content:center;font-size:11px;border:1px solid #eee;border-radius:3px}
.mc.skip{border-style:dashed;border-color:#bbb}
.mc.e{border:none}
.cd{font-size:11px;color:#666;margin-top:4px}
</style></head><body>
<div id="status">연결 중...</div>
<p style="margin:0 0 10px"><button id="snd">🔔 소리 켜기</button>
  <span class="note">쉬는 시간 시작 = 낮은 삑 1번, 쉬는 시간 끝 = 높은 삑 2번 (브라우저 규칙상 버튼을 한 번 눌러야 소리가 납니다)</span></p>
<div class="row">
  <div>
    <div id="grid"></div>
    <p id="dist" style="font-size:15px"></p>
    <p class="note">위 = 앞, 아래 = 뒤. 점선 칸 = CSV에 저장 안 하는 채널</p>
  </div>
  <div style="flex:1;min-width:380px">
    <div class="panel">
      <div class="bar"><span>FSR (회색 = 쉬는 시간, 세로선 = 자세 바뀜)</span>
        <select id="win"><option value="120">최근 2분</option><option value="0">전체</option></select></div>
      <canvas id="cf"></canvas><div class="legend" id="lf"></div></div>
    <div class="panel" style="margin-top:12px"><div>거리 (cm)</div><canvas id="cd"></canvas><div class="legend" id="ld"></div></div>
  </div>
</div>
<h3>자세별 FSR 평균 (수집 구간, 다시 잰 자세는 마지막 시도) - 파란 테두리 = 지금 자세</h3>
<div id="cards"></div>
<script>
const COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#17becf"];
const DCOL=["#1f77b4","#ff7f0e"];
const st=document.getElementById("status"), winEl=document.getElementById("win");
let pts=[], agg={}, meta=null, legendDone=false;
let audio=null, soundOn=false, firstLoad=true;

function beep(freq, times){
  if(!soundOn||!audio) return;
  for(let i=0;i<times;i++){
    const t0=audio.currentTime+i*0.22, o=audio.createOscillator(), g=audio.createGain();
    o.type="sine"; o.frequency.value=freq; o.connect(g); g.connect(audio.destination);
    g.gain.setValueAtTime(0.0001,t0);
    g.gain.exponentialRampToValueAtTime(0.3,t0+0.01);
    g.gain.exponentialRampToValueAtTime(0.0001,t0+0.18);
    o.start(t0); o.stop(t0+0.2);
  }
}
document.getElementById("snd").onclick=function(){
  if(!audio) audio=new (window.AudioContext||window.webkitAudioContext)();
  audio.resume(); soundOn=!soundOn;
  this.textContent=soundOn?"🔔 소리 켜짐 (누르면 끔)":"🔕 소리 꺼짐 (누르면 켬)";
  if(soundOn) beep(800,1);
};

// 새로 들어온 점들에서 "쉬는 시간 시작/끝"을 찾는다. 반환: 0 없음, 1 쉬는 시간 시작, 2 쉬는 시간 끝
function findEvent(prev, arr){
  let ev=0;
  arr.forEach(p=>{
    if(p.p===0 && (prev===null || prev.p===1 || prev.l!==p.l || prev.a!==p.a)) ev=1;
    else if(p.p===1 && prev!==null && prev.p===0) ev=2;
    prev=p;
  });
  return ev;
}

function fmtT(t){return Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0");}

function draw(cv, series, o){
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr;
  const g=cv.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,w,h);
  const L=44,R=8,T=o.marks?24:8,B=20;
  let lo=Infinity, hi=-Infinity;
  series.forEach(s=>s.pts.forEach(q=>{ if(q[1]!==null){ if(q[1]<lo) lo=q[1]; if(q[1]>hi) hi=q[1]; } }));
  if(lo===Infinity){ lo=0; hi=1; }
  let y0=o.fixedMin!==undefined?o.fixedMin:lo, y1=hi;
  if(y1-y0<1) y1=y0+1;
  const pad=(y1-y0)*0.08; y1+=pad; if(o.fixedMin===undefined) y0-=pad;
  const x0=o.x0, x1=Math.max(o.x1,o.x0+1);
  const X=x=>L+(x-x0)/(x1-x0)*(w-L-R), Y=y=>T+(1-(y-y0)/(y1-y0))*(h-T-B);
  g.fillStyle="rgba(128,128,128,0.18)";
  (o.spans||[]).forEach(([a,b])=>{ const xa=Math.max(a,x0), xb=Math.min(b,x1); if(xb>xa) g.fillRect(X(xa),T,X(xb)-X(xa),h-T-B); });
  g.strokeStyle="#eee"; g.fillStyle="#666"; g.font="11px sans-serif"; g.lineWidth=1;
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4, y=Y(v); g.beginPath(); g.moveTo(L,y); g.lineTo(w-R,y); g.stroke(); g.fillText(v.toFixed(0),4,y+4);}
  g.fillText(fmtT(x0),L,h-4); g.fillText(fmtT(x1),w-R-32,h-4);
  if(o.marks){
    g.font="10px sans-serif"; let lastEnd=-1e9;
    o.marks.forEach(([t,txt])=>{
      if(t<x0||t>x1) return;
      const x=X(t); g.strokeStyle="#999"; g.beginPath(); g.moveTo(x,T-4); g.lineTo(x,h-B); g.stroke();
      if(x>lastEnd){ g.fillStyle="#444"; g.fillText(txt,x+3,T-8); lastEnd=x+3+g.measureText(txt).width+6; }
    });
  }
  series.forEach(s=>{
    g.strokeStyle=s.color; g.lineWidth=1.5; g.beginPath(); let pen=false;
    s.pts.forEach(q=>{
      if(q[1]===null){pen=false; return;}
      const x=X(q[0]), y=Y(q[1]);
      if(pen) g.lineTo(x,y); else {g.moveTo(x,y); pen=true;}
    });
    g.stroke();
  });
}

function legend(el, items){
  el.innerHTML=items.map(it=>`<span><i style="background:${it[1]}"></i>${it[0]}</span>`).join("");
}

// 선 하나의 점 목록. 시간 간격이 3초 넘게 벌어지면(일시정지) 선을 끊는다
function line(vis, getY){
  const out=[]; let prev=null;
  vis.forEach(p=>{
    if(prev!==null && p.t-prev.t>3) out.push([prev.t+0.25,null]);
    out.push([p.t,getY(p)]); prev=p;
  });
  return out;
}

function spansMarks(vis){
  const spans=[], marks=[]; let rs=null, prev=null, key="";
  vis.forEach(p=>{
    const k=p.l+"/"+p.a;
    if(k!==key){ marks.push([p.t, meta.labels[p.l]+(p.a>1?` (재시도 ${p.a-1})`:"")]); key=k; }
    if(p.p===0){ if(rs===null) rs=p.t; }
    else if(rs!==null){ spans.push([rs,p.t]); rs=null; }
    prev=p;
  });
  if(rs!==null && prev) spans.push([rs,prev.t+0.5]);
  return {spans,marks};
}

function addPoints(arr){
  arr.forEach(p=>{
    pts.push(p);
    if(p.p===1){
      const A=(agg[p.l]=agg[p.l]||{});
      const x=(A[p.a]=A[p.a]||{n:0,s:new Array(8).fill(0),hs:0,hn:0,cs:0,cn:0});
      x.n++; p.f.forEach((v,i)=>x.s[i]+=v);
      if(p.h!==null){x.hs+=p.h; x.hn++;}
      if(p.c!==null){x.cs+=p.c; x.cn++;}
    }
  });
}

function renderCards(){
  const cur=pts.length?pts[pts.length-1].l:-1, rows=[]; let vmax=1;
  Object.keys(agg).map(Number).sort((a,b)=>a-b).forEach(l=>{
    const A=agg[l], a=Math.max(...Object.keys(A).map(Number)), x=A[a];
    const m=x.s.map(v=>v/x.n); m.forEach(v=>{ if(v>vmax) vmax=v; });
    rows.push({l,a,m,h:x.hn?x.hs/x.hn:null,c:x.cn?x.cs/x.cn:null});
  });
  document.getElementById("cards").innerHTML=rows.map(r=>{
    const cells=meta.layout.map(row=>row.map(ch=>{
      if(ch===null) return '<div class="mc e"></div>';
      const v=r.m[ch], skip=meta.csv_skip.includes(ch)?" skip":"";
      return `<div class="mc${skip}" style="background:rgba(220,60,40,${Math.min(1,v/vmax).toFixed(2)})">${v.toFixed(0)}</div>`;
    }).join("")).join("");
    const d=(r.h===null?"-":r.h.toFixed(0))+" / "+(r.c===null?"-":r.c.toFixed(0))+" cm";
    return `<div class="card${r.l===cur?" cur":""}"><div class="ct">${meta.labels[r.l]}${r.a>1?` <small>(시도 ${r.a})</small>`:""}</div><div class="mg">${cells}</div><div class="cd">머리/가슴 ${d}</div></div>`;
  }).join("");
}

function render(){
  const d=meta;
  st.textContent=d.status||"대기 중 (Pi에서 Enter를 누르면 시작)";
  st.className=(d.phase==="쉬는시간")?"rest":"";
  const last=pts.length?pts[pts.length-1]:null;

  const maxNow=last?Math.max(500,...last.f):500;
  document.getElementById("grid").innerHTML=d.layout.map(row=>row.map(ch=>{
    if(ch===null) return '<div class="cell empty"></div>';
    const v=last?last.f[ch]:null, a=v===null?0:Math.min(1,v/maxNow);
    const skip=d.csv_skip.includes(ch)?" skip":"";
    return `<div class="cell${skip}" style="background:rgba(220,60,40,${a.toFixed(2)})">${d.channels[ch]}<b>${v===null?"-":v}</b>CH${ch}</div>`;
  }).join("")).join("");
  const fmt=(name,v)=>`${name} `+(v===null||v===undefined?"측정 안 됨":v.toFixed(1)+" cm");
  document.getElementById("dist").textContent=last?fmt("머리",last.h)+"  /  "+fmt("가슴",last.c):"";

  if(pts.length){
    const win=Number(winEl.value), tmax=last.t, tmin=win>0?tmax-win:pts[0].t;
    const vis=win>0?pts.filter(p=>p.t>=tmin):pts;
    const sm=spansMarks(vis), x0=vis[0].t, x1=tmax;
    const fs=[]; for(let ch=0;ch<8;ch++) fs.push({color:COLORS[ch], pts:line(vis,p=>p.f[ch])});
    draw(document.getElementById("cf"), fs, {fixedMin:0, x0, x1, spans:sm.spans, marks:sm.marks});
    draw(document.getElementById("cd"), [
      {color:DCOL[0], pts:line(vis,p=>p.h)}, {color:DCOL[1], pts:line(vis,p=>p.c)}],
      {x0, x1, spans:sm.spans});
  }
  if(!legendDone){
    legend(document.getElementById("lf"), [...Array(8).keys()].map(ch=>[`CH${ch} ${d.channels[ch]}`,COLORS[ch]]));
    legend(document.getElementById("ld"), [["머리",DCOL[0]],["가슴",DCOL[1]]]);
    legendDone=true;
  }
  renderCards();
}

async function refresh(){
  let d;
  try{ d=await (await fetch("/data?since="+pts.length,{cache:"no-store"})).json(); }
  catch(e){ st.textContent="연결 끊김 (Pi가 켜져 있는지 확인)"; return; }
  if(d.reset){ pts=[]; agg={}; firstLoad=true; }
  const ev=findEvent(pts.length?pts[pts.length-1]:null, d.points);
  addPoints(d.points);
  if(!firstLoad){ if(ev===1) beep(500,1); else if(ev===2) beep(1000,2); }   // 처음 불러올 때의 옛 기록에는 울리지 않음
  firstLoad=false;
  meta=d; render();
}
winEl.addEventListener("change", ()=>{ if(meta) render(); });
refresh(); setInterval(refresh, 500);
</script></body></html>
"""


def make_handler(sh):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/data":
                try:
                    since = int(parse_qs(url.query).get("since", ["0"])[0])
                except ValueError:
                    since = 0
                with sh.lock:
                    total = len(sh.points)
                    reset = since > total or since < 0      # 서버가 다시 시작된 경우 처음부터 보냄
                    if reset:
                        since = 0
                    data = {
                        "status": sh.status,
                        "phase": sh.phase,
                        "labels": LABELS,
                        "channels": {str(ch): name for ch, name in POSITIONS.items()},
                        "layout": LAYOUT,
                        "csv_skip": [ch for ch in CHANNELS if ch not in CSV_CHANNELS],
                        "reset": reset,
                        "points": sh.points[since:],        # 새로 생긴 점만 보냄
                    }
                body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            elif url.path in ("/", "/index.html"):
                body = LIVE_HTML.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            elif url.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):    # 접속 로그가 화면 갱신을 어지럽히지 않게
            pass

    return Handler


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))     # 실제로 패킷을 보내지는 않음
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname() + ".local"


def start_web(sh):
    """실시간 웹 그래프 서버를 켠다. 포트가 사용 중이면 다음 포트로 넘어간다. 실패해도 수집은 계속한다."""
    if not WEB_PORT:
        return None
    server = None
    port = WEB_PORT
    for port in range(WEB_PORT, WEB_PORT + 10):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(sh))
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                print(f"실시간 웹 그래프를 켤 수 없습니다 ({e}). 수집은 계속합니다.")
                return None
    if server is None:
        print(f"실시간 웹 그래프를 켤 수 없습니다 (포트 {WEB_PORT}~{WEB_PORT + 9} 모두 사용 중). 수집은 계속합니다.")
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="web").start()
    if port != WEB_PORT:
        print(f"(포트 {WEB_PORT}은 다른 프로그램이 쓰고 있어서 {port}로 켰습니다)")
    print(f"실시간 그래프: http://{local_ip()}:{port}  (같은 네트워크의 노트북 브라우저에서 열기)")
    if AUTO_OPEN_BROWSER and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        threading.Thread(target=webbrowser.open, args=(f"http://localhost:{port}",), daemon=True).start()
        print("  (Pi 화면에도 브라우저로 여는 중)")
    return server


def live_update(sh, status, phase, label_idx, attempt):
    """실시간 그래프에 보낼 측정값을 쌓는다."""
    with sh.lock:
        if sh.fsr is None:
            return
        sh.status = status
        sh.phase = phase
        sh.points.append({
            "t": round(time.monotonic() - sh.t0, 2),
            "f": [sh.fsr[ch] for ch in range(8)],
            "h": sh.head_cm,
            "c": sh.chest_cm,
            "l": label_idx,                     # LABELS 안에서 몇 번째 자세인지
            "p": 1 if phase == COLLECT else 0,  # 1 = 수집, 0 = 쉬는 시간
            "a": attempt,
        })


# ---------------- 그래프 PNG (수집이 끝날 때 자동, 또는 --plot 으로 따로) ----------------
GRID = {ch: (r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch is not None}
pd = np = plt = None


def load_plot_libs():
    """pandas, matplotlib는 그래프를 만들 때만 불러온다 (없으면 ImportError)."""
    global pd, np, plt
    import matplotlib
    matplotlib.use("Agg")                    # 화면 없이 파일로만 저장
    import matplotlib.pyplot as _plt
    import numpy as _np
    import pandas as _pd
    from matplotlib import font_manager
    pd, np, plt = _pd, _np, _plt
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
        if n in names:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams["axes.unicode_minus"] = False


def ch_of(col):
    m = re.match(r"CH(\d)", col)
    return int(m.group(1)) if m else None


def plot_load(fsr_path, dist_path):
    fsr = pd.read_csv(fsr_path, encoding="utf-8-sig", parse_dates=["timestamp"])
    if fsr.empty:
        raise ValueError("FSR 파일에 데이터가 없습니다")
    if "phase" not in fsr.columns:
        fsr["phase"] = COLLECT
    if "attempt" not in fsr.columns:         # 예전 파일에는 attempt 열이 없다
        fsr["attempt"] = 1
    dist = None
    if dist_path and Path(dist_path).exists():
        dist = pd.read_csv(dist_path, encoding="utf-8-sig", parse_dates=["timestamp"])
        if dist.empty:
            dist = None
    t0 = fsr["timestamp"].iloc[0]
    fsr["t"] = (fsr["timestamp"] - t0).dt.total_seconds()
    if dist is not None:
        dist["t"] = (dist["timestamp"] - t0).dt.total_seconds()
    return fsr, dist


def with_breaks(df, dt):
    """일시정지한 구간(시간 간격이 큰 곳)에서 선이 이어지지 않게 빈 행을 끼워 넣는다."""
    gap = df["t"].diff() > max(5 * dt, 3)
    if not gap.any():
        return df
    br = pd.DataFrame({"t": df.loc[gap, "t"] - dt / 2}, index=df.index[gap] - 0.5)
    return pd.concat([df, br]).sort_index()


def plot_timeline(fsr, dist, out):
    dt = fsr["t"].diff().median()
    dt = 0.5 if pd.isna(dt) else dt
    fsr_p = with_breaks(fsr, dt)
    dist_p = with_breaks(dist, dt) if dist is not None else None
    ch_cols = [c for c in fsr.columns if ch_of(c) is not None]
    n_rows = 2 if dist is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(24, 4.5 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for c in ch_cols:
        axes[0].plot(fsr_p["t"], fsr_p[c], lw=0.8, label=c)
    axes[0].set_ylabel("FSR (0~4095)")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))

    if dist is not None:
        for c in [c for c in dist.columns if c.startswith("distance_cm")]:
            axes[1].plot(dist_p["t"], dist_p[c], lw=0.8, label=c)
        axes[1].set_ylabel("거리 (cm)")
        axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    axes[-1].set_xlabel("시간 (초)")

    # 쉬는 시간 회색 배경
    for ax in axes:
        is_rest = (fsr["phase"] == REST).to_numpy()
        t = fsr["t"].to_numpy()
        start = None
        for i, r in enumerate(is_rest):
            if r and start is None:
                start = t[i]
            if (not r or i == len(is_rest) - 1) and start is not None:
                end = t[i] if r else t[i - 1] + dt
                ax.axvspan(start, end, color="gray", alpha=0.18, lw=0)
                start = None

    # 자세 구간 구분선 + 이름
    for (label, attempt), g in fsr.groupby(["label", "attempt"], sort=False):
        x0 = g["t"].min()
        text = label if attempt == 1 else f"{label} (재시도 {attempt - 1})"
        for ax in axes:
            ax.axvline(x0, color="k", lw=0.6, alpha=0.5)
        # 자세 이름은 그래프 위쪽 바깥에 써서 선과 겹치지 않게 한다
        axes[0].text(x0 + 1, 1.01, text, transform=axes[0].get_xaxis_transform(),
                     va="bottom", ha="left", fontsize=8)

    fig.suptitle("자세별 센서값 (회색 = 쉬는 시간)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_heatmap(fsr, dist, out):
    ch_cols = {ch_of(c): c for c in fsr.columns if ch_of(c) is not None}
    last = fsr.groupby("label")["attempt"].transform("max")      # 다시 잰 자세는 마지막 시도만 쓴다
    col = fsr[(fsr["phase"] == COLLECT) & (fsr["attempt"] == last)]
    if col.empty:
        return False
    labels = list(dict.fromkeys(col["label"]))
    means = col.groupby("label")[list(ch_cols.values())].mean()
    vmax = max(1.0, float(np.nanmax(means.to_numpy())))

    dist_mean = None
    if dist is not None:
        # 거리 행에는 label이 없으므로 timestamp 기준으로 FSR 라벨을 붙인다 (같은 구간만)
        dd = dist[dist["timestamp"].isin(col["timestamp"])].merge(col[["timestamp", "label"]], on="timestamp", how="left")
        dist_mean = dd.groupby("label")[[c for c in dd.columns if c.startswith("distance_cm")]].mean()

    ncols = 5
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    im = None
    for ax in axes.flat:
        ax.axis("off")
    for ax, label in zip(axes.flat, labels):
        ax.axis("on")
        grid = np.full((3, 3), np.nan)
        for ch, c in ch_cols.items():
            r, k = GRID[ch]
            grid[r, k] = means.loc[label, c]
        im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=vmax)
        for (r, k), v in np.ndenumerate(grid):
            if not np.isnan(v):
                ax.text(k, r, f"{v:.0f}", ha="center", va="center", fontsize=9)
        title = label
        if dist_mean is not None and label in dist_mean.index:
            vals = dist_mean.loc[label]
            parts = [f"{c.split('(')[-1].rstrip(')')} {v:.0f}cm" for c, v in vals.items() if not np.isnan(v)]
            if parts:
                title += "\n" + " / ".join(parts)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("자세별 FSR 평균 (위 = 앞, 아래 = 뒤 / 빈 칸 = 센서 없음 또는 저장 안 함)")
    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="평균값 (0~4095)")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def plot_run(fsr_path, dist_path=None):
    """그래프 PNG 2장을 만들고 저장한 파일 이름 목록을 돌려준다."""
    load_plot_libs()
    fsr_path = Path(fsr_path)
    if dist_path is None:
        guess = fsr_path.with_name(fsr_path.name.replace("posture_data", "distance_data"))
        dist_path = guess if guess.exists() else None
    fsr, dist = plot_load(fsr_path, dist_path)
    stem = fsr_path.with_suffix("")
    outs = []
    out1 = f"{stem}_timeline.png"
    plot_timeline(fsr, dist, out1)
    outs.append(out1)
    out2 = f"{stem}_heatmap.png"
    if plot_heatmap(fsr, dist, out2):
        outs.append(out2)
    return outs


def plot_cli():
    args = sys.argv[2:]
    if not args:
        print("사용법: python3 collect_all.py --plot posture_data_X.csv [distance_data_X.csv]")
        return
    for p in plot_run(args[0], args[1] if len(args) > 1 else None):
        print("저장:", p)


def beep_terminal(n):
    """터미널 벨을 n번 울린다 (화면 갱신을 막지 않게 따로 스레드로)."""
    def run():
        for _ in range(n):
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.2)
    threading.Thread(target=run, daemon=True).start()


class GraphSaver:
    """자세가 끝날 때마다 그래프 PNG를 덮어써서 저장한다 (측정을 방해하지 않게 따로 스레드로)."""

    def __init__(self, fsr_path, dist_path):
        self.fsr_path, self.dist_path = fsr_path, dist_path
        self.busy = threading.Lock()
        self.disabled = False       # pandas/matplotlib가 없으면 더 시도하지 않음
        self.error = None

    def kick(self):
        if self.disabled or not self.busy.acquire(blocking=False):
            return                  # 이전 저장이 아직 진행 중이면 이번엔 건너뜀

        def run():
            try:
                plot_run(self.fsr_path, self.dist_path)
            except ImportError as e:
                self.disabled = True
                self.error = repr(e)
            except Exception as e:   # 쓰는 중인 CSV를 읽다 실패하는 정도는 다음 기회에 다시
                self.error = repr(e)
            finally:
                self.busy.release()

        threading.Thread(target=run, daemon=True, name="graph-save").start()

    def wait(self):
        """진행 중인 저장이 끝날 때까지 기다린다."""
        with self.busy:
            pass


def posture_schedule():
    """자세 하나의 (phase, 샘플 번호, 남은 시간) 을 0.5초 단위로 차례로 내보낸다 (쉬는 시간 -> 수집)."""
    for phase, seconds in ((REST, REST_SEC), (COLLECT, DURATION_SEC)):
        n = int(round(seconds / INTERVAL))
        for i in range(n):
            yield phase, i, seconds - i * INTERVAL


def reset_display():
    """다음 show()가 화면을 새로 그리게 한다 (프롬프트를 출력한 뒤에 필요)."""
    global drawn
    drawn = False


def pause_prompt(sh, label):
    """일시정지 상태에서 사용자에게 다시 할지 끝낼지 묻는다. 'q'면 종료."""
    with sh.lock:
        sh.status = f"일시정지 - {label} (터미널에서 Enter = 이 자세를 처음부터 다시, q = 종료)"
        sh.phase = REST
    print(f"\n\n일시정지됨 (현재 자세: {label})")
    print("  Enter     : 이 자세를 처음(쉬는 시간)부터 다시 측정")
    print("  q + Enter : 종료 (지금까지 저장된 데이터는 파일에 남아 있습니다)")
    try:
        ans = input("> ").strip().lower()
    except EOFError:
        ans = "q"
    sh.pause_req = False          # 일시정지 중에 또 누른 Ctrl+Z는 무시
    reset_display()
    print()
    return ans


def main():
    import spidev
    from gpiozero import DistanceSensor, DistanceSensorNoEcho
    warnings.simplefilter("ignore", DistanceSensorNoEcho)        # 반사 없음 경고가 화면을 어지럽히지 않게

    sh = Shared()
    spi = spidev.SpiDev()
    spi.open(0, 0)       # SPI0, CE0 (Pi 24번 핀 = CS)
    spi.max_speed_hz = SPEED_HZ
    spi.mode = 0
    head = DistanceSensor(echo=HEAD_ECHO_PIN, trigger=HEAD_TRIG_PIN, max_distance=MAX_DIST_M, queue_len=QUEUE_LEN)
    chest = DistanceSensor(echo=CHEST_ECHO_PIN, trigger=CHEST_TRIG_PIN, max_distance=MAX_DIST_M, queue_len=QUEUE_LEN)

    threads = []
    queues = []
    fsr_counts, dist_counts = {}, {}
    attempts = {}                    # 자세 -> 지금까지 시도한 횟수
    saver = None
    fsr_path = dist_path = None
    web = None
    if hasattr(signal, "SIGTSTP"):   # Ctrl+Z를 '멈춤'이 아니라 '일시정지 요청'으로 받는다
        signal.signal(signal.SIGTSTP, lambda signum, frame: setattr(sh, "pause_req", True))
    try:
        print(f"자세 {len(LABELS)}개 x {DURATION_SEC:g}초, 자세 사이 쉬는 시간 {REST_SEC:g}초")
        print(f"FSR, 거리센서(머리/가슴) 모두 {INTERVAL}초 간격으로 같은 순간에 측정 (쉬는 시간에도 저장, phase 열로 구분)")
        print("자세 순서:", " -> ".join(LABELS))
        web = start_web(sh)
        print("Ctrl+Z = 일시정지 (그 뒤 Enter = 지금 자세를 처음부터 다시, q = 종료)")
        input("준비되면 Enter를 누르세요 (첫 자세 전에도 쉬는 시간이 있습니다)... ")
        sh.pause_req = False
        print()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        fsr_path = f"posture_data_{run_id}.csv"
        dist_path = f"distance_data_{run_id}.csv"
        q_fsr, q_dist = queue.Queue(), queue.Queue()
        queues = [q_fsr, q_dist]
        threads = [
            threading.Thread(target=fsr_worker, args=(sh, q_fsr, spi, fsr_path, fsr_counts), name="fsr"),
            threading.Thread(target=dist_worker, args=(sh, q_dist, head, chest, dist_path, dist_counts), name="dist"),
        ]
        for t in threads:
            t.start()
        if AUTOSAVE_GRAPH:
            saver = GraphSaver(fsr_path, dist_path)

        total_labels = len(LABELS)
        completed = True
        k = 0
        while k < total_labels:
            label = LABELS[k]
            attempt = attempts.get(label, 0) + 1
            attempts[label] = attempt
            next_t = time.monotonic()
            paused = False
            for phase, i, left in posture_schedule():
                wait = next_t - time.monotonic()
                if wait > 0:
                    sh.stop.wait(wait)
                if sh.stop.is_set():
                    completed = False
                    break
                if sh.pause_req:
                    paused = True
                    break
                tick = Tick(now_iso(), label, phase, attempt, i, round(i * INTERVAL, 2))
                q_fsr.put(tick)
                q_dist.put(tick)
                next_t += INTERVAL
                if BEEP_TERMINAL and i == 0:   # 쉬는 시간 시작 = 1번, 쉬는 시간 끝(수집 시작) = 2번
                    beep_terminal(1 if phase == REST else 2)
                sh.stop.wait(0.2)            # 두 스레드가 읽을 시간을 주고 화면 갱신
                retry = f" (재시도 {attempt - 1}회)" if attempt > 1 else ""
                if phase == COLLECT:
                    status = f"[{k + 1}/{total_labels}] 수집 중 - {label}{retry} (남은 시간 {left:5.1f}초)"
                else:
                    status = f"[{k + 1}/{total_labels}] 쉬는 중 - 다음 자세: {label}{retry} ({left:4.1f}초 후 시작, 저장은 함)"
                show(sh, status)
                live_update(sh, status, phase, k, attempt)
            if not completed:
                break
            if paused:
                if saver:
                    saver.kick()             # 멈춘 동안 그래프 저장
                if pause_prompt(sh, label) == "q":
                    completed = False
                    break
                continue                     # 같은 자세를 처음부터 다시 (attempt가 올라감)
            if saver:
                saver.kick()                 # 자세 하나가 끝날 때마다 그래프 저장 (다음 자세의 쉬는 시간 동안 진행)
            k += 1
        if completed:
            print("\n\n모든 자세 수집 완료")

    except KeyboardInterrupt:
        print("\n\n중단됨 (지금까지 저장된 데이터는 파일에 남아 있습니다)")

    finally:
        if hasattr(signal, "SIGTSTP"):
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
        for q in queues:
            q.put(None)                  # 스레드에게 끝내라고 알림 (남은 틱을 처리한 뒤 종료)
        for t in threads:
            t.join(timeout=10)
        if web:
            web.shutdown()
            web.server_close()
        spi.close()
        head.close()
        chest.close()
        for e in sh.errors:
            print(e)
        if fsr_path:
            print(f"\n저장 파일:\n  {fsr_path}  (FSR)\n  {dist_path}  (거리: 머리+가슴)")
            print("자세별 저장 개수 (FSR 수집/쉬는시간, 거리 수집/쉬는시간):")
            for label in LABELS:
                a = attempts.get(label)
                if not a:
                    continue
                f_c, f_r = fsr_counts.get((label, COLLECT, a), 0), fsr_counts.get((label, REST, a), 0)
                d_c, d_r = dist_counts.get((label, COLLECT, a), 0), dist_counts.get((label, REST, a), 0)
                note = f"  (시도 {a}회, 마지막 시도 기준)" if a > 1 else ""
                print(f"  {label}: FSR {f_c}/{f_r}, 거리 {d_c}/{d_r}{note}")
            try:
                if saver:
                    saver.wait()             # 자동 저장이 끝난 뒤 마지막 그래프를 저장
                for p in plot_run(fsr_path, dist_path):
                    print(f"그래프 저장: {p}")
            except Exception as e:
                print(f"그래프 PNG는 만들지 못했습니다 ({e!r}). 나중에 python3 collect_all.py --plot {fsr_path} 로 만들 수 있습니다")
        else:
            print("수집을 시작하지 않았습니다")


if __name__ == "__main__":
    if PLOT_ONLY:
        plot_cli()
    else:
        main()