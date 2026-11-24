# SitSense

**카메라·압력센서 융합 기반 비착용형 자세 습관 분석 및 맞춤 피드백 시스템**
(2026 가을 센서 프로그래밍 캡스톤디자인)

모니터에 단 카메라·거리센서와 의자 방석의 FSR 압력센서로 앉은 자세를 판단하고,
Raspberry Pi 서버를 거쳐 안드로이드 앱에서 자세 리포트를 보여준다.

```
센서 → 숫자 → 판단 → 기록 → 앱 화면
[하드웨어]    [AI]   [하드웨어]  [앱]
```

## 주요 결과 (2026.11.24 최종)

처음 보는 사람 기준 평가 (한 사람씩 빼고 학습·시험, LOSO)

| 판정 | 센서 | 결과 (찍기) | 데이터 |
|---|---|---|---|
| ① 바른자세 vs 나쁜 자세 | 압력 + 머리 거리 | **86.3%** (50%) | 2차 수집 11명 |
| ② 편향 vs 다리 꼬기 | 압력 | 59.9% (50%) → 앱은 "추정" 표시 | 2차 수집 11명 |
| ③ 체중이 실린 쪽 | 압력 | **76.6%** (50%) | 2차 수집 11명 |
| 앱 좌면 최종 (바른자세 / 왼쪽 / 오른쪽 체중) | 압력 + 머리 거리 | **67.4%** (33%) | 2차 수집 11명 |
| 앉은 위치 (앞·중·뒤) | 머리 거리 | 80.4% (33%) | 2차 수집 11명 |
| 거북목 | 카메라 + 거리 2개 | **99.4%** | 거북목 수집 3명 (예비 검증) |
| 좌우 기울기 | 카메라 | **99.7%** (방향 98.7%) | 거북목 수집 3명 (예비 검증) |
| 화면 거리 | 머리 거리 | 40cm 경고 / 50cm 적정 | 고용노동부 VDT 지침·OSHA·AOA |

**해결한 문제**
- 1차 3명 모델(79%)이 새 사람 11명에게 31~45% → 원인: FSR 포화(0 아니면 3500 근처)와 앉은 위치가 자세보다 압력을 더 크게 바꿈
  → 5개 자세를 한 번에 맞히지 않고 **① → ② → ③ 단계별 판정**, 머리 거리로 앉은 위치를 알아 **위치별 기준** 사용 (① 71.5% → 85~86%)
- 다리 꼬기 체중 방향을 꼰 쪽으로 묶은 실수 → 꼰 다리 **반대쪽**에 체중 (55% → 79%)
- 센서 고장·접촉 끊김·움직임 구간 → 센서 진단, 짧은 끊김 복구, 움직임 구간 학습 제외
- 거북목 기준이 임의값이고 숙임·기울기를 100% 거북목으로 오판 → 직접 수집한 데이터로 기준을 정하고
  **기울기 판정(카메라)과 가슴 거리센서로 숙임을 걸러냄** (73.5% → 99.6%)
- 0.5초 단위 판정의 잦은 상태 변화 → 최근 2초로 떨림 줄이기 + 4초 유지 (ISO 11226)

전체 기준과 근거: [`docs/judgement_rules.md`](docs/judgement_rules.md) · 분석 그래프: `python -m ai.analysis.story --all`, `python -m ai.analysis.neck_report`

## 팀 구성

| 파트 | 담당 |
|---|---|
| 기획 · 앱 UI 설계(Figma) · AI · 데이터 수집 | 최윤서 |
| 하드웨어 설계 · 구현 · 데이터 수집 | 김리원 |
| 앱 개발 · 데이터 수집 | 유인서 |

## 폴더 구조

```
SPteam6/
├── app/                    # [앱] Android Studio 프로젝트 (Kotlin)
│
├── hardware/               # [하드웨어]
│   ├── sensors/            #   실제 센서 읽기 코드 (Picamera2, MCP3208+FSR, HC-SR04 거리센서)
│   ├── server/             #   FastAPI + SQLite 서버 (앱에 데이터 전달)
│   └── docs/               #   배선도, FSR 방석 배치도, 거치대 설계
│
├── ai/                     # [AI]
│   ├── camera/             #   MediaPipe로 목 전방·어깨 기울기 판단
│   ├── features/           #   압력·거리 값에서 특징(feature) 추출
│   ├── classify/           #   RF/SVM/KNN 학습·평가 (사람 단위 평가)
│   ├── engine/             #   카메라+압력+거리 결과 합치기 (최근 2초 + 4초 유지)
│   ├── analysis/           #   발표용 분석 (story 0~17단계, neck_report)
│   ├── models/             #   학습된 모델 파일 (최종 모델만 git에 올림)
│   └── notebooks/          #   실험·분석용 노트북
│
├── common/                 # [하드웨어+AI 공동] 수정 시 서로 PR 확인 후 합치기
│   ├── sensors_base.py     #   센서 인터페이스 + 센서 값 형식 (압력 8칸/거리 mm/카메라 점 + 시간)
│   ├── schema.py           #   판단 결과 형식 + 자세 라벨 목록 (normal, cross_left …)
│   ├── mock_sensors.py     #   가짜 센서 (하드웨어 없이 테스트용)
│   ├── recorder.py         #   센서 값을 CSV로 저장 (수집 파일 형식 통일)
│   ├── replay.py           #   저장한 CSV를 센서처럼 재생 (맥북에서 반복 테스트)
│   └── config.py           #   config.yaml 읽기
│
├── data/                   # [공동] 수집 데이터 (git 제외, 원본은 구글 드라이브 공유)
│   ├── dataset/            #   학습·분석에 쓰는 수집 데이터
│   │   ├── posture/        #     압력 (2차 수집 11명, posture_data_*.csv)
│   │   ├── distance/       #     머리·가슴 거리 (distance_data_*.csv)
│   │   └── neck/           #     거북목·기울기 (카메라 점 + 거리 2개, n1~n3)
│   ├── raw/                #   main.py --record 세션 CSV
│   └── processed/          #   분석 결과·그래프 (코드로 다시 만들 수 있음)
│
├── docs/                   # [공동] API 명세, 회의록
├── config.yaml             # 센서 주기, 채널 번호, threshold 등 설정값
├── main.py                 # Pi에서 전체 시스템 실행 (센서 → 판단 → 저장)
├── requirements.txt        # 공통 라이브러리 (맥·Pi 모두)
├── requirements-pi.txt     # Pi 전용 라이브러리 (Picamera2, spidev)
├── .gitignore              # git에 안 올릴 파일 목록
└── README.md               # 프로젝트 소개 + 환경 세팅 방법
```

## 브랜치 규칙

```
main          ← 항상 돌아가는 버전. PR로만 합치기
├─ ai         ← AI 담당
├─ hardware   ← 하드웨어 담당
└─ app        ← 앱 담당
```

1. 기능 하나 완성되면 바로 PR로 `main`에 합치기 (최소 주 1회)
2. 작업 시작 전 `git pull origin main`으로 최신 내용 받기
3. `common/` 수정은 상대방이 PR 확인 후 합치기
4. 발표 전에는 태그 붙이기 (예: `v0.1-midterm`)

처음 한 번만 자기 브랜치 만들기:

```bash
git checkout main && git pull
git checkout -b hardware        # 또는 ai, app
git push -u origin hardware
```

## 환경 세팅

Python은 **3.11**로 통일한다 (Raspberry Pi OS Bookworm 64-bit 기본 버전).

### 맥 / 윈도우 (개발용)

```bash
# 맥: brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate
pip install -r requirements.txt
```

### Raspberry Pi 4 (64-bit Bookworm)

```bash
sudo apt update
sudo apt install -y python3-picamera2        # Picamera2는 apt로 설치
python3 -m venv --system-site-packages .venv # 시스템 패키지(picamera2, gpiozero) 사용
source .venv/bin/activate
pip install -r requirements-pi.txt
```

`sudo raspi-config` → Interface Options에서 **SPI**(MCP3208용)를 켠다. HC-SR04는 GPIO라 따로 켤 것이 없고, Echo 핀은 분압 회로로 3.3V로 낮춰서 연결한다.

### 설치 확인

저장소 최상위에서 실행:

```bash
python main.py --mode mock --seconds 3     # 가짜 센서 값이 출력되면 성공
```

## 하드웨어 구성

| 센서 | 연결 | 설치 위치 |
|---|---|---|
| FSR 406 × 6 (MCP3208 CH0 앞왼, CH1 뒤왼, CH3 중왼, CH4 뒤오, CH6 중오, CH7 앞오. CH2·CH5는 쓰지 않음) | MCP3208 1개 → SPI (CE0) | 아크릴 판 위에 고정 → 커버 씌워 방석 안 |
| HC-SR04 머리 (화면 거리·앉은 위치·거북목) | GPIO (Trig 23, Echo 24, BCM) | 모니터 위, 카메라와 함께 |
| HC-SR04 가슴 (거북목 보조·숙임 거르기) | GPIO (Trig 5, Echo 6, BCM) | 가슴 높이, 고정 |
| 카메라 (정면, 1280×720) | CSI (Picamera2) | 모니터 위 |

## 카메라 상체 판단 데모 (맥 웹캠)

```bash
python -m ai.camera.demo                 # 처음 10초 바른 자세 → 거북목·좌우 기울기 판단 표시
python -m ai.camera.demo --no-window     # 화면 없이 1초마다 결과만 출력
```

- 키: `b` 기준 다시 측정, `q` 종료
- 처음 실행 때 맥이 카메라 접근을 물어보면 터미널(또는 VS Code)을 허용
- 판단 기준값은 `config.yaml`의 `upper_rules` (거북목·기울기 수집 3명 데이터로 정한 최종값, 근거는 `docs/judgement_rules.md` 1·2절)
- 데모는 카메라만 써서, 몸 전체 숙임은 걸러지지 않는다 (가슴 거리센서가 있는 실제 서버에서 걸러짐)

## 데이터 수집

| 무엇 | 명령 (Pi) | 방법 |
|---|---|---|
| 압력 + 거리 2개 (좌면 학습용) | `python3 hardware/sensors/collect_all1117.py` | 13자세 × 3분, 실시간 그래프 `http://<Pi>:8000` |
| 거북목·기울기 (카메라 + 거리 2개) | `python -m ai.camera.collect_neck --subject n1` | [`docs/neck_collect.md`](docs/neck_collect.md), 8자세 × 30초 |

분석 (맥): `python -m ai.classify.train_staged`(좌면 모델 다시 만들기), `python -m ai.analysis.story --all`, `python -m ai.analysis.neck_report`

세션 단위 기록 (예전 방식):

```bash
python main.py --record --subject p1 --seat cross_left --head normal --tilt none --pos middle --seconds 40
```

- 한 번 실행 = 세션 1개 → `data/raw/p1_날짜_시간.csv` + 같은 이름의 `.json`(세션 정보)
- **처음 10초는 바른 자세로 앉기** (`phase=baseline`, 기준 측정). 그다음 지정한 자세 유지
- 라벨은 4개를 따로 정한다 (값은 `common/schema.py`에 있는 것만)
  - `--seat`: `normal`, `lean_left`, `lean_right`, `cross_left`, `cross_right`, `empty`
  - `--head`: `normal`, `forward`(거북목)
  - `--tilt`: `none`, `left`, `right` (사람 기준 방향)
  - `--pos`: `front`(의자 앞쪽), `middle`, `back`(등받이까지 깊게), `none`(자리 비움). 앉은 위치
- 저장한 세션 재생: `python main.py --mode replay --file data/raw/파일이름.csv`
- 필드 설명은 [`docs/data_format.md`](docs/data_format.md)

## 데이터 규칙

- `data/raw/` 파일은 수정하지 않는다. 원본은 팀 구글 드라이브에 올려서 공유
- `subject`는 `p1`, `p2`처럼 번호로. 실명 쓰지 않기
- 카메라 이미지는 저장하지 않고, MediaPipe 좌표만 저장
- 카메라나 모니터 위치가 바뀌면 새 세션으로 시작 (기준 자세를 다시 재야 함)
- 최종 모델만 `ai/models/final_*` 이름으로 git에 올린다
