# Pi에서 거북목·좌우 기울기 데이터 수집하기

카메라(상체 점) + 거리센서 2개(머리·가슴)를 같이 기록한다. 거북목과 카메라 좌우 기울기 기준값을 한 번에 확인한다. 코드: `ai/camera/collect_neck.py`
진행 방식은 `docs/pi_camera_collect.md`(카메라 기울기 수집)와 같다: AI 담당이 push → Pi에서 받아서 **실행만**.

## 1. Pi 준비 (처음 한 번, 이미 했으면 생략)

`docs/pi_camera_collect.md`의 "1. Pi 준비"와 같다 (64-bit Bookworm, Python 3.11, `python3-picamera2`, `.venv --system-site-packages`, `pip install -r requirements-pi.txt`).

```bash
uname -m                          # aarch64
rpicam-hello --list-cameras       # 카메라 1개
cd SPteam6 && git pull && source .venv/bin/activate
python -c "import mediapipe, picamera2, gpiozero; print('OK')"
```

## 2. 확인 (사람마다 시작 전 15초, 저장 안 함)

```bash
python -m ai.camera.collect_neck --check
```

- `카메라 판단 가능`이 대부분이어야 함 (아니면 조명, 어깨까지 화면에 들어오는지)
- 머리 거리 50~70cm 근처, 가슴 거리가 숫자로 나와야 함 (가슴 센서는 고정이라 17cm 근처여도 괜찮음)
- 거리센서 핀은 압력 수집 프로그램(`hardware/sensors/collect_all1117.py`)과 같음: 머리 23/24, 가슴 5/6.
  **압력 수집 프로그램과 동시에 켜지 않는다** (같은 핀).

## 3. 수집 (1인당 약 4분 40초)

1. 몸에 붙는 옷 (후드·패딩은 벗기)
2. 휴대폰을 옆(귀·어깨가 보이는 높이)에 두고 **영상 녹화 시작**
3. 실행:
   ```bash
   python -m ai.camera.collect_neck --subject n1      # n1, n2, n3 … (실명 금지)
   ```
4. 화면 안내대로 자세를 바꾸고, 바뀔 때마다 **자세 이름을 소리 내어 읽기** (영상에서 구간 찾기용)

| 순서 | 자세 | 하는 법 | 시간 |
|---|---|---|---|
| 1 | 바른자세 | 바르게 앉아 화면 보기 (기준) | 30초 |
| 2 | 머리내밀기 | 몸통은 그대로, 머리만 화면 쪽으로 | 30초 |
| 3 | 어깨말림 | 머리 위치는 그대로, 어깨를 말고 등을 구부정하게 | 30초 |
| 4 | 상체숙임 | 허리부터 몸 전체를 앞으로 (목은 곧게) | 30초 |
| 5 | 등받이기댐 | 등받이에 기대기 | 30초 |
| 6 | 왼쪽기울기 | 상체를 '내' 왼쪽으로 (평소 삐딱하게 앉는 정도) | 30초 |
| 7 | 오른쪽기울기 | 상체를 '내' 오른쪽으로 | 30초 |
| 8 | 바른자세2 | 다시 바르게 (마무리) | 30초 |

자세 사이 5초 쉬는 시간에 다음 자세로 바꾼다.

(선택) 헐렁한 옷으로 한 번 더:
```bash
python -m ai.camera.collect_neck --subject n1 --clothes loose --only 머리내밀기 어깨말림
```

## 4. 넘기기

- 저장 위치: `data/dataset/neck/{subject}_{tight|loose}_{날짜_시각}.csv` → git으로 올리거나 AI 담당에게 전달
- 휴대폰 영상은 git에 올리지 않는다 (얼굴). 자세마다 옆모습 한 장씩만 골라 CVA(귀–목–수평선 각도)를 재고
  `data/dataset/neck/cva.csv`에 `subject,label,cva` 로 적는다 (선택, 임상 기준 48~50° 미만 = 거북목)
- 이미지는 저장하지 않고 MediaPipe 좌표만 저장한다

## 5. 분석 (Mac, AI 담당)

```bash
python -m ai.analysis.neck_report      # data/processed/neck/ 에 그래프 6장 + summary.md
```

`summary.md`의 yaml 값(거북목 `neck_drop_ratio`·`face_grow_ratio`, 기울기 `shoulder_angle_deg`·`head_angle_deg`·`head_offset`)을
`config.yaml` upper_rules에 넣고 `python -m ai.engine.check_engine`. 데이터가 구분하지 못하는 범위에서는 지금 값을 유지하도록 고른다.

