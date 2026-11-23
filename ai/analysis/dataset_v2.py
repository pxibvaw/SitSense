"""2차 수집 데이터(11명, data/dataset/) 읽기 + 센서 상태 진단 + 접촉 끊김 복구

입력 (한 사람 = 한 세션, 파일 이름의 시각이 같으면 같은 세션)
    posture_data_{YYYYMMDD_HHMMSS}.csv   timestamp, label, phase, attempt, sample, elapsed_s, CH0(맨 앞 왼쪽), …
    distance_data_{YYYYMMDD_HHMMSS}.csv  timestamp, phase, attempt, sample, distance_cm(머리), distance_cm(가슴)
    - phase: 수집(18초 자세 유지) / 쉬는시간(10초, 다음 자세로 바꾸는 중 = 움직임 정답)
    - attempt: 같은 자세를 다시 찍으면 2, 3 … → 마지막 시도만 쓴다 (앞 시도는 잘못 찍은 것)
    - 자세 13개: 바른자세·다리 꼬기는 앞/중/뒤, 편향은 앞/뒤 (편향(중)은 프로토콜상 없음)
    - 0.5초 간격, 압력 7채널 (헤더 괄호 속 위치 이름으로 채널을 찾으므로 배선이 바뀌어도 동작)

제외 (사용자 확인)
    - 이름이 들어간 1차 수집 파일(옛 배선·옛 프로토콜·거리 1개), 2025xxxx 테스트 파일, 76줄짜리 테스트 파일
    - 사람 이름은 데이터에 없으므로 세션 시각 순서대로 p01 ~ p11

센서 상태 진단 (sensor_qc) — 사람 × 채널마다, 자세 유지(수집) 구간 기준
    dead      세션 내내 값이 100 미만 (선 빠짐·고장) → 그 사람에서는 정보 없음
    dropout   자세를 유지하는데 값이 높다가 0 근처로 떨어졌다 돌아옴 (접촉 끊김) 분당 0.5회 이상
    unloaded  바른자세(중) 기준값이 최대치의 5% 미만 (그 사람이 그 센서를 거의 안 누름)
    ok        그 외
접촉 끊김 복구 (repair_dropouts)
    짧은 끊김(3초 = 6개 이하)은 앞뒤 값 사이를 직선으로 채운다.
    긴 끊김(3초 초과)은 복구하지 않고 '신뢰 불가'로 표시 → 학습에서 뺀다.
    자세 유지 중에는 높은 값 → 0 → 높은 값이 나올 수 없어서(같은 자세) 접촉 끊김으로 본다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ai.features.pressure_pipeline import ADC_MAX, DEPTH_MAP, SEAT_LABEL_MAP, channel_key

SESSION_RE = re.compile(r"posture_data_(2026\d{4}_\d{6})\.csv$")
MIN_ROWS = 1000                     # 이보다 짧은 파일은 테스트로 보고 제외
REST, COLLECT = "쉬는시간", "수집"

# 진단 기준
DEAD_P99 = 100                      # 세션 99번째 값이 이보다 작으면 고장
LOW = 50                            # 0 근처 (접촉 끊김)
HIGH = 300                          # 끊김 앞뒤가 이 이상이면 '눌려 있다가 끊긴 것'
SHORT_DROP = 6                      # 이 길이(3초) 이하 끊김은 복구
DROPOUT_PER_MIN = 0.5               # 분당 끊김이 이 이상이면 dropout 상태
UNLOADED_FRAC = 0.05                # 기준값이 최대치의 이 비율 미만이면 unloaded
FRONT = ("L_front", "R_front")      # 판정에 가장 중요한 센서 (1차 분석: 빼면 −16~−22%p)

# 위치 좌표 (무게중심 계산용) x: 왼쪽 − / 오른쪽 +, y: 뒤 − / 앞 +
SIDE_X = {"L": -1.0, "C": 0.0, "R": 1.0}
ROW_Y = {"front": 1.0, "mid": 0.0, "back": -1.0}


def find_sessions(folder: str | Path) -> list[tuple[str, Path, Path | None]]:
    """(세션 시각, posture 파일, distance 파일) — 2차 수집 형식만, 짧은 테스트 파일 제외"""
    folder = Path(folder)
    out = []
    for f in sorted(folder.glob("posture_data_*.csv")):
        m = SESSION_RE.search(f.name)
        if not m:
            continue                                    # 이름 들어간 1차 파일, 형식 다른 파일
        head = pd.read_csv(f, encoding="utf-8-sig", nrows=1)
        if not {"phase", "attempt"} <= set(head.columns):
            continue                                    # 2025 테스트 파일 등 옛 형식
        with open(f, encoding="utf-8-sig") as fh:
            if sum(1 for _ in fh) - 1 < MIN_ROWS:
                continue
        sid = m.group(1)
        dist = folder / f"distance_data_{sid}.csv"
        out.append((sid, f, dist if dist.exists() else None))
    if not out:
        raise FileNotFoundError(f"{folder}에 2차 수집 형식(posture_data_YYYYMMDD_HHMMSS.csv) 파일이 없음")
    return out


def load_session_v2(sid: str, posture: Path, distance: Path | None, person: str) -> tuple[pd.DataFrame, list[str]]:
    p = pd.read_csv(posture, encoding="utf-8-sig")
    rename, channels = {}, []
    for c in p.columns:
        k = channel_key(c)
        if k:
            rename[c] = k
            channels.append(k)
    p = p.rename(columns=rename)
    p["ts"] = pd.to_datetime(p["timestamp"])
    p["person"], p["session"] = person, sid
    p["cls"] = p["label"].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    p["depth"] = p["label"].str.extract(r"\((.)\)")[0].map(DEPTH_MAP)
    p["seat_label"] = p["cls"].map(SEAT_LABEL_MAP)
    if p["seat_label"].isna().any():
        raise ValueError(f"{posture.name}: 모르는 라벨 {sorted(set(p.loc[p.seat_label.isna(), 'cls']))}")
    # 구간 번호: 라벨·시도·단계가 바뀔 때마다 새 구간
    change = (p["label"] != p["label"].shift()) | (p["attempt"] != p["attempt"].shift()) | \
             (p["phase"] != p["phase"].shift())
    p["seg"] = change.cumsum()
    last = p.groupby("label")["attempt"].transform("max")
    p["last_attempt"] = p["attempt"] == last
    p["rest"] = p["phase"] == REST
    if distance is not None:
        d = pd.read_csv(distance, encoding="utf-8-sig")
        head = next(c for c in d.columns if "머리" in c)
        chest = next(c for c in d.columns if "가슴" in c)
        d["ts"] = pd.to_datetime(d["timestamp"])
        m = pd.merge_asof(p[["ts"]].reset_index().sort_values("ts"),
                          d[["ts", head, chest]].sort_values("ts"), on="ts", direction="nearest",
                          tolerance=pd.Timedelta(seconds=0.25)).set_index("index").sort_index()
        p["dist_head"], p["dist_chest"] = m[head].values, m[chest].values
    else:
        p["dist_head"] = p["dist_chest"] = np.nan
    cols = ["ts", "person", "session", "seg", "phase", "rest", "attempt", "last_attempt", "label", "cls",
            "seat_label", "depth"]
    return p[cols + channels + ["dist_head", "dist_chest"]], channels


def load_dataset(folder: str | Path = "data/dataset") -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """(전체 DataFrame, 채널 목록, 사람 표 person/session/rows)"""
    from common.config import repo_path
    sessions = find_sessions(repo_path(folder))
    frames, channels, people = [], None, []
    seg_offset = 0
    for i, (sid, pf, df_) in enumerate(sessions, 1):
        person = f"p{i:02d}"
        df, ch = load_session_v2(sid, pf, df_, person)
        df["seg"] += seg_offset                 # 구간 번호를 사람끼리 겹치지 않게 (groupby('seg')가 사람을 섞지 않도록)
        seg_offset = int(df["seg"].max())
        if channels is None:
            channels = ch
        elif set(ch) != set(channels):
            raise ValueError(f"{pf.name}: 채널 {ch}이 다른 세션 {channels}과 다름")
        frames.append(df[["ts", "person", "session", "seg", "phase", "rest", "attempt", "last_attempt", "label",
                          "cls", "seat_label", "depth"] + channels + ["dist_head", "dist_chest"]])
        people.append({"person": person, "session": sid, "rows": len(df),
                       "retried_labels": ", ".join(sorted(set(df.loc[df.attempt > 1, "label"])))})
    data = pd.concat(frames, ignore_index=True)
    order = ["L_front", "R_front", "L_mid", "R_mid", "L_back", "C_back", "R_back"]
    channels = [c for c in order if c in channels] + [c for c in channels if c not in order]
    return data, channels, pd.DataFrame(people)


# ---------------------------------------------------------------------------
# 접촉 끊김 찾기·복구
# ---------------------------------------------------------------------------
def dropout_runs(v: np.ndarray, low: float = LOW, high: float = HIGH) -> list[tuple[int, int]]:
    """높은 값 사이에 낀 0 근처 구간 (시작, 끝+1) 목록"""
    runs, i, n = [], 0, len(v)
    lo = v <= low
    while i < n:
        if lo[i]:
            j = i
            while j < n and lo[j]:
                j += 1
            if i > 0 and j < n and v[i - 1] >= high and v[j] >= high:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def repair_dropouts(df: pd.DataFrame, channels: list[str], max_len: int = SHORT_DROP):
    """자세 유지(수집) 구간 안의 접촉 끊김: 짧으면 직선으로 채우고, 길면 신뢰 불가로 표시

    돌려주는 값: (복구된 DataFrame, 신뢰 불가 줄 표시 Series, 복구 기록 DataFrame)"""
    out = df.copy()
    bad = pd.Series(False, index=df.index)
    log = []
    for seg, g in df[~df["rest"]].groupby("seg", sort=False):
        for c in channels:
            v = g[c].to_numpy(dtype=float)
            for i, j in dropout_runs(v):
                idx = g.index[i:j]
                if j - i <= max_len:
                    out.loc[idx, c] = np.round(np.linspace(v[i - 1], v[j], j - i + 2)[1:-1])
                    kind = "repaired"
                else:
                    bad.loc[idx] = True
                    kind = "unreliable"
                log.append({"person": g["person"].iloc[0], "seg": seg, "label": g["label"].iloc[0],
                            "channel": c, "start": int(idx[0]), "length": j - i, "kind": kind})
    return out, bad, pd.DataFrame(log, columns=["person", "seg", "label", "channel", "start", "length", "kind"])


# ---------------------------------------------------------------------------
# 센서 상태 진단
# ---------------------------------------------------------------------------
def baselines(df: pd.DataFrame, channels: list[str], movement: pd.Series | None = None,
              n: int = 20, skip: int = 4) -> dict[str, pd.Series]:
    """사람마다 바른자세(중) 기준값 = 마지막 시도 수집 구간에서 앞 2초(skip)를 건너뛰고, 움직임이 아닌 첫 10초(n) 중앙값
    (실제 서비스의 '측정 시작 바른자세 10초'와 같은 길이)"""
    out = {}
    for person, g in df.groupby("person", sort=False):
        b = g[(g["cls"] == "바른자세") & (g["depth"] == "middle") & ~g["rest"] & g["last_attempt"]]
        b = b.iloc[skip:]
        if movement is not None:
            b = b[~movement.reindex(b.index, fill_value=False)]
        if len(b) < n:
            raise ValueError(f"{person}: 바른자세(중) 기준 측정에 쓸 순간이 {len(b)}개뿐")
        out[person] = b[channels].head(n).median()
    return out


def sensor_qc(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """사람 × 채널 상태표 (자세 유지 구간, 마지막 시도)"""
    use = df[~df["rest"] & df["last_attempt"]]
    base = baselines(df, channels)
    rows = []
    for person, g in use.groupby("person", sort=False):
        minutes = len(g) * 0.5 / 60
        segs = list(g.groupby("seg", sort=False))
        for c in channels:
            n_drop = sum(len(dropout_runs(s[c].to_numpy(dtype=float))) for _, s in segs)
            zero_blocks = sum(int(s[c].median() < LOW) for _, s in segs)
            p99 = float(g[c].quantile(0.99))
            b = float(base[person][c])
            if p99 < DEAD_P99:
                status = "dead"
            elif n_drop / minutes >= DROPOUT_PER_MIN:
                status = "dropout"
            elif b < UNLOADED_FRAC * ADC_MAX:
                status = "unloaded"
            else:
                status = "ok"
            rows.append({"person": person, "channel": c, "status": status, "p99": round(p99),
                         "baseline": round(b), "dropouts": n_drop, "dropouts_per_min": round(n_drop / minutes, 2),
                         "zero_blocks": zero_blocks, "blocks": len(segs)})
    return pd.DataFrame(rows)


def session_flags(qc: pd.DataFrame) -> pd.DataFrame:
    """학습 제외 후보: 앞쪽 센서(판정 핵심)가 고장이거나, 바르게 앉아도 안 눌리고 대부분 자세에서 0인 사람"""
    rows = []
    for person, g in qc.groupby("person", sort=False):
        f = g[g["channel"].isin(FRONT)]
        reasons = []
        for _, r in f.iterrows():
            if r["status"] == "dead":
                reasons.append(f"{r['channel']} 고장")
            elif r["baseline"] < UNLOADED_FRAC * ADC_MAX and r["zero_blocks"] >= r["blocks"] / 2:
                reasons.append(f"{r['channel']} 자세 {r['zero_blocks']}/{r['blocks']}개에서 0")
        rows.append({"person": person, "flagged": bool(reasons), "reason": ", ".join(reasons)})
    return pd.DataFrame(rows)
