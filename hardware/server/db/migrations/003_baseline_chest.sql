-- 003: 기준 자세에 가슴 거리(mm) 저장 — 서버를 다시 켜도 거북목 가슴 보조 기준이 유지되도록
-- 가슴 센서가 없거나 측정 실패면 NULL

ALTER TABLE baselines ADD COLUMN chest_mm REAL;
