-- 004: 기준 자세에 앉은 위치별(앞·뒤) 압력 기준 저장 — JSON {"front": [채널별], "back": [채널별]}
-- 바른자세(중) 기준을 다시 재면 새 행이라 NULL로 시작한다. 앞·뒤를 안 쟀으면 NULL

ALTER TABLE baselines ADD COLUMN positions TEXT;
