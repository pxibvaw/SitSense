-- 005: 요약이 끝난 날의 상세(착석 구간·세션 이벤트·지난 기준 이력)를 지운 시각
-- 지운 날은 요약을 다시 계산할 수 없으므로(착석 구간에서 계산) 다시 계산 대상에서 뺀다

ALTER TABLE day_jobs ADD COLUMN details_deleted_at REAL;
