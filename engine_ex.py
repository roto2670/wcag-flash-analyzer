"""
engine_ex.py — 참조 코드(flash_analyzer_ex.py) 기반 엔진.

flash_analyzer_ex.FlashAnalyzer를 그대로 쓰고, engine.FpaCoreEngine과 같은 인터페이스
(update(frame) → (휘도 dict, 적색 dict), .sat)로 감싼다. 분석 해상도는 참조 기본값(96×72)이 아니라
scale=30(288×216) — FPA 비교에서 96×72는 PEAT FAIL 7구간 중 2개, 288×216은 6개를 잡았다.
판정 창 선택이 파이썬 루프라 engine.py보다 몇 배 느리다.
"""

from flash_analyzer_ex import FlashAnalyzer, FlashConfig

EX_SCALE = 30             # 분석 해상도 배율 (10 → 96×72, 30 → 288×216)


class ExEngine:
    """프레임마다 update(frame) → (휘도 dict, 적색 dict).
    dict = count(진단 전환 수), fail, warn, risk(점수 R), area(위험 면적 비율)."""

    def __init__(self, fps, area_thr=0.25):
        cfg = FlashConfig(flash_area_pct=int(round(area_thr * 100)), scale=EX_SCALE)
        self.fa = FlashAnalyzer(mode=2 if fps >= 27.5 else 1, config=cfg)
        self.sat = None

    def update(self, frame_bgr):
        r = self.fa.process_frame_bgr(frame_bgr)
        cur = self.fa.frames[-1]
        self.sat = cur.red_ratio() > self.fa.cfg.red_saturation_pct
        lum = {"count": r.lum_k_count, "fail": r.lum_class == 2, "warn": r.lum_class == 1,
               "risk": r.lum_score, "area": r.lum_area_pct / 100.0}
        red = {"count": r.red_k_count, "fail": r.red_class == 2, "warn": r.red_class == 1,
               "risk": r.red_score, "area": r.red_area_pct / 100.0}
        return lum, red
