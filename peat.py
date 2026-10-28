"""
peat.py — Harding/ITU-R BT.1702-3 기반 광과민성 발작 분석 엔진 (PEAT 충실 재구현)

공식 PEAT 분석 코어는 Cambridge Research Systems의 Harding FPA 엔진을 라이선스한
독점 모듈이라 코드 복제가 불가능하다. 이 파일은 PEAT가 실제로 따르는 표준
(WCAG 2.3.1 + ITU-R BT.1702-3 + Harding/Wilkins 패턴 기준)을 직접 구현한 것이다.

핵심 구현:
  1. cd/m² 절대 휘도 기준: ≥20 cd/m² 변화(어두운쪽 <160), 그 이상은 Michelson >1/17
  2. 적색 플래시: WCAG 작업정의 — 포화적색 R/(R+G+B)≥0.8 + (R−G−B)×320 변화 >20
  3. 줄무늬(공간) 패턴 분석 추가
  4. 페이싱 면제(앞뒤 간격이 모두 ≥334ms/60Hz·≥360ms/50Hz인 '고립' 전환만)

출처:
  - WCAG 2.3.1: https://www.w3.org/WAI/WCAG21/Understanding/three-flashes-or-below-threshold.html
  - ITU-R BT.1702-3: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf
  - PEAT 오픈 클론(참조): https://github.com/rakeeb-hossain/PEAT_V2
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
# 표준 상수 (ITU-R BT.1702-3 / WCAG 2.3.1 / Harding)
# ──────────────────────────────────────────────────────────────────────────
AREA_THRESHOLD = 0.25     # 플래시 면적: 10° 박스(화면 1/3×1/3)의 25% '초과'
WINDOW_SECONDS = 1.0      # 전환 수를 세는 창
FAIL_TRANSITIONS = 7      # 1초 내 전환 ≥7회 (= >3 flash)

EXTENDED_SECONDS = 5.0    # 5초 지속 플래시 경고

# 공간 패턴(줄무늬) 기준 — Harding/Wilkins
PATTERN_MIN_PAIRS = 5         # >5 광-암 쌍이면 위험 후보
PATTERN_LIGHT_CD = 50.0       # 가장 밝은 줄무늬 > 50 cd/m²
PATTERN_AREA_STATIC = 0.40    # 정지 패턴: 화면 >40%
PATTERN_AREA_MOVING = 0.25    # 움직임/점멸/반전 패턴: 화면 >25%
PATTERN_MIN_SECONDS = 0.5     # ≥0.5초 지속
DISPLAY_PEAK_CD = 200.0       # 패턴 cd/m² 환산: SDR 백색 200 cd/m² 가정 (ITU)
DISPLAY_GAMMA = 2.4           # 패턴 cd/m² 환산: BT.1886 디스플레이 감마 (ITU-R BT.1702-3 Table 1)
FLASH_DELTA_CD = 20.0         # 패턴 광-암 차 하한 (cd/m²)

# ──────────────────────────────────────────────────────────────────────────
# 휘도 플래시: PEAT식 픽셀 엔진
# ──────────────────────────────────────────────────────────────────────────
# 공식 PEAT 결과 파일(.fpa)을 합성 테스트 영상 63개(면적·밝기·횟수·이동·해상도를 하나씩
# 바꾼 288×216 AVI)와 실제 QA 영상 7개로 역산해 맞춘 규칙 (2026-10-27).
#  - 밝기: sRGB 선형화 + Rec.709 상대휘도 Y(0~1). 감마 2.4는 0↔96 점멸(PEAT FAIL)을 19cd로
#    환산해 놓쳤고, PEAT 위험도 세기도 sRGB ΔY에 1차식으로 정확히 맞음(최대오차 0.3%).
#    플래시 기준 ΔY≈0.095(0↔86 안 셈, 88 셈), 전환 수 기준 ΔY≈0.048(60 안 셈, 64 셈),
#    어두운 쪽 Y≥0.8이면 면제(230↔255 FAIL, 235↔255 면제 — WCAG 규정 그대로).
#  - 전환 수(진단 실선, PEAT 보고서 휘도 횟수) = 10° 박스(화면 1/3×1/3)에서 '최근 1초 전환
#    ≥k' 픽셀 면적이 25%를 '초과'하는 최대 k (정확히 25% 패치는 0회).
#  - 움직임 허용: 지금 값이 직전 프레임 이웃(십자 ±r) 값 범위 안이고 그 반대도 성립하면
#    이동으로 보고 전환으로 세지 않는다(극값은 현재값으로 흡수). 이동 줄무늬 실험에서 288폭
#    기준 프레임당 7px 이하는 무시·9px 이상은 FAIL, 해상도에 비례(576폭에서 2배). 경계가
#    부드러운 사인 줄무늬는 같은 이동에도 FAIL이라 블록 추적이 아니라 '주변 값 존재' 방식.
#    정확한 이웃 표본 모양은 합성으로 구분 불가 — 실제 7개 FPA에서 십자 r=5가 누락 0·과민 최소.
#  - FAIL = 전환 수 ≥7 AND 최근 7프레임(0.23초) 안에 플래시 기준 전환한 픽셀 면적 >10%.
#    깜빡임이 멈추면 1초 창 카운트가 남아 있어도 0.23초 뒤 FAIL이 끝난다(PEAT와 동일).
#  - 위험도(점선, PEAT 보고서 Lum 0.0~3.4, ≥0.5 FAIL): 합성 실측 0.46×(A/0.25)^3.07,
#    A = 최근 플래시 픽셀 중 1초 전환 ≥k인 면적. 7회 미만이면 CAUTION(5회 0.3, 6회 0.4) 상한.
PEAT_WIDTH = 288          # PEAT 분석 기준 폭 — 입력은 이 폭으로 축소(인코딩 공간)
PEAT_COUNT_DY = 0.048     # 전환으로 세는 최소 상대휘도 변화
PEAT_FLASH_DY = 0.095     # 플래시(위험도) 기준 상대휘도 변화 ≈ 20cd
PEAT_DARK_Y = 0.8         # 어두운 쪽이 이 이상이면 플래시 아님
PEAT_MOTION_R = 5         # 움직임 허용 십자 이웃 반경(PEAT_WIDTH 기준 px)
PEAT_RECENT_S = 7 / 30.0  # FAIL 유지에 필요한 최근 플래시 시간창
PEAT_RECENT_AREA = 0.10   # 최근 플래시 픽셀의 10° 박스 면적 하한
# 적색(합성 테스트 8차): WCAG 적색값 (R−G−B)×320을 sRGB 선형값으로 계산(감마 2.4는 r75·80 FAIL을 놓침).
#   플래시 기준 20(r70=19.6 안 셈, r75=22.5 셈), 전환 수 기준은 더 낮음(r60=14.5도 셈),
#   포화 적색 R/(R+G+B) 경계는 0.8이 아니라 약 0.816(합성 9차: (255,94,94)=0.8171 셈,
#   (255,95,95)=0.8145 안 셈; 초록·파랑만 섞어도 같은 비율에서 같은 결과 → 공식 R/(R+G+B) 확인).
#   위험 세기 = 1.8×적색값 변화+229 (혼합색 포함 오차 1% 안), 면적 25% 초과·FAIL 종료는 휘도와 같음.
PEAT_RED_FLASH = 20.0
PEAT_RED_COUNT = 10.0
PEAT_RED_SAT = 0.816
RISK_FAIL = 0.5           # PEAT 위험도 합격선
RISK_MAX = 3.4            # PEAT 위험도 천장
# 휘도(2026-10-28, FPA 13개 프레임 대조로 맞춤 — 전환 수 일치 72.8%→89.2%, ±1 이내 99.3%):
#  - 휘도 = 채널별 sRGB 선형값×200을 정수 cd/m²로 반올림 → Rec.709 고정소수점 가중합(>>16).
#  - 블록(36px) 히스토그램(0~200 cd/m² 61칸) 차이 L1 > 0.36×블록 픽셀 수인 블록만 '실제 밝기
#    변화'로 보고, 나머지는 움직임으로 보아 전환을 세지 않는다. 물체가 움직여도 블록 밝기 분포는
#    거의 그대로라 이 구분이 이동 허용 규칙(±5px 이웃)보다 PEAT에 가깝다. 픽셀을 옮겨 맞추는
#    이동 보정이나 '옮긴 블록' 히스토그램 비교는 오히려 PEAT와 멀어졌다.
#  - 전환 = 같은 위치 직전 프레임 대비 ±7 cd/m² 초과 + 방향 교대(기대 방향), 1초 창 전환이
#    0이 되면 기대 방향 초기화. 면적은 화면 0.4×0.4 박스(1/3 박스보다 FAIL 경계가 PEAT에 가까움).
PEAT_PEAK_CD = 200.0
PEAT_HIST_BLOCK = 36      # PEAT_WIDTH 기준 블록 크기(px)
PEAT_HIST_BINS = 61
PEAT_HIST_THR = 0.36
PEAT_LUM_COUNT_CD = 7.0   # 전환 수용 변화 기준
PEAT_LUM_FLASH_CD = PEAT_FLASH_DY * PEAT_PEAK_CD   # 플래시(FAIL·위험도)용 변화 기준
PEAT_DARK_CD = PEAT_DARK_Y * PEAT_PEAK_CD          # 어두운 쪽이 이 이상이면 플래시 아님
PEAT_LUM_BOX = 0.4        # 휘도 면적 박스(화면 비율)

_SRGB_LUT = np.arange(256, dtype=np.float32) / 255.0
_SRGB_LUT = np.where(_SRGB_LUT <= 0.04045, _SRGB_LUT / 12.92,
                     np.power((_SRGB_LUT + 0.055) / 1.055, 2.4)).astype(np.float32)


def peat_rgb(frame_bgr, resize=True):
    """BGR 프레임 → PEAT 기준 폭(resize=False면 원본 크기)의 sRGB 선형 BGR(0~1)."""
    H, W = frame_bgr.shape[:2]
    if resize and W != PEAT_WIDTH:
        frame_bgr = cv2.resize(frame_bgr, (PEAT_WIDTH, max(3, int(round(H * PEAT_WIDTH / W)))),
                               interpolation=cv2.INTER_AREA)
    return _SRGB_LUT[frame_bgr]


def peat_luma(frame_bgr, resize=True):
    """BGR 프레임 → sRGB 상대휘도 Y(0~1)."""
    lin = peat_rgb(frame_bgr, resize)
    return 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]


def peat_flash_mask(frame_bgr, prev_bgr):
    """두 프레임 사이 PEAT 플래시 기준(ΔY≥0.095, 어두운 쪽 <0.8)을 넘는 픽셀 — 미리보기 강조용."""
    a, b = peat_luma(frame_bgr, resize=False), peat_luma(prev_bgr, resize=False)
    return (np.abs(a - b) >= PEAT_FLASH_DY) & (np.minimum(a, b) < PEAT_DARK_Y)


def peat_red_mask(frame_bgr, prev_bgr):
    """두 프레임 사이 PEAT 적색 플래시 기준(포화 적색 + 적색값 변화 ≥20)을 넘는 픽셀 — 미리보기 강조용."""
    out = []
    for f in (frame_bgr, prev_bgr):
        lin = peat_rgb(f, resize=False)
        B, G, R = lin[..., 0], lin[..., 1], lin[..., 2]
        out.append((np.maximum(R - G - B, 0.0) * 320.0, R >= PEAT_RED_SAT * (R + G + B + 1e-6)))
    (va, sa), (vb, sb) = out
    return (np.abs(va - vb) >= PEAT_RED_FLASH) & (sa | sb)


def _box_area_map(mask, frac=None):
    """박스(기본 10° = 화면 1/3×1/3, frac를 주면 화면의 frac배)가 화면 안에 완전히 들어가는
    위치별 면적 비율."""
    H, W = mask.shape
    if frac is None:
        kh, kw = max(1, H // 3), max(1, W // 3)
    else:
        kh, kw = max(1, int(round(H * frac))), max(1, int(round(W * frac)))
    loc = cv2.boxFilter(mask.astype(np.float32), -1, (kw, kh), normalize=True)
    y0, x0 = kh // 2, kw // 2
    return loc[y0:y0 + H - kh + 1, x0:x0 + W - kw + 1]


def _risk_from_area(area):
    """플래시 면적(10° 박스 비율) → PEAT 위험도. 합성 실측(25.7%→0.5, 30%→0.8, 40%→1.9)."""
    return min(RISK_MAX, 0.459 * (area / AREA_THRESHOLD) ** 3.07) if area > 0 else 0.0


class _PixelExtremes:
    """픽셀별 극값·방향 추적과 최근 1초 전환 이력.
    sat를 주면(적색) 지금 또는 극값 상태가 포화 적색인 픽셀만 전환으로 센다."""

    def __init__(self, shape, thr, window_frames, dark=None):
        self.thr, self.dark, self.wf = thr, dark, window_frames
        self.ext = None
        self.sat_ext = None
        self.dr = np.zeros(shape, np.int8)
        self.hist = np.zeros((window_frames,) + shape, bool)
        self.age = np.full(shape, 9999, np.int32)   # 마지막 전환 후 지난 프레임 수
        self.i = 0

    def step(self, Y, explained, sat=None):
        if self.ext is None:
            self.ext = Y.copy()
            self.sat_ext = sat
            return
        d = Y - self.ext
        sg = np.sign(d).astype(np.int8)
        big = np.abs(d) >= self.thr
        if self.dark is not None:
            big &= np.minimum(Y, self.ext) < self.dark
        if sat is not None:
            big &= sat | self.sat_ext
        tr = big & (sg != 0) & (sg != self.dr) & ~explained
        self.ext = np.where(explained, Y, self.ext)          # 이동으로 설명된 변화는 흡수
        self.ext = np.where(tr, Y, self.ext)
        self.dr = np.where(tr, sg, self.dr)
        grow = (~tr) & (((self.dr > 0) & (Y > self.ext)) | ((self.dr < 0) & (Y < self.ext)))
        self.ext = np.where(grow, Y, self.ext)
        self.ext = np.where((self.dr == 0) & ~big, Y, self.ext)
        if sat is not None:                                  # 극값이 바뀐 픽셀은 포화 상태도 갱신
            self.sat_ext = np.where(self.ext == Y, sat, self.sat_ext)
        self.age = np.where(tr, 0, np.minimum(self.age + 1, 9999))
        self.hist[self.i % self.wf] = tr
        self.i += 1

    def counts(self):
        return self.hist.sum(axis=0)


class _PeatChannel:
    """한 채널(휘도 Y 또는 적색값)의 전환 수·FAIL·위험도. 전환 수용/플래시용 추적기 2개."""

    def __init__(self, window_frames, recent, count_thr, flash_thr, dark=None, area_thr=AREA_THRESHOLD):
        self.wf, self.recent, self.area_thr = window_frames, recent, area_thr
        self.count_thr, self.flash_thr, self.dark = count_thr, flash_thr, dark
        self.tc = self.tf = None

    def update(self, V, explained, sat=None):
        if self.tc is None:
            self.tc = _PixelExtremes(V.shape, self.count_thr, self.wf)
            self.tf = _PixelExtremes(V.shape, self.flash_thr, self.wf, self.dark)
        self.tc.step(V, explained, sat)
        self.tf.step(V, explained, sat)
        C = self.tc.counts()
        count = 0
        for k in range(1, int(C.max()) + 1):
            if _box_area_map(C >= k).max() > self.area_thr + 1e-6:
                count = k
            else:
                break
        recent = self.tf.age <= self.recent
        recent_area = float(_box_area_map(recent).max()) if recent.any() else 0.0
        fail = count >= FAIL_TRANSITIONS and recent_area > PEAT_RECENT_AREA
        risk = 0.0
        if count >= FAIL_TRANSITIONS - 2 and recent.any():
            level = min(count, FAIL_TRANSITIONS)
            risk = _risk_from_area(float(_box_area_map(recent & (C >= level)).max()))
            if not fail:
                risk = min(risk, 0.4 if count >= FAIL_TRANSITIONS - 1 else 0.3)
        return {"count": count, "fail": fail, "risk": risk, "area": recent_area}


def _risk_from_a7(a7):
    """1초 전환 ≥7 픽셀 면적(박스 비율) → FAIL 중 PEAT 휘도 위험도 (FPA 13개 맞춤)."""
    return float(np.clip(0.65 + 8.75 * max(a7 - AREA_THRESHOLD, 0.0) ** 1.2, RISK_FAIL, RISK_MAX))


def peat_lum_cd(frame_bgr):
    """BGR 프레임 → PEAT 기준 폭의 정수 휘도(cd/m², float32 배열)."""
    lin = np.rint(peat_rgb(frame_bgr) * PEAT_PEAK_CD).astype(np.int64)
    return ((13933 * lin[..., 2] + 46871 * lin[..., 1] + 4732 * lin[..., 0]) >> 16).astype(np.float32)


def _block_hist(L, b, bins):
    """블록(b×b)별 휘도 히스토그램 (by, bx, bins). 0~PEAT_PEAK_CD를 bins칸으로."""
    H, W = L.shape
    by, bx = H // b, W // b
    q = np.clip(np.rint(L[:by * b, :bx * b] / PEAT_PEAK_CD * (bins - 1)), 0, bins - 1).astype(np.int32)
    blk = q.reshape(by, b, bx, b).transpose(0, 2, 1, 3).reshape(by * bx, b * b)
    idx = blk + (np.arange(by * bx) * bins)[:, None]
    return np.bincount(idx.ravel(), minlength=by * bx * bins).reshape(by, bx, bins)


class _FrameDiffTransitions:
    """직전 프레임 대비 ±thr 초과 변화 + 블록 플래그 + 방향 교대 → 전환, 최근 1초 이력.
    1초 창 전환이 0이 되면 기대 방향을 초기화한다."""

    def __init__(self, shape, thr, window_frames, dark=None):
        self.thr, self.dark, self.wf = thr, dark, window_frames
        self.expect = np.zeros(shape, np.int8)   # 다음에 셀 방향: +1 밝아짐, −1 어두워짐, 0 아무쪽
        self.hist = np.zeros((window_frames,) + shape, bool)
        self.age = np.full(shape, 9999, np.int32)
        self.i = 0

    def step(self, L, P, flag):
        d = L - P
        ok = flag if self.dark is None else flag & (np.minimum(L, P) < self.dark)
        up = (d > self.thr) & ok & (self.expect >= 0)
        dn = (d < -self.thr) & ok & (self.expect <= 0)
        tr = up | dn
        self.expect = np.where(up, -1, np.where(dn, 1, self.expect)).astype(np.int8)
        self.age = np.where(tr, 0, np.minimum(self.age + 1, 9999))
        self.hist[self.i % self.wf] = tr
        self.i += 1
        self.expect = np.where(self.hist.any(axis=0), self.expect, 0).astype(np.int8)

    def counts(self):
        return self.hist.sum(axis=0)


class _HistLumChannel:
    """휘도 채널: 블록 히스토그램 플래그 + 프레임차 전환. 반환 dict는 _PeatChannel과 같다."""

    def __init__(self, window_frames, recent, area_thr=AREA_THRESHOLD):
        self.wf, self.recent, self.area_thr = window_frames, recent, area_thr
        self.prev = None

    def update(self, L):
        b = PEAT_HIST_BLOCK
        H, W = L.shape
        by, bx = H // b, W // b
        if self.prev is None:
            self.tc = _FrameDiffTransitions(L.shape, PEAT_LUM_COUNT_CD, self.wf)
            self.tf = _FrameDiffTransitions(L.shape, PEAT_LUM_FLASH_CD, self.wf, PEAT_DARK_CD)
            flag, P = np.zeros(L.shape, bool), L
        else:
            D = np.abs(_block_hist(L, b, PEAT_HIST_BINS) - _block_hist(self.prev, b, PEAT_HIST_BINS)).sum(-1)
            flag = np.zeros(L.shape, bool)
            flag[:by * b, :bx * b] = np.repeat(np.repeat(D > round(PEAT_HIST_THR * b * b), b, 0), b, 1)
            P = self.prev
        self.tc.step(L, P, flag)
        self.tf.step(L, P, flag)
        self.prev = L

        C = self.tc.counts()
        count = 0
        for k in range(1, int(C.max()) + 1):
            if _box_area_map(C >= k, PEAT_LUM_BOX).max() > self.area_thr + 1e-6:
                count = k
            else:
                break
        recent = self.tf.age < self.recent
        recent_area = float(_box_area_map(recent, PEAT_LUM_BOX).max()) if recent.any() else 0.0
        fail = count >= FAIL_TRANSITIONS and recent_area > PEAT_RECENT_AREA
        risk = 0.0
        if fail:
            # FAIL 중 위험도 = 1초 전환 ≥7 픽셀 면적의 함수(FPA 564프레임 맞춤, 평균오차 0.33).
            # '최근 7프레임' 면적은 프레임마다 출렁여 PEAT 점선의 계단·평탄 모양과 달랐다.
            a7 = float(_box_area_map(C >= FAIL_TRANSITIONS, PEAT_LUM_BOX).max())
            risk = _risk_from_a7(a7)
        elif count >= FAIL_TRANSITIONS - 2 and recent.any():
            level = min(count, FAIL_TRANSITIONS)
            risk = _risk_from_area(float(_box_area_map(recent & (C >= level), PEAT_LUM_BOX).max()))
            risk = min(risk, 0.4 if count >= FAIL_TRANSITIONS - 1 else 0.3)
        return {"count": count, "fail": fail, "risk": risk, "area": recent_area}


class PeatEngine:
    """프레임마다 update() → (휘도 dict, 적색 dict), 각 dict(count, fail, risk, area)."""

    def __init__(self, window_frames, fps, area_thr=AREA_THRESHOLD):
        recent = max(1, int(round(PEAT_RECENT_S * fps)))
        r = PEAT_MOTION_R
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * r + 1, 2 * r + 1))
        self.lum = _HistLumChannel(window_frames, recent, area_thr)
        self.red = _PeatChannel(window_frames, recent, PEAT_RED_COUNT, PEAT_RED_FLASH, area_thr=area_thr)
        self.prev_Y = self.prev_R = None
        self.sat = None

    def _explained(self, V, P):
        k = self.kernel
        return ((V >= cv2.erode(P, k)) & (V <= cv2.dilate(P, k))
                & (P >= cv2.erode(V, k)) & (P <= cv2.dilate(V, k)))

    def update(self, frame_bgr):
        lin = peat_rgb(frame_bgr)
        B, G, R = lin[..., 0], lin[..., 1], lin[..., 2]
        Y = 0.0722 * B + 0.7152 * G + 0.2126 * R
        Rv = np.maximum(R - G - B, 0.0) * 320.0                # WCAG 적색값 (R−G−B)×320
        self.sat = R >= PEAT_RED_SAT * (R + G + B + 1e-6)      # 포화 적색
        first = self.prev_Y is None
        ex_r = np.zeros(Y.shape, bool) if first else self._explained(Rv, self.prev_R)
        self.prev_Y, self.prev_R = Y, Rv
        return self.lum.update(peat_lum_cd(frame_bgr)), self.red.update(Rv, ex_r, self.sat)


# ──────────────────────────────────────────────────────────────────────────
# 색/휘도 변환
# ──────────────────────────────────────────────────────────────────────────
def bgr_to_rgb01(img_bgr):
    return img_bgr[..., ::-1].astype(np.float32) / 255.0


def linearize_video(c):
    """영상 신호 → 선형 광량(BT.1886 감마 2.4, ITU-R BT.1702-3 Table 1). 줄무늬 패턴 전용 —
    휘도·적색 판정은 PEAT 실측대로 sRGB(`peat_rgb`)를 쓴다."""
    return np.power(c, DISPLAY_GAMMA)


def luminance_cd(rgb_lin):
    """선형 RGB → 절대 휘도(cd/m²). Y_rel × DISPLAY_PEAK_CD"""
    Y = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    return Y * DISPLAY_PEAK_CD


def _stripe_axis(profile):
    """
    1D 휘도 프로파일(cd/m²)에서 지배적 줄무늬 주파수를 FFT로 검출.
    반환: (광-암 쌍 수, 광-암 휘도차 cd/m², 가장 밝은 줄무늬 cd/m²)
    """
    n = len(profile)
    if n < 12:
        return 0, 0.0, 0.0
    sig = profile - profile.mean()
    spec = np.abs(np.fft.rfft(sig))
    # DC·초저주파(전역 그라데이션, <3 사이클) 제외하고 지배 주파수를 3 사이클부터
    # 탐색. 유해 여부의 최종 기준은 detect_stripe_pattern의 pairs > PATTERN_MIN_PAIRS(5).
    if len(spec) <= 6:
        return 0, 0.0, 0.0
    k = int(np.argmax(spec[3:]) + 3)  # 지배 주파수 인덱스 = 사이클 수
    amp = 2.0 * spec[k] / n           # 사인 진폭 추정
    light_dark_diff = 2.0 * amp       # 광-암 휘도차
    lightest = float(profile.mean() + amp)
    return k, float(light_dark_diff), lightest


def detect_stripe_pattern(L_cd, moving):
    """
    프레임 휘도(cd/m²)에서 유해 줄무늬 패턴 여부 판정.
    수평/수직 두 축의 프로파일을 FFT로 분석한다. (근사 — 대각/방사 패턴 미포함)
    """
    row_profile = L_cd.mean(axis=1)  # 수평 줄무늬 → 세로축 변화
    col_profile = L_cd.mean(axis=0)  # 수직 줄무늬 → 가로축 변화
    area_thresh = PATTERN_AREA_MOVING if moving else PATTERN_AREA_STATIC

    best_pairs = 0
    for profile in (row_profile, col_profile):
        pairs, diff, lightest = _stripe_axis(profile)
        harmful = (
            pairs > PATTERN_MIN_PAIRS
            and diff >= FLASH_DELTA_CD
            and lightest > PATTERN_LIGHT_CD
        )
        if harmful:
            best_pairs = max(best_pairs, pairs)

    # 면적: 프로파일 기반 전역 검출이므로 화면 대부분을 차지한다고 보고 1.0 근사.
    # (블록 단위 정밀 면적 산출은 TODO — 현재는 area_thresh 통과로 처리)
    is_harmful = best_pairs > 0 and 1.0 >= area_thresh
    return is_harmful, best_pairs


# ──────────────────────────────────────────────────────────────────────────
# 메인 분석
# ──────────────────────────────────────────────────────────────────────────
def analyze_video(video_path, area_threshold=AREA_THRESHOLD,
                  window_seconds=WINDOW_SECONDS, downsample=1, max_frames=0,
                  enable_pattern=True, on_frame=None, should_stop=None):
    """비디오 전체를 분석하고 metrics(dict)를 반환한다. GUI/CLI 공용 단일 소스.

    on_frame(dict): 프레임 하나 처리할 때마다 호출 — 실시간 차트용
        (time/processed/total/lum_area/red_area/sat_area/lum_window_count/
         lum_window_raw_count/lum_risk/red_window_count/pattern_hit/red_opposing)
    should_stop() -> bool: 매 프레임 전에 호출, True면 중단하고 None 반환
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    effective_fps = fps / max(1, downsample)
    window_frames = max(1, int(round(window_seconds * effective_fps)))

    # 진행률 표시용 총 처리 프레임 수 (추정)
    total_proc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // max(1, downsample)
    if max_frames > 0:
        total_proc = min(total_proc, max_frames)

    # 누적 시계열
    times = []
    pattern_flags = []         # 유해 줄무늬 패턴 프레임
    sat_area_series = []       # 포화 적색 존재 면적(전환 무관 — 적색 활동선용)

    # 휘도·적색: PEAT식 픽셀 엔진
    engine = PeatEngine(window_frames, effective_fps, area_threshold)
    lum_count, lum_fail_flags, lum_risk, lum_area_series = [], [], [], []
    red_count, red_fail_flags, red_risk, red_area_series = [], [], [], []

    prev_L_cd = None

    processed = 0
    frame_idx = -1

    while True:
        if should_stop is not None and should_stop():
            cap.release()
            return None
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % downsample != 0:
            continue
        if max_frames > 0 and processed >= max_frames:
            break

        L_cd = luminance_cd(linearize_video(bgr_to_rgb01(frame)))   # 줄무늬 패턴용
        t = processed / effective_fps

        lum, red = engine.update(frame)
        for (cnt, fl, rk, ar), d in (((lum_count, lum_fail_flags, lum_risk, lum_area_series), lum),
                                     ((red_count, red_fail_flags, red_risk, red_area_series), red)):
            cnt.append(d["count"]); fl.append(d["fail"]); rk.append(d["risk"]); ar.append(d["area"])
        sat_area_series.append(float(_box_area_map(engine.sat).max()))

        # ── 공간 패턴(줄무늬) ──
        pattern_hit = False
        if enable_pattern:
            moving = prev_L_cd is not None and float(np.abs(L_cd - prev_L_cd).mean()) >= FLASH_DELTA_CD * 0.1
            pattern_hit, _ = detect_stripe_pattern(L_cd, moving)

        times.append(t)
        pattern_flags.append(pattern_hit)
        prev_L_cd = L_cd
        processed += 1

        if on_frame is not None:
            # 판정 카운트: 전환 수 ≥7이라도 최근 플래시가 없으면(FAIL 아님) 6으로 표시
            def judged(d):
                return d["count"] if (d["fail"] or d["count"] < FAIL_TRANSITIONS) else FAIL_TRANSITIONS - 1
            on_frame({
                "time": t, "processed": processed, "total": total_proc,
                "lum_area": lum["area"], "red_area": red["area"],
                "sat_area": sat_area_series[-1],
                "lum_window_count": judged(lum),
                "red_window_count": judged(red),
                "lum_window_raw_count": lum["count"],
                "red_window_raw_count": red["count"],
                "lum_act_count": float(judged(lum)),
                "red_act_count": float(judged(red)),
                "lum_risk": lum["risk"],
                "red_risk": red["risk"],
                "pattern_hit": pattern_hit,
            })

    cap.release()

    # ── 휘도·적색: 픽셀 엔진 결과 ──
    pattern_arr = np.array(pattern_flags, dtype=bool)

    def judged(count, fail):
        """판정 카운트: FAIL 프레임만 7 이상 — 전환 수가 남아도 최근 플래시가 없으면 6으로 클램프."""
        c = np.array(count, dtype=np.int32)
        return np.where(np.array(fail, dtype=bool) | (c < FAIL_TRANSITIONS), c,
                        FAIL_TRANSITIONS - 1).astype(np.int32)

    lum_window_raw = np.array(lum_count, dtype=np.int32)      # 진단 실선 = PEAT 휘도 횟수
    red_window_raw = np.array(red_count, dtype=np.int32)      # 진단 실선 = PEAT 적색 횟수
    lum_window = judged(lum_count, lum_fail_flags)
    red_window = judged(red_count, red_fail_flags)
    lum_risk_arr = np.array(lum_risk, dtype=np.float32)
    red_risk_arr = np.array(red_risk, dtype=np.float32)
    lum_window_act = lum_window.astype(np.float32)   # 점선은 위험도로 그림 — 호환용
    red_window_act = red_window.astype(np.float32)

    lum_fail = bool((lum_window >= FAIL_TRANSITIONS).any())
    red_fail = bool((red_window >= FAIL_TRANSITIONS).any())

    # 패턴: ≥0.5초 연속 지속 시 FAIL
    pattern_min_frames = max(1, int(round(PATTERN_MIN_SECONDS * effective_fps)))
    pattern_fail = _has_run(pattern_arr, pattern_min_frames)

    # 확장 플래시(Extended Flash): HardingFPA 설명서(How to interpret results Q19~21) —
    # '실패 직전의 경고 수준' 플래시가 최근 5초의 80% 이상 지속되면 발생. 실제 FAIL 중에는
    # 꺼지고, 진단선이 CAUTION 밴드 아래로 내려가도 꺼진다. 이전엔 FAIL 프레임까지 '활동'으로
    # 쳐서 연속 FAIL 구간(Knox 0:15·1:40)마다 파란선이 떴지만 PEAT엔 없음.
    # 경고 수준 = 3전환(≈2 flash) 이상: 2전환(1 flash)로 두면 일상적 장면 변화만으로도
    # 5초 80%를 채워 Knox 0:39·1:42, Ballie 40~50초에 남음. QA 1차 PEAT는 전 영상 Extended 없음.
    extended_frames = max(1, int(round(EXTENDED_SECONDS * effective_fps)))
    onethird = max(1, FAIL_TRANSITIONS // 3)  # 실패(7전환)의 1/3 ≈ 2전환 = 1 flash
    EXT_WARN_TRANSITIONS = FAIL_TRANSITIONS // 2  # 3
    flash_fail = (lum_window >= FAIL_TRANSITIONS) | (red_window >= FAIL_TRANSITIONS)
    flash_warn = ((lum_window >= EXT_WARN_TRANSITIONS) | (red_window >= EXT_WARN_TRANSITIONS)) & ~flash_fail
    diag_caution = (lum_window_raw >= onethird) | (red_window_raw >= onethird)
    extended_series = _rolling_mean(flash_warn.astype(np.float32), extended_frames)
    extended_series = np.where(flash_fail | ~diag_caution, 0.0, extended_series).astype(np.float32)
    extended_warn = bool((extended_series >= 0.8).any())

    # ── 4단계 판정 (공식 PEAT 밴드: PASS / CAUTION(PASS) / CAUTION(FAIL) / FAIL) ──
    lum_max = int(lum_window.max()) if len(lum_window) else 0
    red_max = int(red_window.max()) if len(red_window) else 0
    flash_max = max(lum_max, red_max)
    # WCAG 2.3.1은 플래시(휘도/적색)만 다룬다 — 줄무늬 패턴은 Harding/Ofcom 기준이므로
    # wcag_2_3_1_fail에서는 제외하고 종합 판정(verdict)에만 반영한다.
    wcag_fail = lum_fail or red_fail
    if wcag_fail or pattern_fail:
        verdict = "FAIL"
    elif extended_warn or flash_max >= FAIL_TRANSITIONS - 1:
        verdict = "CAUTION (FAIL)"          # pass/fail 라인 직전 — 위험 경고
    elif flash_max >= onethird:
        verdict = "CAUTION (PASS)"          # 실패의 1/3 이상 플래시 활동 = 경고
    else:
        verdict = "PASS"

    summary = {
        "fps": fps,
        "effective_fps": effective_fps,
        "frames_processed": processed,
        "display_peak_cd": DISPLAY_PEAK_CD,
        "luminance_fail": lum_fail,
        "red_fail": red_fail,
        "pattern_fail": pattern_fail,
        "extended_flash_warn": extended_warn,
        "flash_max_per_window": flash_max,
        "verdict": verdict,
        "wcag_2_3_1_fail": wcag_fail,
    }

    return {
        "times": np.array(times),
        "lum_area": np.array(lum_area_series, dtype=np.float32),
        "lum_risk": lum_risk_arr,
        "red_risk": red_risk_arr,
        "red_area": np.array(red_area_series),
        "sat_area": np.array(sat_area_series),
        "lum_window": lum_window,
        "red_window": red_window,
        "lum_window_raw": lum_window_raw,
        "red_window_raw": red_window_raw,
        "lum_window_act": lum_window_act,
        "red_window_act": red_window_act,
        "extended_series": extended_series,
        "pattern_flags": pattern_arr,
        "summary": summary,
    }


def _has_run(flags, min_len):
    """min_len 이상 연속 True 구간 존재 여부"""
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= min_len:
            return True
    return False


def _rolling_mean(x, win):
    """각 인덱스에서 직전 win개(현재 포함) 값의 평균. 길이 보존.
    윈도우가 채워지지 않은 초반 구간(i < win-1)은 0.0으로 처리하여
    extended flash 초반 과대평가를 방지한다."""
    out = np.zeros(len(x), dtype=np.float32)
    if len(x) >= win:
        csum = np.concatenate([[0.0], np.cumsum(x)])
        out[win - 1:] = (csum[win:] - csum[:-win]) / win
    return out


# ──────────────────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────────────────
# ── 차트 (PEAT 결과창 실측: wuwa 화면 캡처의 픽셀 위치를 FPA 값과 프레임별로 대조) ──
# 세로축은 위험도 단위 0~3.5 일직선 하나. 점선 = 위험도 그대로, 실선 = 전환 수 × 0.5/7
# (전환 1회 = 같은 간격, 전환 7회 = 위험도 0.5 = FAIL 기준과 같은 높이). 밴드 경계는
# 0.25(PASS) / 0.45(CAUTION(PASS), 검은 합격선) / 0.65(CAUTION(FAIL)), 그 위가 FAIL.
CHART_YMAX = 3.5
CHART_BANDS = (0.25, 0.45, 0.65)
CHART_PASS_LINE = 0.45


def count_to_chart_y(counts):
    """전환 수 → 차트 높이(위험도 단위). 7회 = 0.5."""
    return np.asarray(counts, dtype=np.float32) * (RISK_FAIL / FAIL_TRANSITIONS)


def risk_to_chart_y(risk):
    """위험도 → 차트 높이 (그대로, 0~RISK_MAX)."""
    return np.clip(np.asarray(risk, dtype=np.float32), 0.0, RISK_MAX)


def print_summary(metrics):
    s = metrics["summary"]
    print("=== PEAT (Harding/ITU-R BT.1702-3) 분석 결과 ===")
    for key in ("fps", "effective_fps", "frames_processed", "display_peak_cd"):
        print(f"  {key}: {s[key]}")
    print("  ─ 판정 ─")
    print(f"  휘도 플래시 FAIL:   {s['luminance_fail']}")
    print(f"  적색 플래시 FAIL:   {s['red_fail']}")
    print(f"  줄무늬 패턴 FAIL:   {s['pattern_fail']}")
    print(f"  확장 플래시 경고:    {s['extended_flash_warn']}")
    print(f"  1초내 최대 전환수:   {s['flash_max_per_window']} (FAIL 기준 {FAIL_TRANSITIONS})")
    print(f"  >>> 최종 등급: {s['verdict']}")


def plot_unified(metrics):
    """공식 PEAT 결과창 모양으로 그린다(GUI와 같은 규칙: 위험도 축 0~3.5, 점선 = 위험도,
    실선 = 전환 수×0.5/7, 밴드 0.25/0.45/0.65, 검은 합격선 0.45)."""
    t = metrics["times"]
    verdict = metrics["summary"]["verdict"]
    ext = metrics["extended_series"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#c9ccd0")
    lo = 0.0
    for lab, hi, shade in zip(["PASS", "CAUTION (PASS)", "CAUTION (FAIL)"], CHART_BANDS,
                              ["#c9ccd0", "#b3b7bb", "#b3b7bb"]):
        ax.axhspan(lo, hi, color=shade, zorder=0)
        ax.text((t[0] if len(t) else 0), (lo + hi) / 2, "  " + lab, va="center", ha="left",
                fontsize=8, color="#888", zorder=1)
        lo = hi
    ax.text((t[0] if len(t) else 0), CHART_BANDS[2] + 0.1, "  FAIL", va="center", ha="left",
            fontsize=8, color="#888", zorder=1)
    ax.axhline(CHART_PASS_LINE, color="black", linewidth=0.8, zorder=1)

    ax.plot(t, risk_to_chart_y(metrics["lum_risk"]), color="white", linestyle=":", linewidth=2.2,
            label="Luminance flash")
    ax.plot(t, risk_to_chart_y(metrics["red_risk"]), color="red", linestyle=":", linewidth=2.0,
            label="Red flash")
    ax.plot(t, np.where(ext >= 0.8, ext * CHART_BANDS[0], 0.0), color="blue", linewidth=1.2,
            label="Extended Flash")
    ax.plot(t, count_to_chart_y(metrics["lum_window_raw"]), color="white", linewidth=1.0,
            label="Lum flash diag")
    ax.plot(t, count_to_chart_y(metrics["red_window_raw"]), color="darkred", linewidth=1.0,
            label="Red flash diag")

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_ylim(0, CHART_YMAX)
    ax.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
    ax.set_title(f"PEAT (Harding / ITU-R BT.1702-3)   —   {verdict}")
    legend = ax.legend(loc="upper right", fontsize=7)
    legend.get_frame().set_facecolor("#c9ccd0")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="PEAT 충실 재구현 (Harding/ITU-R BT.1702-3)")
    ap.add_argument("--video", required=True, help="입력 비디오 경로")
    ap.add_argument("--area_threshold", type=float, default=AREA_THRESHOLD, help="10° 박스 플래시 면적 임계 (기본 0.25 초과)")
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS, help="슬라이딩 윈도우 초 (기본 1.0)")
    ap.add_argument("--downsample", type=int, default=1, help="N번째 프레임만 처리 (기본 1)")
    ap.add_argument("--max_frames", type=int, default=0, help="처리 프레임 제한 (0=무제한)")
    ap.add_argument("--no_pattern", action="store_true", help="줄무늬 패턴 분석 비활성화")
    ap.add_argument("--no_show", action="store_true", help="그래프 창 없이 요약만 출력")
    args = ap.parse_args()

    if args.downsample > 1:
        print(f"[!] --downsample {args.downsample}: 유효 fps가 1/{args.downsample}로 줄어 "
              f"그보다 빠른 점멸은 감지되지 않을 수 있습니다 (나이퀴스트 한계).")

    metrics = analyze_video(
        args.video,
        area_threshold=args.area_threshold,
        window_seconds=args.window,
        downsample=args.downsample,
        max_frames=args.max_frames,
        enable_pattern=not args.no_pattern,
    )
    print_summary(metrics)
    if not args.no_show:
        plot_unified(metrics)


if __name__ == "__main__":
    main()
