"""
engine.py — 광과민성 발작 분석 엔진 (PEAT fpaCore 수식 재구현)

PEAT 분석 코어(fpaCore.dll v1.19)의 디스어셈블리에서 읽은 수식("깜빡임 계산식 수학 정리.md")을
정수 연산까지 그대로 옮긴 휘도·적색 깜빡임 엔진. 줄무늬 패턴은 PEAT에서 비활성이라
Harding/Wilkins 기준으로 따로 구현했다. 이전 엔진(FPA 출력 역산)은 legacy.py에 보관.

출처:
  - WCAG 2.3.1: https://www.w3.org/WAI/WCAG21/Understanding/three-flashes-or-below-threshold.html
  - ITU-R BT.1702-3: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf
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

RISK_FAIL = 0.5           # PEAT 위험도 합격선 (점수 0.5 이상 FAIL)
RISK_MAX = 3.4            # PEAT 위험도 천장

_SRGB_LUT = np.arange(256, dtype=np.float32) / 255.0
_SRGB_LUT = np.where(_SRGB_LUT <= 0.04045, _SRGB_LUT / 12.92,
                     np.power((_SRGB_LUT + 0.055) / 1.055, 2.4)).astype(np.float32)


def _box_area_map(mask):
    """박스(10° = 화면 1/3×1/3)가 화면 안에 완전히 들어가는 위치별 면적 비율."""
    H, W = mask.shape
    kh, kw = max(1, H // 3), max(1, W // 3)
    loc = cv2.boxFilter(mask.astype(np.float32), -1, (kw, kh), normalize=True)
    y0, x0 = kh // 2, kw // 2
    return loc[y0:y0 + H - kh + 1, x0:x0 + W - kw + 1]


class _FrameDiffTransitions:
    """직전 프레임 대비 ±thr 초과 변화 + 블록 플래그 + 방향 교대 → 전환, 최근 1초 이력.
    1초 창 전환이 0이 되면 기대 방향을 초기화한다."""

    def __init__(self, shape, thr, window_frames):
        self.thr, self.wf = thr, window_frames
        self.expect = np.zeros(shape, np.int8)   # 다음에 셀 방향: +1 밝아짐, −1 어두워짐, 0 아무쪽
        self.hist = np.zeros((window_frames,) + shape, bool)
        self.count = np.zeros(shape, np.int16)  # 최근 1초 전환 수 (hist 합계를 누적으로 유지)
        self.i = 0

    def step(self, L, P, flag):
        d = L - P
        up = (d > self.thr) & flag & (self.expect >= 0)
        dn = (d < -self.thr) & flag & (self.expect <= 0)
        self.expect = np.where(up, -1, np.where(dn, 1, self.expect)).astype(np.int8)
        slot = self.i % self.wf
        tr = up | dn
        self.count += tr.astype(np.int16) - self.hist[slot]
        self.hist[slot] = tr
        self.i += 1
        self.expect[self.count == 0] = 0

    def counts(self):
        return self.count


# ──────────────────────────────────────────────────────────────────────────
# fpaCore 역어셈블 엔진 ("깜빡임 계산식 수학 정리.md", 2026-10-27 @wade)
# ──────────────────────────────────────────────────────────────────────────
# fpaCore.dll v1.19 디스어셈블리에서 읽은 식을 정수 연산까지 그대로 옮긴 것. 절 번호는 문서 기준.
#  - 축소(선형 휘도 면적 평균 → 8비트 재양자화), 휘도 = 채널별 정수 cd의 Rec.709 고정소수점 합.
#  - 블록(폭/8) 히스토그램 플래그 + ±6 cd 방향 교대 전환, 1초 창 전환 >6 & 9프레임 최소 <160 = 위험 픽셀.
#  - 위험 픽셀마다 9프레임 월시 대역 에너지 E → 강도 x → 면적별 포물선 f → 점수 round(f)/10.
#  - 패턴 판정은 원본에서 비활성(항상 0), 인터레이스(필드) 모드는 미구현.
# 문서와 다른 점 하나: 문서는 프레임을 96×72로 축소한다고 했지만, 그렇게 하면 FPA 13개에서
# PEAT FAIL 31구간 중 10개만 잡혔다(wuwa: PEAT 위험도 3.4인 장면의 위험 면적이 화면의 20%뿐).
# 기존 FPA 역산(10° 박스 = 288×216의 정확히 1/3 = 96×72)과 6.6절 '판정 창을 옮겨 가며'에 맞춰
# 288폭으로 분석하고 96×72를 판정 창으로 쓴다. 문서 그대로는 FPA_ANALYSIS_W = 96.
FPA_ANALYSIS_W = 288              # 분석 폭 (블록 한 변 b = 폭/8)
FPA_WIN_W, FPA_WIN_H = 96, 72     # 판정 창 W
FPA_HIST_LUM = 0.36               # 블록 플래그 기준 × b²
FPA_HIST_RED = 0.44
FPA_TRANS_CD = 6                  # 전환 문턱 (cd/m², 고정)
FPA_COUNT_GT = 6                  # 위험 픽셀: 1초 전환 > 6 (설정값)
FPA_DARK_CD = 160                 # 위험 픽셀: 최근 9프레임 최소 휘도 < 160
FPA_HIST_FRAMES = 9
FPA_SWING_CD = 20                 # 9프레임 변동폭 문턱 (설정값)
FPA_AREA_PCT = 25.0               # 면적 곡선 경계 (설정값)
FPA_RED_SAT_PCT = 80              # 포화 적색 비율 (설정 +0x20)
# 강도 x의 ROI 넓이: 문서는 |ROI|(판정 창 96×72)지만 FPA의 u16@4 = 10·x 대조로 192×144(창 넓이 ×4)임을
# 확인(wuwa 5.53초 w2=172 ↔ 창 넓이로 나눈 x 68.9). 창 넓이로 나누면 작은 면적 곡선이 상한 0.4에 붙었다.
FPA_X_ROI = 192 * 144
FPA_X_LUM = 1.5 * 64 / 9 * 25344 / 1e6 / FPA_X_ROI
FPA_X_RED = 64 / 9 * 25344 / 1e6 / FPA_X_ROI
_FPA_TAPS = np.array([2, 3, 4, 5, 6, 7, 6, 5], np.int64)
_FPA_WALSH = np.array([          # W1..W11 (6.3절 표), 행 = 계수, 열 = a1..a8
    [+1, +1, +1, +1, -1, -1, -1, -1],
    [+1, +1, -1, -1, -1, -1, +1, +1],
    [+1, +1, -1, -1, +1, +1, -1, -1],
    [+1, -1, -1, +1, +1, -1, -1, +1],
    [+1, -1, -1, +1, -1, +1, +1, -1],
    [+1, -1, +1, -1, -1, +1, -1, +1],
    [+1, -1, +1, -1, +1, -1, +1, -1],
    [+1, +1, +1, -1, -1, -1, -1, +1],
    [+1, -1, -1, -1, -1, +1, +1, +1],
    [-1, -1, +1, -1, +1, +1, -1, +1],
    [-1, +1, -1, -1, +1, -1, +1, +1],
], np.int64)
_FPA_CD = np.rint(_SRGB_LUT.astype(np.float64) * 200.0).astype(np.int64)    # L(p), 짝수 반올림
_FPA_CD_F32 = _FPA_CD.astype(np.float32).reshape(256, 1)   # cv2.LUT용
_v = np.arange(201, dtype=np.float64) / 200.0
_FPA_INV = np.rint(255.0 * np.where(_v <= 0.0031308, 12.92 * _v,
                                    1.055 * np.power(_v, 1 / 2.4) - 0.055)).astype(np.int64)   # L⁻¹(cd)
del _v


def _trunc_div(a, b):
    """C 정수 나눗셈(0 쪽 버림)."""
    return np.trunc(a / b).astype(np.int64)


def fpa_channels_cd(frame_bgr, width=FPA_ANALYSIS_W):
    """BGR 프레임 → 분석 폭의 채널별 정수 cd (L(R), L(G), L(B)). 1절의 이중 양자화 축소."""
    H, W = frame_bgr.shape[:2]
    size = (width, max(3, int(round(H * width / W))))
    avg = cv2.resize(cv2.LUT(frame_bgr, _FPA_CD_F32), size, interpolation=cv2.INTER_AREA)   # 세 채널 한 번에
    L = _FPA_CD[_FPA_INV[np.clip(np.rint(avg), 0, 200).astype(np.int64)]]
    return L[..., 2], L[..., 1], L[..., 0]


def _fpa_block_flag(V, Vp, thr_frac):
    """3절: 블록별 10 cd 간격 61칸 히스토그램 L1 차 > round(thr_frac·b²)인 블록의 픽셀 마스크."""
    H, W = V.shape
    b = W // 8
    by, bx = H // b, W // b

    def hist(S):
        q = np.clip(S[:by * b, :bx * b] // 10, 0, 60)
        blk = q.reshape(by, b, bx, b).transpose(0, 2, 1, 3).reshape(by * bx, b * b)
        idx = blk + (np.arange(by * bx) * 61)[:, None]
        return np.bincount(idx.ravel(), minlength=by * bx * 61).reshape(by, bx, 61)

    D = np.abs(hist(V) - hist(Vp)).sum(-1)
    flag = np.zeros(V.shape, bool)
    flag[:by * b, :bx * b] = np.repeat(np.repeat(D > round(thr_frac * b * b), b, 0), b, 1)
    return flag


def _fpa_energy(hist9, mask):
    """6.1~6.5절: 9프레임 이력(9, H, W; 오래된 것부터)으로 픽셀 에너지 E. mask 밖은 0."""
    h = hist9[:, mask].astype(np.int64)                  # (9, n)
    E = np.zeros(mask.shape, np.int64)
    if h.shape[1] == 0:
        return E
    m = h.sum(0) // 9
    d = h - m
    ok = (d.max(0) - d.min(0)) >= FPA_SWING_CD
    a = _trunc_div(_FPA_TAPS[:, None] * d[1:] + 4, 8)  # (8, n) — h1 버림
    Wc = _FPA_WALSH @ a                                  # (11, n)
    A = np.abs(Wc)
    B = np.stack([
        _trunc_div(A[0] + A[1] + A[7] + A[8] + 4, 8),
        (A[2] + np.abs(Wc[3] + 2)) // 4,
        _trunc_div(A[4] + A[5] + A[9] + A[10] + 4, 8),
        np.abs(_trunc_div(Wc[6], 2)),
    ])
    band = np.arange(1, 5)[:, None]
    e = ((B >= 1) * band * (B + 50)).sum(0)
    E[mask] = np.where(ok, e, 0)
    return E


def fpa_risk_curve(x, a_pct, area_pct=FPA_AREA_PCT):
    """7절: 면적별 위험도 f (음수는 0). area_pct~2·area_pct 구간은 보간. 배열 입력 가능."""
    xs, xl = np.minimum(x, 8.3), np.minimum(x, 23.0)
    fS = ((23 - xs) * xs - 60) / 14
    fL = ((46 - xl) * xl - 120) / 12
    t = np.clip((np.asarray(a_pct, np.float64) - area_pct) / area_pct, 0.0, 1.0)
    return np.maximum(0.0, t * fL + (1 - t) * fS)


def _window_sums(M, kh, kw):
    """판정 창(kh×kw)이 화면 안에 완전히 들어가는 위치별 합 (정수 적분 영상)."""
    I = np.pad(M.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return I[kh:, kw:] - I[:-kh, kw:] - I[kh:, :-kw] + I[:-kh, :-kw]


class _FpaChannel:
    """휘도 또는 적색 한 채널. update(전환 신호, 직전 신호, 블록 플래그, 9프레임용 신호, 에너지 조건)."""

    def __init__(self, window_frames, x_coef, area_pct=FPA_AREA_PCT):
        self.wf, self.x_coef, self.area_pct = window_frames, x_coef, area_pct
        self.tr = None
        self.hist9 = []

    def push(self, H9):
        """9프레임 이력에만 추가 (전환 판정 없이)."""
        self.hist9.append(H9)
        if len(self.hist9) > FPA_HIST_FRAMES:
            self.hist9.pop(0)

    def update(self, V, P, flag, H9, sig9_ok=None):
        if self.tr is None:
            self.tr = _FrameDiffTransitions(V.shape, FPA_TRANS_CD, self.wf)
        self.tr.step(V, P, flag)
        self.push(H9)
        hist9 = np.stack(self.hist9)
        C = self.tr.counts()

        kh, kw = min(FPA_WIN_H, V.shape[0]), min(FPA_WIN_W, V.shape[1])
        roi = kh * kw
        D = (C > FPA_COUNT_GT) & (hist9.min(0) < FPA_DARK_CD)            # 5절
        nD = _window_sums(D, kh, kw)
        f, a_pct = 0.0, 100.0 * nD.max() / roi
        if D.any() and len(self.hist9) == FPA_HIST_FRAMES:
            mask = D if sig9_ok is None else D & sig9_ok
            S = _window_sums(_fpa_energy(hist9, mask), kh, kw)            # 6.6절
            fW = fpa_risk_curve(self.x_coef * S, 100.0 * nD / roi, self.area_pct)
            best = np.flatnonzero(fW == fW.max())                         # 동률이면 위험 픽셀 많은 창
            best = best[np.argmax(nD.ravel()[best])]
            f, a_pct = float(fW.ravel()[best]), 100.0 * nD.ravel()[best] / roi
        r = round(f)                                                      # 8절 (짝수 반올림)
        # 진단선: 창에서 h(k) > 25%인 최대 k (1~30). C≥k 면적은 k에 대해 단조 감소라 이분 탐색
        need = round(roi * self.area_pct / 100)
        lo, hi = 0, min(30, int(C.max()))
        while lo < hi:
            k = (lo + hi + 1) // 2
            if _window_sums(C >= k, kh, kw).max() > need:
                lo = k
            else:
                hi = k - 1
        count = lo
        return {"count": count, "fail": r >= 5, "warn": 1 <= r <= 4, "risk": r / 10.0,
                "area": a_pct / 100.0}


_FPA_IDLE = {"count": 0, "fail": False, "warn": False, "risk": 0.0, "area": 0.0}


class FpaCoreEngine:
    """프레임마다 update(frame) → (휘도 dict, 적색 dict).
    dict = count(진단 전환 수), fail, warn, risk(점수 R, 0~3.4), area(위험 면적 비율).
    area_thr: 면적 곡선 경계(설정값, 기본 0.25 = 25%)."""

    def __init__(self, window_frames, area_thr=AREA_THRESHOLD):
        self.lum = _FpaChannel(window_frames, FPA_X_LUM, 100.0 * area_thr)
        self.red = _FpaChannel(window_frames, FPA_X_RED, 100.0 * area_thr)
        self.prev = None
        self.sat_hist = []
        self.sat = None
        self.n = 0

    def update(self, frame_bgr):
        LR, LG, LB = fpa_channels_cd(frame_bgr)
        Y = (13933 * LR + 46871 * LG + 4732 * LB) >> 16                   # 1절
        diff = LR - LG - LB
        r9 = np.maximum(0, 3 * np.rint(1.6 * diff / 3)).astype(np.int64)   # 11절 9프레임 적색 신호
        q3 = 3 * np.maximum(0, _trunc_div(410 * diff + 128, 768))          # 11절 전환용 적색 신호
        sigma = (100 * LR) // (LR + LG + LB + 1)
        sat = sigma > FPA_RED_SAT_PCT
        self.sat = sat
        self.sat_hist.append(sat)
        if len(self.sat_hist) > FPA_HIST_FRAMES:
            self.sat_hist.pop(0)

        # 9프레임이 모이기 전에는 이력만 쌓고 분석하지 않는다. 첫 분석 프레임의 휘도 전환은
        # 직전 휘도를 0으로 놓고 판정(원본 버퍼 초기값) — pointplay 0.6~1.2초 위험도가 PEAT와 일치.
        self.n += 1
        if self.n < FPA_HIST_FRAMES:
            self.lum.push(Y)
            self.red.push(r9)
            self.prev = (Y, q3, sat)
            return dict(_FPA_IDLE), dict(_FPA_IDLE)
        Yp, qp, satp = self.prev
        fl_l = _fpa_block_flag(Y, Yp, FPA_HIST_LUM)
        fl_r = _fpa_block_flag(q3, qp, FPA_HIST_RED)
        if self.n == FPA_HIST_FRAMES:
            Yp = np.zeros_like(Y)
        self.prev = (Y, q3, sat)
        lum = self.lum.update(Y, Yp, fl_l, Y)
        red = self.red.update(q3, qp, fl_r & (sat | satp), r9, np.any(self.sat_hist, axis=0))
        return lum, red


def _preview_signals(frame_bgr):
    """원본 크기 프레임 → (휘도 cd, 적색 지표 cd, 포화 적색 마스크). 미리보기 강조용."""
    LR, LG, LB = (_FPA_CD[frame_bgr[..., c]] for c in (2, 1, 0))
    Y = (13933 * LR + 46871 * LG + 4732 * LB) >> 16
    Rex = np.maximum(0.0, 1.6 * (LR - LG - LB))
    sat = (100 * LR) // (LR + LG + LB + 1) > FPA_RED_SAT_PCT
    return Y, Rex, sat


def flash_mask(frame_bgr, prev_bgr):
    """두 프레임 사이 휘도 변화 ≥20 cd/m²이고 어두운 쪽 <160 cd/m²인 픽셀 — 미리보기 강조용."""
    (Y, _, _), (Yp, _, _) = _preview_signals(frame_bgr), _preview_signals(prev_bgr)
    return (np.abs(Y - Yp) >= FPA_SWING_CD) & (np.minimum(Y, Yp) < FPA_DARK_CD)


def red_mask(frame_bgr, prev_bgr):
    """두 프레임 사이 적색 지표 변화 ≥20이고 어느 한쪽이 포화 적색인 픽셀 — 미리보기 강조용."""
    (_, R, s), (_, Rp, sp) = _preview_signals(frame_bgr), _preview_signals(prev_bgr)
    return (np.abs(R - Rp) >= FPA_SWING_CD) & (s | sp)


# ──────────────────────────────────────────────────────────────────────────
# 색/휘도 변환
# ──────────────────────────────────────────────────────────────────────────
# 영상 신호 → 선형 광량(BT.1886 감마 2.4, ITU-R BT.1702-3 Table 1). 줄무늬 패턴 전용 —
# 휘도·적색 판정은 fpaCore 수식대로 sRGB(`fpa_channels_cd`)를 쓴다.
_GAMMA_LUT = np.power(np.arange(256, dtype=np.float32) / np.float32(255.0), DISPLAY_GAMMA).astype(np.float32).reshape(256, 1)


def luminance_cd(frame_bgr):
    """BGR 프레임 → 절대 휘도(cd/m²) = Rec.709 상대휘도 × DISPLAY_PEAK_CD (감마 2.4 표)."""
    lin = cv2.LUT(frame_bgr, _GAMMA_LUT)
    Y = 0.2126 * lin[..., 2] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 0]
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
                  enable_pattern=True, on_frame=None, should_stop=None, engine="fpacore"):
    """비디오 전체를 분석하고 metrics(dict)를 반환한다. GUI/CLI 공용 단일 소스.

    engine: "fpacore"(이 파일의 fpaCore 수식 엔진, 기본) 또는 "ex"(참조 코드 기반, engine_ex.py)

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

    # 휘도·적색 엔진
    if engine == "ex":
        from engine_ex import ExEngine          # 참조 코드 기반 (선택 시에만 불러옴)
        engine = ExEngine(effective_fps, area_threshold)
    else:
        engine = FpaCoreEngine(window_frames, area_threshold)
    lum_count, lum_fail_flags, lum_risk, lum_area_series = [], [], [], []
    red_count, red_fail_flags, red_risk, red_area_series = [], [], [], []
    warn_flags = []            # 휘도·적색 경고(점수 0.1~0.4) 프레임

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

        t = processed / effective_fps

        lum, red = engine.update(frame)
        for (cnt, fl, rk, ar), d in (((lum_count, lum_fail_flags, lum_risk, lum_area_series), lum),
                                     ((red_count, red_fail_flags, red_risk, red_area_series), red)):
            cnt.append(d["count"]); fl.append(d["fail"]); rk.append(d["risk"]); ar.append(d["area"])
        warn_flags.append(lum["warn"] or red["warn"])
        sat_area_series.append(float(_box_area_map(engine.sat).max()))

        # ── 공간 패턴(줄무늬) ──
        pattern_hit = False
        if enable_pattern:
            L_cd = luminance_cd(frame)
            moving = prev_L_cd is not None and float(np.abs(L_cd - prev_L_cd).mean()) >= FLASH_DELTA_CD * 0.1
            pattern_hit, _ = detect_stripe_pattern(L_cd, moving)
            prev_L_cd = L_cd

        times.append(t)
        pattern_flags.append(pattern_hit)
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

    # 확장 플래시(9절): 경고 프레임(둘 다 FAIL 아님)이 최근 5초 중 4초 초과 & 현재도 경고.
    w = np.array(warn_flags, bool) & ~np.array(lum_fail_flags, bool) & ~np.array(red_fail_flags, bool)
    M = max(1, int(round(EXTENDED_SECONDS * effective_fps)))
    csum = np.concatenate([[0], np.cumsum(w)])
    wsum = csum[1:] - csum[np.maximum(np.arange(len(w)) + 1 - M, 0)]
    extended_series = np.where((wsum > 0.8 * M) & w, wsum / M, 0.0).astype(np.float32)
    extended_warn = bool((extended_series >= 0.8).any())
    onethird = max(1, FAIL_TRANSITIONS // 3)  # 실패(7전환)의 1/3 ≈ 2전환 = 1 flash

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
    ap = argparse.ArgumentParser(description="광과민성 발작 분석 (PEAT fpaCore 수식 재구현)")
    ap.add_argument("--video", required=True, help="입력 비디오 경로")
    ap.add_argument("--area_threshold", type=float, default=AREA_THRESHOLD, help="판정 창 위험 면적 경계 (기본 0.25 = 25%%)")
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS, help="슬라이딩 윈도우 초 (기본 1.0)")
    ap.add_argument("--downsample", type=int, default=1, help="N번째 프레임만 처리 (기본 1)")
    ap.add_argument("--max_frames", type=int, default=0, help="처리 프레임 제한 (0=무제한)")
    ap.add_argument("--no_pattern", action="store_true", help="줄무늬 패턴 분석 비활성화")
    ap.add_argument("--no_show", action="store_true", help="그래프 창 없이 요약만 출력")
    ap.add_argument("--engine", choices=("fpacore", "ex"), default="fpacore",
                    help="fpacore: fpaCore 수식 엔진(기본) / ex: 참조 코드(flash_analyzer_ex) 기반")
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
        engine=args.engine,
    )
    print_summary(metrics)
    if not args.no_show:
        plot_unified(metrics)


if __name__ == "__main__":
    main()
