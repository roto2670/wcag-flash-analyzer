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
DISPLAY_PEAK_CD = 200.0   # SDR 기준 백색 휘도 가정 (ITU). 상대휘도 1.0 = 200 cd/m²
DISPLAY_GAMMA = 2.4       # BT.1886 디스플레이 감마 (ITU-R BT.1702-3 Table 1)
FLASH_DELTA_CD = 20.0     # 유해 휘도 변화 임계 (어두운 상태 < 160 cd/m²일 때)
DARK_BOUND_CD = 160.0     # 이 이상 밝으면 Michelson 대비 기준으로 전환
# ※ 상수 커플링: 20/200=10%(변화), 160/200=0.8(어두운쪽 경계)이 WCAG의 상대휘도
#   기준과 정확히 대응한다. DISPLAY_PEAK_CD를 바꾸면 FLASH_DELTA_CD와
#   DARK_BOUND_CD도 같은 비율로 함께 조정해야 WCAG 등가가 유지된다.
MICHELSON_THRESH = 1.0 / 17.0  # 어두운 상태 ≥160 cd/m²일 때 유해 대비 (≈0.0588)
AREA_THRESHOLD = 0.25     # 플래시 면적: 전체 화면의 25% (ITU/Ofcom 기준)
RED_RATIO_THRESH = 0.80   # 포화 적색: R/(R+G+B) ≥ 0.8
RED_DELTA_THRESH = 20.0   # 적색 전환: (R−G−B)×320 변화 > 20 (WCAG 작업정의)
WINDOW_SECONDS = 1.0      # 슬라이딩 윈도우 (판정용)
FAIL_TRANSITIONS = 7      # 1초 내 방향전환 ≥7회 (= >3 flash = >6 transition)
PACE_SAFE_60HZ = 0.334    # leading edge 간격 ≥334ms면 안전 (60Hz 환경)
PACE_SAFE_50HZ = 0.360    # ≥360ms면 안전 (50Hz 환경)

EXTENDED_SECONDS = 5.0    # 5초 지속 플래시 경고

# 공간 패턴(줄무늬) 기준 — Harding/Wilkins
PATTERN_MIN_PAIRS = 5         # >5 광-암 쌍이면 위험 후보
PATTERN_LIGHT_CD = 50.0       # 가장 밝은 줄무늬 > 50 cd/m²
PATTERN_AREA_STATIC = 0.40    # 정지 패턴: 화면 >40%
PATTERN_AREA_MOVING = 0.25    # 움직임/점멸/반전 패턴: 화면 >25%
PATTERN_MIN_SECONDS = 0.5     # ≥0.5초 지속

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
#   포화 적색 R/(R+G+B)는 0.8보다 엄격(255,80,80=0.86 셈 / 255,95,95=0.81 안 셈 → 중간 0.84).
#   위험 세기 = 1.8×적색값 변화+229 (혼합색 포함 오차 1% 안), 면적 25% 초과·FAIL 종료는 휘도와 같음.
PEAT_RED_FLASH = 20.0
PEAT_RED_COUNT = 10.0
PEAT_RED_SAT = 0.84
RISK_FAIL = 0.5           # PEAT 위험도 합격선
RISK_MAX = 3.4            # PEAT 위험도 천장

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


def _box_area_map(mask):
    """10° 박스(화면 1/3×1/3)가 화면 안에 완전히 들어가는 위치별 면적 비율."""
    H, W = mask.shape
    kh, kw = max(1, H // 3), max(1, W // 3)
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

    def __init__(self, window_frames, recent, count_thr, flash_thr, dark=None):
        self.wf, self.recent = window_frames, recent
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
            if _box_area_map(C >= k).max() > AREA_THRESHOLD + 1e-6:
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


class PeatEngine:
    """프레임마다 update() → (휘도 dict, 적색 dict), 각 dict(count, fail, risk, area)."""

    def __init__(self, window_frames, fps):
        recent = max(1, int(round(PEAT_RECENT_S * fps)))
        r = PEAT_MOTION_R
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * r + 1, 2 * r + 1))
        self.lum = _PeatChannel(window_frames, recent, PEAT_COUNT_DY, PEAT_FLASH_DY, PEAT_DARK_Y)
        self.red = _PeatChannel(window_frames, recent, PEAT_RED_COUNT, PEAT_RED_FLASH)
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
        ex_y = np.zeros(Y.shape, bool) if first else self._explained(Y, self.prev_Y)
        ex_r = np.zeros(Y.shape, bool) if first else self._explained(Rv, self.prev_R)
        self.prev_Y, self.prev_R = Y, Rv
        return self.lum.update(Y, ex_y), self.red.update(Rv, ex_r, self.sat)


# ──────────────────────────────────────────────────────────────────────────
# 색/휘도 변환
# ──────────────────────────────────────────────────────────────────────────
def bgr_to_rgb01(img_bgr):
    return img_bgr[..., ::-1].astype(np.float32) / 255.0


def linearize_video(c):
    """영상 신호 → 선형 광량. ITU-R BT.1702-3 Table 1(휘도 환산표)은 SDR을 BT.1886
    디스플레이 감마 2.4로 환산한다(예: 신호 0.52 → 41.7 cd/m²). 웹용 sRGB(선형 토 +
    ≈2.2)는 암부를 밝게 환산해 어두운 장면의 밝기 차이를 과장한다 — QA 1차에서
    sRGB는 Super Speed 5.7초를 과민하게 잡았고 감마 2.4는 12개 영상 판정이 PEAT와 일치."""
    return np.power(c, DISPLAY_GAMMA)


def luminance_cd(rgb_lin):
    """선형 RGB → 절대 휘도(cd/m²). Y_rel × DISPLAY_PEAK_CD"""
    Y = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    return Y * DISPLAY_PEAK_CD


def saturated_red_ratio(rgb_lin):
    """포화 적색 비율 R/(R+G+B) (선형 RGB 기준, WCAG 작업정의)"""
    R, G, B = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    return R / (R + G + B + 1e-6)


def red_flash_value(rgb_lin):
    """WCAG 적색 플래시 작업정의(2.0 시대, PEAT 구현 기준)의 적색도: (R−G−B)×320,
    음수는 0. 두 상태 간 이 값의 변화가 RED_DELTA_THRESH(20)를 넘으면 적색 전환.

    색공간 검증(2026-08): W3C 원문이 "R, G, B values range from 0-1 as specified
    in 'relative luminance' definition"이라 명시 — 상대 휘도 정의의 R,G,B는
    선형화(감마 제거) 값이므로 이 함수가 rgb_lin(선형)을 받는 것이 표준 그대로다.
    ※ WCAG 2.1/2.2 Understanding의 2022년 '새 작업정의'는 CIE 1976 UCS 색차
    >0.2 방식이지만, 공식 PEAT는 ×320 방식을 구현하므로 PEAT 재현 목표상 이쪽을 따른다."""
    v = rgb_lin[..., 0] - rgb_lin[..., 1] - rgb_lin[..., 2]
    return np.clip(v, 0.0, None) * 320.0


def local_area_fraction(mask):
    """
    WCAG/PEAT '10도 시야각' 면적: 전체 화면이 아니라, 화면의 ~1/9 크기
    (가로·세로 1/3) 로컬 영역 안의 최대 깜빡임 밀도를 반환.
    341×256 @1024×768 ≈ 화면 (1/3, 1/3) 영역에 해당. boxFilter로 슬라이딩 평균.
    """
    H, W = mask.shape
    kh, kw = max(1, H // 3), max(1, W // 3)
    local = cv2.boxFilter(mask.astype(np.float32), -1, (kw, kh), normalize=True)
    # 박스가 화면 안에 완전히 들어가는 '유효' 중심만 취함 — 기본 border 반사가
    # 모서리 플래시를 거울상으로 복제해 밀도를 과대평가하는 것을 방지.
    y0, x0 = kh // 2, kw // 2
    return float(local[y0:y0 + H - kh + 1, x0:x0 + W - kw + 1].max())


def harmful_luminance_mask(L_a, L_b):
    """
    두 휘도 상태(cd/m²) 사이 변화가 유해 기준을 넘는 픽셀 마스크.
      - 어두운 쪽 < 160 cd/m²: 차이 ≥ 20 cd/m² (ITU-R BT.1702-3 / WCAG 동일)
      - 어두운 쪽 ≥ 160 cd/m²: Michelson 대비 ≥ 1/17 — **의도적 보수성(표준 초과)**.
        WCAG/ITU는 어두운 쪽이 상대휘도 0.8(=160 cd/m²) 이상이면 완전 면제하지만,
        고휘도 대비 점멸도 잡도록 Michelson 기준을 유지한다 (위양성 방향으로만 작용).
    """
    darker = np.minimum(L_a, L_b)
    brighter = np.maximum(L_a, L_b)
    diff = brighter - darker
    low_region = (darker < DARK_BOUND_CD) & (diff >= FLASH_DELTA_CD)
    michelson = diff / (brighter + darker + 1e-6)
    high_region = (darker >= DARK_BOUND_CD) & (michelson >= MICHELSON_THRESH)
    return low_region | high_region


# ──────────────────────────────────────────────────────────────────────────
# 공간 패턴(줄무늬) 분석 — Harding/Wilkins (근사 구현)
# ──────────────────────────────────────────────────────────────────────────
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
    pace_safe = PACE_SAFE_50HZ if fps <= 55 else PACE_SAFE_60HZ
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
    engine = PeatEngine(window_frames, effective_fps)
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
        "pace_safe_ms": pace_safe * 1000.0,
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
BAND_TOP_FRAC = 0.22  # 4단계 밴드가 차지하는 y축 하단 비율 — 공식 PEAT(ex.png)와 동일한 두께


def count_to_band_y(counts, band_top):
    """1초 윈도우 전환수 → 밴드 경계에 맞춘 y좌표 (구간별 선형 보간).

      0                    → 0/4·band_top   PASS 바닥
      FAIL_TRANSITIONS//3  → 1/4·band_top   PASS / CAUTION(PASS) 경계
      FAIL_TRANSITIONS-1   → 2/4·band_top   CAUTION(PASS) / CAUTION(FAIL) 경계
      FAIL_TRANSITIONS     → 3/4·band_top   CAUTION(FAIL) / FAIL 경계
      2×FAIL_TRANSITIONS   → 4/4·band_top   천장 (이상은 클리핑)

    자기 최댓값 정규화가 아니므로 선이 놓인 밴드가 그대로 그 시점의 등급이다.
    """
    onethird = max(1, FAIL_TRANSITIONS // 3)
    xp = [0.0, float(onethird), float(FAIL_TRANSITIONS - 1),
          float(FAIL_TRANSITIONS), float(FAIL_TRANSITIONS * 2)]
    fp = [0.0, 0.25 * band_top, 0.50 * band_top, 0.75 * band_top, band_top]
    return np.interp(np.asarray(counts, dtype=np.float32), xp, fp).astype(np.float32)


# FAIL 확정 구간의 점선 최소 높이 — PEAT는 딱 7회 FAIL(Lint 19초)도 합격선 위로 보이게 솟는다.
RISK_MIN_H = 0.10


def count_to_activity_y(counts, band_top, ymax, risk=None):
    """점선 = PEAT '위험도 선'(보고서 Lum 0.0~3.4). risk가 있으면 그 값으로 그린다:
      - 0.2 이하 바닥, 0.3/0.4(CAUTION)는 CAUTION 밴드 안, 0.5 이상(FAIL)은 FAIL 경계부터
        천장(3.4)까지 비례. 판정 카운트(counts)가 FAIL 미만이면 위험도도 FAIL 밴드에 올리지 않는다.
    risk가 없으면 횟수만으로 근사한다(적색)."""
    c = np.asarray(counts, dtype=np.float32)
    fail = float(FAIL_TRANSITIONS)
    line = 0.75 * band_top
    if risk is not None:
        r = np.asarray(risk, dtype=np.float32)
        below = np.interp(np.minimum(r, RISK_FAIL), [0.0, 0.2, 0.3, 0.4, RISK_FAIL],
                          [0.0, 0.0, 0.25 * band_top, 0.5 * band_top, line])
        h = (r - RISK_FAIL) / (RISK_MAX - RISK_FAIL)
    else:
        below = np.interp(c, [0.0, fail - 3, fail - 2, fail - 1, fail],
                          [0.0, 0.0, 0.25 * band_top, 0.5 * band_top, line])
        h = (c - fail) / 9.0
    h = np.clip(h, RISK_MIN_H, 1.0)
    above = line + h * (0.95 * ymax - line)
    return np.where(c < fail, below, above).astype(np.float32)


def print_summary(metrics):
    s = metrics["summary"]
    print("=== PEAT (Harding/ITU-R BT.1702-3) 분석 결과 ===")
    for key in ("fps", "effective_fps", "frames_processed", "display_peak_cd", "pace_safe_ms"):
        print(f"  {key}: {s[key]}")
    print("  ─ 판정 ─")
    print(f"  휘도 플래시 FAIL:   {s['luminance_fail']}")
    print(f"  적색 플래시 FAIL:   {s['red_fail']}")
    print(f"  줄무늬 패턴 FAIL:   {s['pattern_fail']}")
    print(f"  확장 플래시 경고:    {s['extended_flash_warn']}")
    print(f"  1초내 최대 전환수:   {s['flash_max_per_window']} (FAIL 기준 {FAIL_TRANSITIONS})")
    print(f"  >>> 최종 등급: {s['verdict']}")


def plot_unified(metrics):
    """공식 PEAT 결과창(ex.png) 모방. 공식 정의대로 시리즈 매핑:
      - Luminance/Red flash (점선) = 플래시 '활동'(면적의 1초 윈도우 강도) — 메인, 솟음
      - Lum/Red flash diag (실선) = 플래시 '카운트' — 보조
      - Extended Flash (파란) = 경고 발생(≥0.8) 시에만 표시
    플래시 선 높이는 자기 최댓값 정규화가 아니라 전환수 임계를 밴드 경계에 고정한 값이다.
    """
    t = metrics["times"]
    verdict = metrics["summary"]["verdict"]

    ext = metrics["extended_series"]
    ext_disp = np.where(ext >= 0.8, ext, 0.0)               # 경고 발생 시에만

    ymax = 10.0
    band_top = ymax * BAND_TOP_FRAC
    h = band_top / 4

    # 선 높이를 밴드에 맞춤: 자기 최댓값 정규화 대신 전환수 임계를 밴드 경계에 고정.
    # 점선(활동선) = 판정 카운트 — FAIL 초과 시 밴드를 뚫고 천장까지 (ex.png 모양).
    # 실선(diag) = 게이트(모션 제외) 전 원시 반전 후보의 '플래시 수'(전환÷2) —
    # 공식 PEAT diag처럼 낮은 활동까지 길게 이어지고 정점이 CAUTION(FAIL) 부근에 머묾.
    lum_act = count_to_activity_y(metrics["lum_window"], band_top, ymax,
                                  metrics.get("lum_risk"))
    red_act = count_to_activity_y(metrics["red_window"], band_top, ymax, metrics.get("red_risk"))
    lum_cnt = count_to_band_y(metrics["lum_window_raw"] / 2.0, band_top)
    red_cnt = count_to_band_y(metrics["red_window_raw"] / 2.0, band_top)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#cfd2d6")

    # 4단계 밴드(하단 얇게) + 좌측 레이블
    for i, (lab, sh) in enumerate(zip(
            ["PASS", "CAUTION (PASS)", "CAUTION (FAIL)", "FAIL"],
            ["#c8cbcf", "#bfc3c7", "#b6babe", "#adb1b5"])):
        ax.axhspan(i * h, (i + 1) * h, color=sh, zorder=0)
        ax.text((t[0] if len(t) else 0), i * h + h / 2, "  " + lab,
                va="center", ha="left", fontsize=8, color="#777", zorder=1)

    # 메인: 휘도 활동(점선) — ex.png에서 천장까지 솟는 'Luminance flash'
    ax.plot(t, lum_act, color="white", linestyle="--", linewidth=1.8, label="Luminance flash")
    # 하단: 적색 존재(점선) — 7~8초 표시
    ax.plot(t, red_act, color="red", linestyle="--", linewidth=1.6, label="Red flash")
    # Extended Flash — 경고 발생 시에만 (PEAT_wuwa는 미발생 → 안 보임)
    ax.plot(t, ext_disp * band_top, color="blue", linewidth=1.2, label="Extended Flash")
    # 하단: 카운트(실선) — Lum/Red flash diag
    ax.plot(t, lum_cnt, color="white", linewidth=1.0, label="Lum flash diag")
    ax.plot(t, red_cnt, color="darkred", linewidth=1.0, label="Red flash diag")

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_ylim(0, ymax)
    ax.set_xlim(t[0] if len(t) else 0, t[-1] if len(t) else 1)
    ax.set_title(f"PEAT (Harding / ITU-R BT.1702-3)   —   {verdict}")
    legend = ax.legend(loc="upper right", fontsize=7)
    legend.get_frame().set_facecolor("#cfd2d6")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="PEAT 충실 재구현 (Harding/ITU-R BT.1702-3)")
    ap.add_argument("--video", required=True, help="입력 비디오 경로")
    ap.add_argument("--area_threshold", type=float, default=AREA_THRESHOLD, help="플래시 면적 임계 (기본 0.25)")
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
