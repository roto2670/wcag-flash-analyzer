"""
legacy.py — 이전 휘도·적색 엔진 (FPA 출력 역산, 2026-10-27~28). 보관용 — engine.py는 쓰지 않는다.

PEAT 결과 파일(.fpa)을 합성·실제 영상으로 역산해 맞춘 규칙. fpaCore 역어셈블 수식 엔진
(engine.FpaCoreEngine)으로 교체되기 전까지 기본 엔진이었다. PeatEngine은 FpaCoreEngine과
같은 인터페이스(update(frame) → (휘도 dict, 적색 dict))라 비교용으로 바로 쓸 수 있다.
"""

import cv2
import numpy as np

from engine import AREA_THRESHOLD, FAIL_TRANSITIONS, _SRGB_LUT


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
