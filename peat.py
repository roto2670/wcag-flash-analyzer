"""
peat.py — Harding/ITU-R BT.1702-3 기반 광과민성 발작 분석 엔진 (PEAT 충실 재구현)

공식 PEAT 분석 코어는 Cambridge Research Systems의 Harding FPA 엔진을 라이선스한
독점 모듈이라 코드 복제가 불가능하다. 이 파일은 PEAT가 실제로 따르는 표준
(WCAG 2.3.1 + ITU-R BT.1702-3 + Harding/Wilkins 패턴 기준)을 직접 구현한 것이다.

핵심 구현:
  1. cd/m² 절대 휘도 기준: ≥20 cd/m² 변화(어두운쪽 <160), 그 이상은 Michelson >1/17
  2. 적색 플래시: WCAG 작업정의 — 포화적색 R/(R+G+B)≥0.8 + (R−G−B)×320 변화 >20
  3. 줄무늬(공간) 패턴 분석 추가
  4. 페이싱 면제(leading edge ≥334ms/60Hz, ≥360ms/50Hz)

출처:
  - WCAG 2.3.1: https://www.w3.org/WAI/WCAG21/Understanding/three-flashes-or-below-threshold.html
  - ITU-R BT.1702-3: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-3-202311-I!!PDF-E.pdf
  - PEAT 오픈 클론(참조): https://github.com/rakeeb-hossain/PEAT_V2
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ──────────────────────────────────────────────────────────────────────────
# 표준 상수 (ITU-R BT.1702-3 / WCAG 2.3.1 / Harding)
# ──────────────────────────────────────────────────────────────────────────
DISPLAY_PEAK_CD = 200.0   # SDR 기준 백색 휘도 가정 (ITU). 상대휘도 1.0 = 200 cd/m²
FLASH_DELTA_CD = 20.0     # 유해 휘도 변화 임계 (어두운 상태 < 160 cd/m²일 때)
DARK_BOUND_CD = 160.0     # 이 이상 밝으면 Michelson 대비 기준으로 전환
# ※ 상수 커플링: 20/200=10%(변화), 160/200=0.8(어두운쪽 경계)이 WCAG의 상대휘도
#   기준과 정확히 대응한다. DISPLAY_PEAK_CD를 바꾸면 FLASH_DELTA_CD와
#   DARK_BOUND_CD도 같은 비율로 함께 조정해야 WCAG 등가가 유지된다.
MICHELSON_THRESH = 1.0 / 17.0  # 어두운 상태 ≥160 cd/m²일 때 유해 대비 (≈0.0588)
AREA_THRESHOLD = 0.25     # 플래시 면적: 전체 화면의 25% (ITU/Ofcom 기준)
RED_RATIO_THRESH = 0.80   # 포화 적색: R/(R+G+B) ≥ 0.8
RED_DELTA_THRESH = 20.0   # 적색 전환: (R−G−B)×320 변화 > 20 (WCAG 작업정의)
WINDOW_SECONDS = 1.0      # 슬라이딩 윈도우
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
# 색/휘도 변환
# ──────────────────────────────────────────────────────────────────────────
def bgr_to_rgb01(img_bgr):
    return img_bgr[..., ::-1].astype(np.float32) / 255.0


def linearize_srgb(c):
    """sRGB 감마 → 선형 광량 (WCAG 2.1, 2021년 이후 breakpoint 0.04045)"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


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


def uv_prime(rgb_lin):
    """선형 sRGB(D65) → CIE 1976 UCS (u', v'). shape (...,2)"""
    R, G, B = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    X = 0.4124 * R + 0.3576 * G + 0.1805 * B
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Z = 0.0193 * R + 0.1192 * G + 0.9505 * B
    denom = X + 15.0 * Y + 3.0 * Z + 1e-6
    up = 4.0 * X / denom
    vp = 9.0 * Y / denom
    return np.stack([up, vp], axis=-1)


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


def block_sign_map(L_cd, ext_arr, harmful, grid=(24, 32)):
    """극값 대비 유해 변화의 방향(+1/-1) 블록 지도. WCAG의 '동일 영역'
    opposing 판정용 — 이전 지도와 부호 상관이 음수면 같은 영역이 반전한 것."""
    signed = np.zeros(L_cd.shape, np.float32)
    signed[harmful] = np.sign((L_cd - ext_arr)[harmful])
    return cv2.resize(signed, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)


def signs_reversed(s_now, s_prev):
    """두 부호 지도가 '같은 블록에서 반대 방향'이면 True (동일 영역 반전 = 점멸).
    모션/줌은 각 영역의 부호가 유지되어 상관이 양수 → False.

    겹치는 블록이 없으면(플래시 영역이 프레임 사이에 이동/점프한 경우) 각 지도의
    지배 부호로 판정: 점멸은 각 지도가 한 방향으로 치우치고 서로 반대다
    (예: A영역 소등 → B영역 점등, 교차 스트로브). 모션은 상승(전연)/하강(후연)
    블록이 한 지도 안에 섞여 지배 부호가 약하므로 여전히 False."""
    if s_prev is None:
        return False
    act_now = np.abs(s_now) > 0.05
    act_prev = np.abs(s_prev) > 0.05
    overlap = act_now & act_prev
    if overlap.any():
        return float((s_now * s_prev)[overlap].mean()) < -0.2
    if not act_now.any() or not act_prev.any():
        return False
    m_now = float(s_now[act_now].mean())
    m_prev = float(s_prev[act_prev].mean())
    return m_now * m_prev < -0.36  # 양쪽 지배 부호 |m|≈0.6 이상 & 서로 반대


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
        (time/processed/total/lum_area/red_area/sat_area/
         lum_window_count/red_window_count/pattern_hit/lum_opposing/red_opposing)
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
    lum_area_series = []       # 극값 대비 휘도 플래시 면적
    red_area_series = []       # 극값 대비 적색 플래시 면적
    lum_event_flags = []       # 휘도 방향전환(opposing) 발생 프레임
    red_event_flags = []       # 적색 방향전환 발생 프레임
    pattern_flags = []         # 유해 줄무늬 패턴 프레임

    lum_mag_series = []        # 휘도 변화량(diag, 그래프용 — ex.png 흰 실선)
    red_mag_series = []        # 적색 정도(diag, 그래프용 — ex.png 빨강 실선)
    sat_area_series = []       # 포화 적색 존재 면적(전환 무관 — 적색 활동선용)

    # 휘도 극값(peak/valley) 추적 상태 — 평균 휘도 기반
    lum_dir = 0
    last_ext_L_mean = None
    last_ext_L_arr = None
    last_ext_sign = None      # 직전 극값을 만든 유해 변화의 방향 블록 지도
    last_lum_event_t = None
    lum_pending_idx = None    # 페이싱 잠정 면제 이벤트의 flags 인덱스 (소급 카운트용)
    last_lum_sig_t = None     # 마지막 유의(면적≥임계) 변화 시각 — 버스트 시작 판정용
    lum_cut_pending = []      # 장면전환 잠정 제외 (flags 인덱스, 시각) — 반복 점멸 판명 시 복원

    # 적색 극값 추적 상태
    red_dir = 0
    last_ext_redval = None    # 극값 상태의 (R−G−B)×320 지도
    last_ext_satmask = None
    last_ext_red_sign = None  # 직전 적색 극값을 만든 변화의 방향 블록 지도
    last_red_event_t = None
    red_pending_idx = None    # 페이싱 잠정 면제 이벤트의 flags 인덱스 (소급 카운트용)
    last_red_sig_t = None     # 마지막 유의 적색 변화 시각 — 버스트 시작 판정용

    prev_L_cd = None
    consec_big_times = deque()  # 직전 프레임 대비 화면 급변(장면전환 후보) 시각들
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

        rgb_lin = linearize_srgb(bgr_to_rgb01(frame))
        L_cd = luminance_cd(rgb_lin)
        sat_ratio = saturated_red_ratio(rgb_lin)
        satmask = sat_ratio >= RED_RATIO_THRESH
        red_val = red_flash_value(rgb_lin)
        L_mean = float(L_cd.mean())
        t = processed / effective_fps

        # diag(그래프)용 변화량 — main.py 방식, ex.png 실선과 동일 성격
        red_soft = np.clip(rgb_lin[..., 0] - np.maximum(rgb_lin[..., 1], rgb_lin[..., 2]), 0.0, 1.0)
        red_mag = float(red_soft.mean())

        # ── 첫 프레임 초기화 ──
        if last_ext_L_arr is None:
            times.append(t)
            lum_area_series.append(0.0)
            red_area_series.append(0.0)
            lum_mag_series.append(0.0)
            red_mag_series.append(red_mag)
            sat_area_series.append(local_area_fraction(satmask))
            lum_event_flags.append(False)
            red_event_flags.append(False)
            pattern_flags.append(False)
            last_ext_L_mean = L_mean
            last_ext_L_arr = L_cd.copy()
            last_ext_redval = red_val.copy()
            last_ext_satmask = satmask.copy()
            prev_L_cd = L_cd
            processed += 1
            if on_frame is not None:
                on_frame({
                    "time": t, "processed": processed, "total": total_proc,
                    "lum_area": 0.0, "red_area": 0.0,
                    "sat_area": sat_area_series[-1],
                    "lum_window_count": 0, "red_window_count": 0,
                    "pattern_hit": False,
                    "lum_opposing": False, "red_opposing": False,
                })
            continue

        # ── 휘도 플래시: 극값 대비 유해 픽셀의 '방향별' 로컬 면적(10° 시야각) ──
        # 방향까지 로컬 판정: WCAG는 플래시 '영역'의 opposing change 기준이므로
        # 전역 평균으로 방향을 재면 국소 플래시(화면 구석)를 놓친다.
        harmful = harmful_luminance_mask(L_cd, last_ext_L_arr)
        up_area = local_area_fraction(harmful & (L_cd > last_ext_L_arr))
        down_area = local_area_fraction(harmful & (L_cd < last_ext_L_arr))
        # WCAG: '동시 발생 플래시의 결합 면적'이 10° 시야각의 25% 초과 — 방향별
        # max가 아니라 결합(전체 유해) 면적으로 유의성 판정. 역위상 점멸
        # (A 밝아짐 + B 어두워짐 동시)은 방향별로는 25% 미만일 수 있다.
        lum_area = local_area_fraction(harmful)
        lum_delta = L_mean - last_ext_L_mean
        lum_mag = abs(lum_delta) / DISPLAY_PEAK_CD           # diag 변화량(0~1)
        sign_now = block_sign_map(L_cd, last_ext_L_arr, harmful)
        if (up_area >= area_threshold * 0.5 and down_area >= area_threshold * 0.5
                and lum_area >= area_threshold):
            # 양방향 동시 유의: '같은 영역이 실제 반전'(역위상 점멸)했을 때만 opposing.
            # 모션/줌은 up/down이 공존해도 각 영역의 부호가 유지되므로 제외.
            if signs_reversed(sign_now, last_ext_sign):
                lum_cur_dir = -lum_dir if lum_dir != 0 else 1
            else:
                lum_cur_dir = 1 if up_area >= down_area else -1
        elif up_area > 0.0 or down_area > 0.0:
            lum_cur_dir = 1 if up_area >= down_area else -1
        else:
            # 유해 픽셀 없음 → 전역 평균 부호로 극값 드리프트만 추적
            lum_cur_dir = 1 if lum_delta > 0 else (-1 if lum_delta < 0 else 0)
        lum_significant = lum_area >= area_threshold

        lum_opposing_raw = (
            lum_significant and lum_dir != 0
            and lum_cur_dir != 0 and lum_cur_dir != lum_dir
        )
        # WCAG '동일 영역의 opposing change': 지배 박스가 근소한 차이로 뒤바뀌는
        # 모션은 부호 지도가 유지되므로, 같은 영역이 실제 반전했을 때만 전환 인정.
        # 버스트 시작 보정: 1초 이상 유의 변화가 없다가 나타난 첫 유의 전환도 1회로
        # 카운트 — 반전만 세면 물리 전환 N회가 N−1회로 집계되는 오프바이원이 생겨
        # Harding 관행(1초 내 전환 ≥7회 FAIL)보다 한 단계 관대해진다.
        lum_burst_start = (
            lum_significant and lum_cur_dir != 0
            and (last_lum_sig_t is None or (t - last_lum_sig_t) > window_seconds)
        )
        lum_opposing = (lum_opposing_raw and signs_reversed(sign_now, last_ext_sign)) \
            or lum_burst_start
        if lum_significant:
            last_lum_sig_t = t
        lum_counted = lum_opposing

        # 페이싱 면제 (장면전환 검사보다 먼저): 앞뒤 간격이 모두 pace_safe 이상인
        # '고립' 이벤트만 안전 면제. WCAG는 1초 윈도우 내 플래시 수만 보므로 개별
        # 간격 면제는 저속 점멸에만 해당한다. 뒤 간격은 아직 모르므로 잠정 면제하고,
        # 다음 이벤트가 pace_safe 안에 오면 소급하여 카운트로 되돌린다.
        # ※ 순서 중요: 장면전환 소급 복원은 '페이싱을 통과한' 이벤트만 대상으로
        #   해야 저속 점멸이 복원 경로로 면제를 우회하지 않는다.
        if lum_counted:
            gap = None if last_lum_event_t is None else (t - last_lum_event_t)
            if gap is None or gap >= pace_safe:
                lum_counted = False
                lum_pending_idx = len(lum_event_flags)  # 이번 프레임이 받을 인덱스
            else:
                if lum_pending_idx is not None:
                    lum_event_flags[lum_pending_idx] = True
                lum_pending_idx = None

        # 장면 전환(scene cut) 제외: 직전 프레임 대비 화면 급변이 1초 내 '고립'(≤2회)이면
        # 컷으로 보고 잠정 제외. 급변이 1초 내 3회 이상으로 늘어나면 반복 점멸로
        # 판명되므로 잠정 제외분을 소급 복원한다 (버스트 초반 1~2 전환 누락 방지).
        consec_area = local_area_fraction(harmful_luminance_mask(L_cd, prev_L_cd))
        if consec_area > 0.7:
            consec_big_times.append(t)
        while consec_big_times and (t - consec_big_times[0]) > 1.0:
            consec_big_times.popleft()
        lum_cut_pending = [(i, tt) for (i, tt) in lum_cut_pending if (t - tt) <= 1.0]
        if consec_area > 0.7:
            if len(consec_big_times) <= 2:
                if lum_counted:
                    lum_cut_pending.append((len(lum_event_flags), t))
                    lum_counted = False
            else:
                for i, _tt in lum_cut_pending:
                    lum_event_flags[i] = True
                lum_cut_pending = []

        if lum_opposing:
            last_ext_L_mean, last_ext_L_arr = L_mean, L_cd.copy()
            if np.abs(sign_now).max() > 0.05:
                last_ext_sign = sign_now
            lum_dir = lum_cur_dir
            last_lum_event_t = t
        elif lum_opposing_raw:
            # 반전 후보였으나 동일 영역 반전 아님(모션) → 카운트 없이 방향/극값만 전환.
            # (상태를 갱신해야 다음 실제 반전을 놓치지 않는다)
            last_ext_L_mean, last_ext_L_arr = L_mean, L_cd.copy()
            if np.abs(sign_now).max() > 0.05:
                last_ext_sign = sign_now
            lum_dir = lum_cur_dir
        elif lum_cur_dir != 0 and (lum_dir == 0 or lum_cur_dir == lum_dir):
            last_ext_L_mean, last_ext_L_arr = L_mean, L_cd.copy()
            # 유해 변화가 있었을 때만 부호 지도 갱신 — 조용한 프레임이
            # 직전 점멸의 방향 기록을 지워버리지 않도록.
            if np.abs(sign_now).max() > 0.05:
                last_ext_sign = sign_now
            if lum_dir == 0:
                lum_dir = lum_cur_dir

        # ── 적색 플래시: WCAG 작업정의 ──
        # 한쪽 상태가 포화적색(R/(R+G+B)≥0.8)이고 (R−G−B)×320 변화 >20인 픽셀을
        # '적색으로 진입(+)/이탈(−)' 방향별 로컬 면적으로 판정 (휘도와 동일 구조).
        red_dv = red_val - last_ext_redval
        red_trans_mask = (satmask | last_ext_satmask) & (np.abs(red_dv) > RED_DELTA_THRESH)
        red_up_area = local_area_fraction(red_trans_mask & (red_dv > 0))
        red_down_area = local_area_fraction(red_trans_mask & (red_dv < 0))
        red_area = local_area_fraction(red_trans_mask)  # 결합 면적 (휘도와 동일)
        red_sign_now = block_sign_map(red_val, last_ext_redval, red_trans_mask)
        if (red_up_area >= area_threshold * 0.5 and red_down_area >= area_threshold * 0.5
                and red_area >= area_threshold):
            # 양방향 동시 유의: 같은 영역이 실제 반전했을 때만 opposing (휘도와 동일)
            if signs_reversed(red_sign_now, last_ext_red_sign):
                red_cur_dir = -red_dir if red_dir != 0 else 1
            else:
                red_cur_dir = 1 if red_up_area >= red_down_area else -1
        elif red_up_area > 0.0 or red_down_area > 0.0:
            red_cur_dir = 1 if red_up_area >= red_down_area else -1
        else:
            red_cur_dir = 0
        red_significant = red_area >= area_threshold

        red_opposing_raw = (
            red_significant and red_dir != 0
            and red_cur_dir != 0 and red_cur_dir != red_dir
        )
        # 휘도와 동일: 같은 영역이 실제 반전했을 때만 전환 인정 (모션 제외)
        # + 버스트 시작 보정(오프바이원): 1초 이상 조용하다 나타난 첫 유의 전환도 카운트
        red_burst_start = (
            red_significant and red_cur_dir != 0
            and (last_red_sig_t is None or (t - last_red_sig_t) > window_seconds)
        )
        red_opposing = (red_opposing_raw and signs_reversed(red_sign_now, last_ext_red_sign)) \
            or red_burst_start
        if red_significant:
            last_red_sig_t = t
        red_counted = red_opposing
        # 페이싱 면제: 휘도와 동일 — 고립 이벤트만 잠정 면제 + 소급 카운트
        if red_counted:
            gap = None if last_red_event_t is None else (t - last_red_event_t)
            if gap is None or gap >= pace_safe:
                red_counted = False
                red_pending_idx = len(red_event_flags)
            else:
                if red_pending_idx is not None:
                    red_event_flags[red_pending_idx] = True
                red_pending_idx = None

        if red_opposing:
            last_ext_redval, last_ext_satmask = red_val.copy(), satmask.copy()
            if np.abs(red_sign_now).max() > 0.05:
                last_ext_red_sign = red_sign_now
            red_dir = red_cur_dir
            last_red_event_t = t
        elif red_opposing_raw:
            # 동일 영역 반전 아님(모션) → 카운트 없이 방향/극값만 전환
            last_ext_redval, last_ext_satmask = red_val.copy(), satmask.copy()
            if np.abs(red_sign_now).max() > 0.05:
                last_ext_red_sign = red_sign_now
            red_dir = red_cur_dir
        elif red_cur_dir != 0 and (red_dir == 0 or red_cur_dir == red_dir):
            last_ext_redval, last_ext_satmask = red_val.copy(), satmask.copy()
            if np.abs(red_sign_now).max() > 0.05:
                last_ext_red_sign = red_sign_now
            if red_dir == 0:
                red_dir = red_cur_dir

        # ── 공간 패턴(줄무늬) ──
        pattern_hit = False
        if enable_pattern:
            moving = float(np.abs(L_cd - prev_L_cd).mean()) >= FLASH_DELTA_CD * 0.1
            pattern_hit, _ = detect_stripe_pattern(L_cd, moving)

        times.append(t)
        lum_area_series.append(lum_area)
        red_area_series.append(red_area)
        lum_mag_series.append(lum_mag)
        red_mag_series.append(red_mag)
        sat_area_series.append(local_area_fraction(satmask))
        lum_event_flags.append(lum_counted)
        red_event_flags.append(red_counted)
        pattern_flags.append(pattern_hit)

        prev_L_cd = L_cd
        processed += 1

        if on_frame is not None:
            # 실시간 1초 윈도우 카운트 — 페이싱 소급 카운트 반영 위해 flags 꼬리 합산
            on_frame({
                "time": t, "processed": processed, "total": total_proc,
                "lum_area": lum_area, "red_area": red_area,
                "sat_area": sat_area_series[-1],
                "lum_window_count": int(np.count_nonzero(lum_event_flags[-window_frames:])),
                "red_window_count": int(np.count_nonzero(red_event_flags[-window_frames:])),
                "pattern_hit": pattern_hit,
                "lum_opposing": lum_counted,
                "red_opposing": red_counted,
            })

    cap.release()

    # ── 1초 슬라이딩 윈도우 방향전환 카운트 ──
    lum_flags = np.array(lum_event_flags, dtype=bool)
    red_flags = np.array(red_event_flags, dtype=bool)
    pattern_arr = np.array(pattern_flags, dtype=bool)

    def windowed_count(flags):
        out = np.zeros(len(flags), dtype=np.int32)
        dq = deque()
        for i, f in enumerate(flags):
            dq.append(f)
            if len(dq) > window_frames:
                dq.popleft()
            out[i] = int(np.count_nonzero(dq))
        return out

    lum_window = windowed_count(lum_flags)
    red_window = windowed_count(red_flags)

    lum_fail = bool((lum_window >= FAIL_TRANSITIONS).any())
    red_fail = bool((red_window >= FAIL_TRANSITIONS).any())

    # 패턴: ≥0.5초 연속 지속 시 FAIL
    pattern_min_frames = max(1, int(round(PATTERN_MIN_SECONDS * effective_fps)))
    pattern_fail = _has_run(pattern_arr, pattern_min_frames)

    # 확장 플래시(Extended Flash): 5초 윈도우의 80% 프레임에서 실패의 1/3 이상 플래시
    # (PEAT User Guide 원문 정의). 시계열 = 최근 5초 내 flash_active 비율.
    extended_frames = max(1, int(round(EXTENDED_SECONDS * effective_fps)))
    onethird = max(1, FAIL_TRANSITIONS // 3)  # 실패(7전환)의 1/3 ≈ 2전환 = 1 flash
    flash_active = (lum_window >= onethird) | (red_window >= onethird)
    extended_series = _rolling_mean(flash_active.astype(np.float32), extended_frames)
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
        "lum_area": np.array(lum_area_series),
        "red_area": np.array(red_area_series),
        "lum_mag": np.array(lum_mag_series),
        "red_mag": np.array(red_mag_series),
        "sat_area": np.array(sat_area_series),
        "lum_window": lum_window,
        "red_window": red_window,
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
    if len(x) == 0:
        return x
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        if i < win - 1:
            # 윈도우 미채움: 아직 충분한 데이터 없음, 판정 보류
            out[i] = 0.0
        else:
            lo = i - win + 1
            out[i] = (csum[i + 1] - csum[lo]) / win
    return out


# ──────────────────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────────────────
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


def area_to_band_y(areas, band_top, area_threshold=AREA_THRESHOLD):
    """플래시 면적(0~1) → y좌표. 면적 임계가 첫 밴드 경계에 오도록 선형."""
    a = np.asarray(areas, dtype=np.float32)
    return np.clip(a / max(1e-9, area_threshold) * (0.25 * band_top),
                   0.0, band_top).astype(np.float32)


def smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def _rolling_max(x, win):
    """각 인덱스에서 직전 win개(현재 포함) 최댓값. 활동선을 매끄러운 봉우리로."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    out = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        out[i] = x[max(0, i - win + 1): i + 1].max()
    return out


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
    eff_fps = metrics["summary"]["effective_fps"]
    win = max(1, int(round(eff_fps)))                       # 1초 윈도우

    ext = metrics["extended_series"]
    ext_disp = np.where(ext >= 0.8, ext, 0.0)               # 경고 발생 시에만

    ymax = 10.0
    band_top = ymax * 0.30
    h = band_top / 4

    # 선 높이를 밴드에 맞춤: 자기 최댓값 정규화 대신 전환수 임계를 밴드 경계에 고정
    lum_act = count_to_band_y(metrics["lum_window"], band_top)
    red_act = area_to_band_y(_rolling_max(metrics["sat_area"], win), band_top)
    lum_cnt = count_to_band_y(metrics["lum_window"], band_top)      # 휘도 카운트(diag)
    red_cnt = count_to_band_y(metrics["red_window"], band_top)      # 적색 카운트(diag)

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
    ax.plot(t, lum_act, color="white", linestyle="--", linewidth=1.1, label="Luminance flash")
    # 하단: 적색 존재(점선) — 7~8초 표시
    ax.plot(t, red_act, color="red", linestyle="--", linewidth=1.0, label="Red flash")
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
