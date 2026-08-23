"""
worker.py — QThread 기반 PEAT 분석 워커.
프레임 단위로 분석 진행하며, 매 프레임(또는 N프레임) 결과를 시그널로 GUI에 전달.
"""

import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from peat import (
    DISPLAY_PEAK_CD, FLASH_DELTA_CD, DARK_BOUND_CD, MICHELSON_THRESH,
    AREA_THRESHOLD, RED_RATIO_THRESH, RED_UV_DIST_THRESH,
    WINDOW_SECONDS, FAIL_TRANSITIONS, PACE_SAFE_60HZ, PACE_SAFE_50HZ,
    EXTENDED_SECONDS, PATTERN_MIN_SECONDS,
    bgr_to_rgb01, linearize_srgb, luminance_cd, saturated_red_ratio,
    uv_prime, local_area_fraction, harmful_luminance_mask,
    block_sign_map, signs_reversed,
    detect_stripe_pattern, _has_run, _rolling_mean,
)
from collections import deque


class AnalysisWorker(QThread):
    """프레임 단위 PEAT 분석을 수행하는 워커 스레드.

    Signals:
        progress(int, int): (현재 프레임 인덱스, 전체 프레임 수)
        frame_result(dict): 매 프레임의 분석 결과 딕셔너리
        finished(dict): 최종 summary 딕셔너리
        error(str): 에러 메시지
    """

    progress = pyqtSignal(int, int)
    frame_result = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path, area_threshold=AREA_THRESHOLD,
                 window_seconds=WINDOW_SECONDS, downsample=1,
                 max_frames=0, enable_pattern=True, emit_every=1):
        super().__init__()
        self.video_path = video_path
        self.area_threshold = area_threshold
        self.window_seconds = window_seconds
        self.downsample = downsample
        self.max_frames = max_frames
        self.enable_pattern = enable_pattern
        self.emit_every = emit_every  # N프레임마다 시그널 전송
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._analyze()
        except Exception as e:
            self.error.emit(str(e))

    def _analyze(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"영상을 열 수 없습니다: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_frames > 0:
            total_frames = min(total_frames, self.max_frames * self.downsample)

        effective_fps = fps / max(1, self.downsample)
        pace_safe = PACE_SAFE_50HZ if fps <= 55 else PACE_SAFE_60HZ
        area_threshold = self.area_threshold
        window_frames = max(1, int(round(self.window_seconds * effective_fps)))

        # 시계열 누적
        times = []
        lum_area_series = []
        red_area_series = []
        lum_event_flags = []
        red_event_flags = []
        pattern_flags = []
        sat_area_series = []

        # 극값 추적 상태
        lum_dir = 0
        last_ext_L_mean = None
        last_ext_L_arr = None
        last_ext_sign = None      # 직전 극값을 만든 유해 변화의 방향 블록 지도
        last_lum_event_t = None

        red_dir = 0
        last_ext_uv = None
        last_ext_satmask = None
        last_red_event_t = None

        prev_L_cd = None
        consec_big_times = deque()
        processed = 0
        frame_idx = -1

        # 슬라이딩 윈도우용 deque
        lum_dq = deque()
        red_dq = deque()

        while True:
            if self._stop_flag:
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % self.downsample != 0:
                continue
            if self.max_frames > 0 and processed >= self.max_frames:
                break

            rgb_lin = linearize_srgb(bgr_to_rgb01(frame))
            L_cd = luminance_cd(rgb_lin)
            sat_ratio = saturated_red_ratio(rgb_lin)
            satmask = sat_ratio >= RED_RATIO_THRESH
            uv = uv_prime(rgb_lin)
            L_mean = float(L_cd.mean())
            t = processed / effective_fps

            # ── 첫 프레임 초기화 ──
            if last_ext_L_arr is None:
                times.append(t)
                lum_area_series.append(0.0)
                red_area_series.append(0.0)
                sat_area_series.append(local_area_fraction(satmask))
                lum_event_flags.append(False)
                red_event_flags.append(False)
                pattern_flags.append(False)
                last_ext_L_mean = L_mean
                last_ext_L_arr = L_cd.copy()
                last_ext_uv = uv.copy()
                last_ext_satmask = satmask.copy()
                prev_L_cd = L_cd

                lum_dq.append(False)
                red_dq.append(False)

                processed += 1
                self.progress.emit(processed, total_frames // max(1, self.downsample))
                continue

            # ── 휘도 플래시: 방향별 로컬 면적 판정 (peat.py와 동일 로직) ──
            harmful = harmful_luminance_mask(L_cd, last_ext_L_arr)
            up_area = local_area_fraction(harmful & (L_cd > last_ext_L_arr))
            down_area = local_area_fraction(harmful & (L_cd < last_ext_L_arr))
            lum_area = max(up_area, down_area)
            lum_delta = L_mean - last_ext_L_mean
            sign_now = block_sign_map(L_cd, last_ext_L_arr, harmful)
            if up_area >= area_threshold and down_area >= area_threshold:
                # 같은 영역이 실제 반전했을 때만 opposing — 모션/줌 제외 (peat.py 동일)
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

            lum_opposing = (
                lum_significant and lum_dir != 0
                and lum_cur_dir != 0 and lum_cur_dir != lum_dir
            )
            if lum_opposing and last_lum_event_t is not None and (t - last_lum_event_t) >= pace_safe:
                lum_counted = False
            else:
                lum_counted = lum_opposing

            # 장면 전환 제외
            consec_area = local_area_fraction(harmful_luminance_mask(L_cd, prev_L_cd))
            if consec_area > 0.7:
                consec_big_times.append(t)
            while consec_big_times and (t - consec_big_times[0]) > 1.0:
                consec_big_times.popleft()
            if consec_area > 0.7 and len(consec_big_times) <= 2:
                lum_counted = False

            if lum_opposing:
                last_ext_L_mean, last_ext_L_arr = L_mean, L_cd.copy()
                if np.abs(sign_now).max() > 0.05:
                    last_ext_sign = sign_now
                lum_dir = lum_cur_dir
                last_lum_event_t = t
            elif lum_cur_dir != 0 and (lum_dir == 0 or lum_cur_dir == lum_dir):
                last_ext_L_mean, last_ext_L_arr = L_mean, L_cd.copy()
                # 유해 변화가 있었을 때만 부호 지도 갱신 (peat.py 동일)
                if np.abs(sign_now).max() > 0.05:
                    last_ext_sign = sign_now
                if lum_dir == 0:
                    lum_dir = lum_cur_dir

            # ── 적색 플래시 ──
            uv_dist = np.sqrt(((uv - last_ext_uv) ** 2).sum(axis=-1))
            red_transition_mask = (satmask | last_ext_satmask) & (uv_dist > RED_UV_DIST_THRESH)
            red_area = local_area_fraction(red_transition_mask)

            curr_sat_area = float(satmask.mean())
            prev_sat_area = float(last_ext_satmask.mean())
            sat_area_delta = curr_sat_area - prev_sat_area
            if abs(sat_area_delta) > 0.02:
                red_cur_dir = 1 if sat_area_delta > 0 else -1
            elif red_area >= area_threshold:
                red_cur_dir = -red_dir if red_dir != 0 else 1
            else:
                red_cur_dir = 0
            red_significant = red_area >= area_threshold

            red_opposing = (
                red_significant and red_dir != 0
                and red_cur_dir != 0 and red_cur_dir != red_dir
            )
            if red_opposing and last_red_event_t is not None and (t - last_red_event_t) >= pace_safe:
                red_counted = False
            else:
                red_counted = red_opposing

            if red_opposing:
                last_ext_uv, last_ext_satmask = uv.copy(), satmask.copy()
                red_dir = red_cur_dir
                last_red_event_t = t
            elif red_cur_dir != 0 and (red_dir == 0 or red_cur_dir == red_dir):
                last_ext_uv, last_ext_satmask = uv.copy(), satmask.copy()
                if red_dir == 0:
                    red_dir = red_cur_dir

            # ── 공간 패턴 ──
            pattern_hit = False
            if self.enable_pattern:
                moving = float(np.abs(L_cd - prev_L_cd).mean()) >= FLASH_DELTA_CD * 0.1
                pattern_hit, _ = detect_stripe_pattern(L_cd, moving)

            # 누적
            times.append(t)
            lum_area_series.append(lum_area)
            red_area_series.append(red_area)
            sat_area_series.append(local_area_fraction(satmask))
            lum_event_flags.append(lum_counted)
            red_event_flags.append(red_counted)
            pattern_flags.append(pattern_hit)

            # 슬라이딩 윈도우 카운트 (실시간)
            lum_dq.append(lum_counted)
            if len(lum_dq) > window_frames:
                lum_dq.popleft()
            lum_window_count = sum(lum_dq)

            red_dq.append(red_counted)
            if len(red_dq) > window_frames:
                red_dq.popleft()
            red_window_count = sum(red_dq)

            prev_L_cd = L_cd
            processed += 1

            # 시그널 emit
            if processed % self.emit_every == 0:
                self.progress.emit(processed, total_frames // max(1, self.downsample))
                self.frame_result.emit({
                    "time": t,
                    "lum_area": lum_area,
                    "red_area": red_area,
                    "sat_area": local_area_fraction(satmask),
                    "lum_window_count": lum_window_count,
                    "red_window_count": red_window_count,
                    "pattern_hit": pattern_hit,
                    "lum_opposing": lum_counted,
                    "red_opposing": red_counted,
                })

        cap.release()

        # ── 최종 판정 ──
        lum_flags_arr = np.array(lum_event_flags, dtype=bool)
        red_flags_arr = np.array(red_event_flags, dtype=bool)
        pattern_arr = np.array(pattern_flags, dtype=bool)

        # 전체 윈도우 카운트 재계산
        def windowed_count(flags):
            out = np.zeros(len(flags), dtype=np.int32)
            dq = deque()
            for i, f in enumerate(flags):
                dq.append(f)
                if len(dq) > window_frames:
                    dq.popleft()
                out[i] = int(np.count_nonzero(dq))
            return out

        lum_window = windowed_count(lum_flags_arr)
        red_window = windowed_count(red_flags_arr)

        lum_fail = bool((lum_window >= FAIL_TRANSITIONS).any())
        red_fail = bool((red_window >= FAIL_TRANSITIONS).any())

        pattern_min_frames = max(1, int(round(PATTERN_MIN_SECONDS * effective_fps)))
        pattern_fail = _has_run(pattern_arr, pattern_min_frames)

        extended_frames = max(1, int(round(EXTENDED_SECONDS * effective_fps)))
        onethird = max(1, FAIL_TRANSITIONS // 3)
        flash_active = (lum_window >= onethird) | (red_window >= onethird)
        extended_series = _rolling_mean(flash_active.astype(np.float32), extended_frames)
        extended_warn = bool((extended_series >= 0.8).any())

        lum_max = int(lum_window.max()) if len(lum_window) else 0
        red_max = int(red_window.max()) if len(red_window) else 0
        flash_max = max(lum_max, red_max)
        wcag_fail = lum_fail or red_fail or pattern_fail

        if wcag_fail:
            verdict = "FAIL"
        elif extended_warn or flash_max >= FAIL_TRANSITIONS - 1:
            verdict = "CAUTION (FAIL)"
        elif flash_max >= onethird:
            verdict = "CAUTION (PASS)"
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
            "times": times,
            "lum_window": lum_window.tolist(),
            "red_window": red_window.tolist(),
            "extended_series": extended_series.tolist(),
            "sat_area_series": sat_area_series,
        }

        self.finished.emit(summary)
