import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def rgb_from_bgr(img_bgr):
    return img_bgr[..., ::-1]

def to_float01(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def relative_luminance_srgb(img_rgb01):
    R = img_rgb01[..., 0]
    G = img_rgb01[..., 1]
    B = img_rgb01[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return Y

def red_mask(img_rgb01):
    R, G, B = img_rgb01[..., 0], img_rgb01[..., 1], img_rgb01[..., 2]
    red_dominance = R / (R + G + B + 1e-6)  # Zero division 방지
    return red_dominance >= 0.8

def analyze_video(video_path):
    lum_delta_threshold = 0.1
    area_threshold = 0.25
    window_seconds = 1.0
    downsample = 1 #처리할 프레임 수
    max_frames = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    prev_rgb = None
    times = []
    lum_flash_ratio = []
    red_flash_ratio = []
    lum_flash_diag = []
    red_flash_diag = []

    frame_indices = []
    processed = 0
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % downsample != 0:
            continue
        if max_frames > 0 and processed >= max_frames:
            break

        rgb = to_float01(rgb_from_bgr(frame))
        Y = relative_luminance_srgb(rgb)

        R = rgb[...,0]; G = rgb[...,1]; B = rgb[...,2]
        red_soft = np.clip(R - np.maximum(G, B), 0.0, 1.0)
        red_score = float(red_soft.mean())

        if prev_rgb is None: #0프레임 처리
            lum_flash_ratio.append(0.0)
            red_flash_ratio.append(0.0)
            lum_flash_diag.append(0.0)
            red_flash_diag.append(red_score)
            times.append(processed / fps * downsample)
            frame_indices.append(frame_idx)
            prev_rgb = rgb.copy()
            processed += 1
            continue

        prev_Y = relative_luminance_srgb(prev_rgb)

        deltaY = np.abs(Y - prev_Y)

        lum_mask = deltaY >= lum_delta_threshold
        lum_area_ratio = float(lum_mask.mean())

        prev_red = red_mask(prev_rgb)
        curr_red = red_mask(rgb)
        red_change = np.logical_xor(prev_red, curr_red) | (deltaY >= lum_delta_threshold)
        red_area_ratio = float(( (prev_red | curr_red) & red_change ).mean())

        avg_deltaY = float(deltaY.mean())

        lum_flash_ratio.append(lum_area_ratio)
        red_flash_ratio.append(red_area_ratio)
        lum_flash_diag.append(avg_deltaY)
        red_flash_diag.append(red_score)
        times.append(frame_idx / fps)
        frame_indices.append(frame_idx)

        prev_rgb = rgb.copy()
        processed += 1

    cap.release()

    event_flags = np.array([(l >= area_threshold) or (r >= area_threshold) for l, r in zip(lum_flash_ratio, red_flash_ratio)], dtype=bool)

    effective_fps = fps / max(1, downsample)
    window_frames = max(1, int(round(window_seconds * effective_fps)))
    event_count_series = np.zeros(len(event_flags), dtype=np.int32)
    dq = deque()
    for i, flag in enumerate(event_flags):
        dq.append(flag)
        if len(dq) > window_frames:
            dq.popleft()
        event_count_series[i] = int(np.count_nonzero(dq))

    extended_flash = np.clip((event_count_series.astype(np.float32) / 3.0), 0.0, 1.0)

    fail = bool((event_count_series >= 4).any())

    summary = {
        "fps": fps,
        "frames_processed": processed,
        "area_threshold": area_threshold,
        "lum_delta_threshold": lum_delta_threshold,
        "window_seconds": window_seconds,
        "wcag_2_3_1_fail": fail,
        "effective_fps": effective_fps
    }

    return {
        "times": np.array(times),
        "luminance_flash_ratio": np.array(lum_flash_ratio),
        "red_flash_ratio": np.array(red_flash_ratio),
        "extended_flash": extended_flash.astype(np.float32),
        "lum_flash_diag": np.array(lum_flash_diag),
        "red_flash_diag": np.array(red_flash_diag),
        "event_count_series": event_count_series,
        "event_flags": event_flags,
        "summary": summary,
    }

def smooth_curve(y, window_size=5):
    if len(y) < window_size:
        return y
    kernel = np.ones(window_size) / window_size
    return np.convolve(y, kernel, mode='same')

def plot_unified(metrics):
    t = metrics["times"]
    y1 = smooth_curve(metrics["luminance_flash_ratio"])
    y2 = smooth_curve(metrics["red_flash_ratio"])
    y3 = metrics["extended_flash"]
    y4 = smooth_curve(metrics["lum_flash_diag"])
    y5 = smooth_curve(metrics["red_flash_diag"])
    area_threshold = metrics["summary"]["area_threshold"]

    plt.figure(figsize=(12, 6))
    plt.gca().set_facecolor("lightgrey")
    plt.plot(t, y1, color='white', linestyle='--', label="Luminance Flash Ratio")
    plt.plot(t, y2, color='red', linestyle='--', label="Red Flash Ratio")
    plt.plot(t, y3, color='blue', linewidth=0.5, linestyle='-', label="Extended Flash")
    plt.plot(t, y4, color='white', linewidth=0.5, linestyle='-', label="Lum Flash Diag")
    plt.plot(t, y5, color='red', linewidth=0.5, linestyle='-', label="Red Flash Diag")

    # 검수 라인
    plt.axhline(y=area_threshold, color='black', linestyle='-', label="Area Threshold (25%)")
    plt.text(t[-1] * 0.95, area_threshold + 0.02, "Fail Zone", color='red', ha='right', va='bottom', fontsize=10, fontweight='bold')
    plt.text(t[-1] * 0.95, area_threshold - 0.02, "Pass Zone", color='green', ha='right', va='top', fontsize=10, fontweight='bold')

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.yticks([])
    plt.title("PEAT-like Flash Metrics (Unified Overlay)")
    legend = plt.legend()
    legend.get_frame().set_facecolor('lightgrey')
    # plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

# def print_summary(metrics):
#     s = metrics["summary"]
#     print("=== Summary (PEAT-like) ===")
#     for k, v in s.items():
#         print(f"{k}: {v}")
#     # Simple markers of fail spans
#     if s["wcag_2_3_1_fail"]:
#         print("Result: FAIL - >3 flash events within a 1-second window occurred.")
#     else:
#         print("Result: PASS - Three flashes or below threshold.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video file")
    # ap.add_argument("--lum_delta_threshold", type=float, default=0.10, help="Relative luminance change threshold (default 0.10)")
    # ap.add_argument("--area_threshold", type=float, default=0.25, help="Area fraction threshold (default 0.25)")
    # ap.add_argument("--window", type=float, default=1.0, help="Sliding window in seconds for flash counting (default 1.0)")
    # ap.add_argument("--downsample", type=int, default=1, help="Process every Nth frame to speed up (default 1)")
    # ap.add_argument("--max_frames", type=int, default=0, help="Limit frames for quick tests (0 means no limit)")
    args = ap.parse_args()

    metrics = analyze_video(
        args.video
        # lum_delta_threshold=args.lum_delta_threshold,
        # area_threshold=args.area_threshold,
        # window_seconds=args.window,
        # downsample=args.downsample,
        # max_frames=args.max_frames
    )
    # print_summary(metrics)
    plot_unified(metrics)

if __name__ == "__main__":
    main()
