import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def calculate_luminance(img):
    """Calculate relative luminance with sRGB weights."""
    r, g, b = img[..., 2], img[..., 1], img[..., 0]  # Convert BGR to RGB
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def calculate_flash_area_ratio(curr_luminance, prev_luminance, threshold):
    delta = np.abs(curr_luminance - prev_luminance) / 255.0
    flash_mask = delta >= threshold  # Boolean mask for significant changes
    return np.mean(flash_mask)  # Ratio of changed area

def calc_luminance_flash_y1(curr_frame, prev_luminance, lum_threshold):
    curr_luminance = calculate_luminance(curr_frame)
    flash_ratio = calculate_flash_area_ratio(curr_luminance, prev_luminance, lum_threshold)
    return curr_luminance, flash_ratio


def detect_luminance_flashes(video_path, lum_threshold=0.3, area_threshold=0.25, window_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    window_frames = int(window_seconds * fps)
    flash_ratios = []
    times = []
    event_flags = []
    event_window = deque(maxlen=window_frames)  # Track events in a sliding window

    # Read the first frame
    success, prev_frame = cap.read()
    if not success:
        raise RuntimeError("Failed to read the first frame.")

    prev_luminance = calculate_luminance(prev_frame)  # Calculate luminance of the first frame
    frame_index = 0

    while cap.isOpened():
        success, curr_frame = cap.read()
        if not success:
            break

        # Calculate luminance for the current frame
        curr_luminance, flash_ratio = calc_luminance_flash_y1(curr_frame, prev_luminance, lum_threshold)
        flash_ratios.append(flash_ratio)

        # Calculate the current time in seconds
        times.append(frame_index / fps)

        # Check if the flash ratio exceeds the area threshold
        event_flags.append(flash_ratio >= area_threshold)
        event_window.append(flash_ratio >= area_threshold)

        prev_luminance = curr_luminance  # Update the previous luminance
        frame_index += 1

    cap.release()

    return {
        "times": np.array(times),
        "flash_ratios": np.array(flash_ratios),
    }

def plot_luminance_flash_results(metrics, area_threshold):
    """
    Plot luminance flash area ratio and highlight detected events.
    """
    times = metrics["times"]
    flash_ratios = metrics["flash_ratios"]

    plt.figure(figsize=(12, 6))
    plt.plot(times, flash_ratios, color="blue", label="Luminance Flash Area Ratio")
    plt.axhline(y=area_threshold, color="red", linestyle="--", label=f"Area Threshold = {area_threshold}")

    plt.xlabel("Time (s)")
    plt.ylabel("Flash Area Ratio")
    plt.title("Luminance Flash Analysis (WCAG 2.3.1)")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Luminance Flash Detection")
    parser.add_argument("--video", type=str, default='./temp/PEAT_wuwa.avi', help="Path to the video file for analysis")
    parser.add_argument("--lum_threshold", type=float, default=0.3, help="Luminance change threshold")
    parser.add_argument("--area_threshold", type=float, default=0.25, help="Area threshold for flash detection")
    args = parser.parse_args()

    metrics = detect_luminance_flashes(
        video_path=args.video,
        lum_threshold=args.lum_threshold,
        area_threshold=args.area_threshold
    )

    plot_luminance_flash_results(metrics, area_threshold=args.area_threshold)

if __name__ == "__main__":
    main()
