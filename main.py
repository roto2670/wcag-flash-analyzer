import time
import yt_dlp
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    type = args.type
    url = args.url

    if type == 'youtube':
        download_youtube_video(url)
    return True

def plot_flash_graph(flash_areas, contrast_changes, fps, flash_frames_per_sec=None):
    time = [f"{int(i//fps)//60:02}:{int(i//fps)%60:02}" for i in range(len(flash_areas))]

    frames_per_sec = int(fps)
    seconds = len(contrast_changes) // frames_per_sec
    avg_contrast_per_sec = []
    times_sec = []

    for sec in range(seconds):
        start = sec * frames_per_sec
        end = start + frames_per_sec
        sec_values = contrast_changes[start:end]
        avg = sum(sec_values) / len(sec_values) if sec_values else 0
        avg_contrast_per_sec.append(avg)
        times_sec.append(f"{sec//60:02}:{sec%60:02}")

    plt.figure(figsize=(16, 9))

    plt.plot(times_sec, avg_contrast_per_sec, label="Avg Contrast Change", color='blue', alpha=0.7)
    plt.axhline(0.1, color='blue', linestyle='--', linewidth=0.5, label="Threshold (10%)")

    # 시각적 결과 범위 표시
    if flash_frames_per_sec:
        for i, flash_count in enumerate(flash_frames_per_sec):
            start = i
            end = i + 1
            if flash_count > 3:
                plt.axvspan(start, end, color='red', alpha=0.3, label='FAIL' if i == 0 else "")
            elif 1 <= flash_count <= 3:
                plt.axvspan(start, end, color='yellow', alpha=0.2, label='CAUTION' if i == 0 else "")
            else:
                plt.axvspan(start, end, color='green', alpha=0.1, label='PASS' if i == 0 else "")

    plt.xlabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.ylabel("Brightness Change (%)")
    plt.title("SC 2.3.1 Flash Analysis Over Time")
    from matplotlib.patches import Patch
    #범례 작업
    custom_legend = [
        Patch(facecolor='red', alpha=0.3, label='FAIL (Flash > 3)'),
        Patch(facecolor='yellow', alpha=0.2, label='CAUTION (1–3 flashes)'),
        Patch(facecolor='green', alpha=0.1, label='PASS (0 flashes)'),
        plt.Line2D([], [], color='blue', alpha=0.7, label='Contrast Change'),
        plt.Line2D([], [], color='blue', linestyle='--', linewidth=0.5, label='Threshold (10%)')
    ]
    plt.legend(handles=custom_legend)
    plt.grid(True)
    plt.tight_layout()
    # 분석 결과 문구 표시
    result_text = "PASS" if not any(f > 3 for f in flash_frames_per_sec) else "FAIL"
    plt.text(0.5, 1.05, result_text, transform=plt.gca().transAxes,
             fontsize=16, fontweight='bold', color='red' if "FAIL" in result_text else 'green',
             ha='center')
    plt.show()

def analyze_flash_risk(video_path, start_time):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    flash_areas = []
    contrast_changes = []

    ret, prev_frame = cap.read()
    if not ret:
        print("비디오를 읽을 수 없습니다.")
        return False

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_brightness = np.mean(prev_gray)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # 밝기 차이 맵 계산 (절대값)
        diff = cv2.absdiff(gray, prev_gray)

        # 임계값 이상 밝기 변화 영역 마스크 생성
        thresh_val = 30  # 밝기 차 임계값 (0~255)
        _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # 깜빡임 영역 비율 계산
        flash_area_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        flash_areas.append(flash_area_ratio)

        # 명도 변화율 계산 (프레임 평균 밝기 차이 절대값 / 255)
        contrast_change = abs(brightness - prev_brightness) / 255
        contrast_changes.append(contrast_change)

        prev_gray = gray
        prev_brightness = brightness

    cap.release()

    flash_frames_per_sec = []

    frame_count = len(flash_areas)
    frames_per_sec = int(fps)
    seconds = frame_count // frames_per_sec

    for sec in range(seconds):
        start = sec * frames_per_sec
        end = start + frames_per_sec
        sec_flash_areas = flash_areas[start:end]
        sec_contrast_changes = contrast_changes[start:end]

        flash_frames = 0
        for area, contrast in zip(sec_flash_areas, sec_contrast_changes):
            if area >= 0.25 and contrast >= 0.1:
                flash_frames += 1

        flash_frames_per_sec.append(flash_frames)

    elapsed = time.time() - start_time
    print(f"소요 시간: {elapsed:.2f}초")
    plot_flash_graph(flash_areas, contrast_changes, fps, flash_frames_per_sec)


def download_youtube_video(url, output_path='./temp'):
    try:
        ydl_opts = {
            'outtmpl': output_path + '/%(title)s.%(ext)s',
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'quiet': False,
            'no_warnings': True,
            'ignoreerrors': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        print(f"오류 발생: {e}")


def init_args():
    parser = argparse.ArgumentParser(description="옵션 사용 예제")
    parser.add_argument(
        "--url",
        required=True,
        help="다운로드할 유튜브 영상의 URL"
    )

    parser.add_argument(
        "--type",
        required=True,
    )
    return parser

# def convert_mp4_to_avi(input_path, output_path):
#     try:
#         (
#             ffmpeg
#             .input(input_path, t=10)
#             .filter('fps', fps=30)
#             .filter('scale', 1024, 768)
#             .output(
#                 output_path,
#                 vcodec='rawvideo',
#                 pix_fmt='bgr24',
#                 an=None
#             )
#             .run(overwrite_output=True)
#         )
#         print(f"변환 성공: {output_path}")
#     except ffmpeg.Error as e:
#         print("변환 중 오류 발생:", e)

if __name__ == "__main__":
    # convert_mp4_to_avi('/Users/sonjeongmin/Desktop/proj/vidoeo_ax_proto/temp/Galaxy Watch6 ｜ Watch6 Classic： How to monitor your heart rate & heart rhythm ｜ Samsung​.mp4', './temp/test_10_1.avi')
    # parser = init_args()
    # args = parser.parse_args()
    # main(args)
    start_time = time.time()
    video_path = "./temp/PEAT_how to monitor your heart rate - browser.avi"  # 분석할 비디오 경로
    analyze_flash_risk(video_path, start_time)
