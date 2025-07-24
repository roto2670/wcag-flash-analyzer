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

def analyze_flash_sc231(video_path, start_time):
    """
    Analyze video for flashes according to SC 2.3.1 guidelines.
    Checks for luminance contrast changes and red flashes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    prev_frame = None
    flash_detected = False
    frame_count = 0
    flash_threshold = 0.4  # example threshold for luminance contrast change
    red_flash_threshold = 0.3  # example threshold for red flash intensity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Convert frame to grayscale for luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate luminance contrast change
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff) / 255.0  # normalize

            # Check for red flashes
            red_channel = frame[:, :, 2].astype(np.float32) / 255.0
            green_channel = frame[:, :, 1].astype(np.float32) / 255.0
            blue_channel = frame[:, :, 0].astype(np.float32) / 255.0

            red_flash = np.mean(red_channel - np.maximum(green_channel, blue_channel))

            if mean_diff > flash_threshold or red_flash > red_flash_threshold:
                flash_detected = True
                print(f"Flash detected at frame {frame_count} - luminance change: {mean_diff:.3f}, red flash: {red_flash:.3f}")

        prev_frame = gray

    cap.release()

    if flash_detected:
        print("Result: Video fails SC 2.3.1 flash guidelines.")
    else:
        print("Result: Video passes SC 2.3.1 flash guidelines.")

if __name__ == "__main__":
    start_time = time.time()
    video_path = "./temp/PEAT_wuwa.avi"  # 분석할 비디오 경로
    analyze_flash_sc231(video_path, start_time)
