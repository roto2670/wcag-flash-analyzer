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

if __name__ == "__main__":
    # convert_mp4_to_avi('/Users/sonjeongmin/Desktop/proj/vidoeo_ax_proto/temp/Galaxy Watch6 ｜ Watch6 Classic： How to monitor your heart rate & heart rhythm ｜ Samsung​.mp4', './temp/test_10_1.avi')
    # parser = init_args()
    # args = parser.parse_args()
    # main(args)
    start_time = time.time()
    video_path = "./temp/PEAT_wuwa.avi"  # 분석할 비디오 경로
    # analyze_flash_risk(video_path, start_time)
