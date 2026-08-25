"""
worker.py — QThread 기반 PEAT 분석 워커 (얇은 래퍼).
분석 루프는 peat.analyze_video 단일 소스이며, 여기서는 콜백으로 구동해
시그널(progress/frame_result/finished/error)로 GUI에 전달만 한다.
"""

from PyQt5.QtCore import QThread, pyqtSignal

from peat import AREA_THRESHOLD, WINDOW_SECONDS, analyze_video


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

    def _on_frame(self, data):
        if data["processed"] % self.emit_every == 0:
            self.progress.emit(data["processed"], data["total"])
            self.frame_result.emit(data)

    def _analyze(self):
        metrics = analyze_video(
            self.video_path,
            area_threshold=self.area_threshold,
            window_seconds=self.window_seconds,
            downsample=self.downsample,
            max_frames=self.max_frames,
            enable_pattern=self.enable_pattern,
            on_frame=self._on_frame,
            should_stop=lambda: self._stop_flag,
        )
        if metrics is None:  # 사용자 중단
            return

        # GUI 최종 차트용: summary에 전체 시계열 포함
        summary = dict(metrics["summary"])
        summary["times"] = [float(v) for v in metrics["times"]]
        summary["lum_window"] = metrics["lum_window"].tolist()
        summary["red_window"] = metrics["red_window"].tolist()
        summary["lum_window_raw"] = metrics["lum_window_raw"].tolist()
        summary["red_window_raw"] = metrics["red_window_raw"].tolist()
        summary["lum_window_act"] = metrics["lum_window_act"].tolist()
        summary["red_window_act"] = metrics["red_window_act"].tolist()
        summary["extended_series"] = metrics["extended_series"].tolist()
        self.finished.emit(summary)
