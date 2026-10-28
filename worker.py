"""
worker.py — QThread 기반 PEAT 분석 워커 (얇은 래퍼).
분석 루프는 engine.analyze_video 단일 소스이며, 여기서는 콜백으로 구동해
시그널(progress/frame_result/finished/error)로 GUI에 전달만 한다.
"""

import time

from PyQt5.QtCore import QThread, pyqtSignal

from engine import AREA_THRESHOLD, WINDOW_SECONDS, analyze_video


class AnalysisWorker(QThread):
    """프레임 단위 PEAT 분석을 수행하는 워커 스레드.

    Signals:
        progress(int, int): (현재 프레임 인덱스, 전체 프레임 수)
        frame_result(list): 프레임 분석 결과 딕셔너리 묶음 (0.1초마다)
        finished(dict): 최종 summary 딕셔너리
        error(str): 에러 메시지
    """

    progress = pyqtSignal(int, int)
    frame_result = pyqtSignal(list)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path, area_threshold=AREA_THRESHOLD,
                 window_seconds=WINDOW_SECONDS, downsample=1,
                 max_frames=0, enable_pattern=True, engine="fpacore"):
        super().__init__()
        self.video_path = video_path
        self.area_threshold = area_threshold
        self.window_seconds = window_seconds
        self.downsample = downsample
        self.max_frames = max_frames
        self.enable_pattern = enable_pattern
        self.engine = engine
        self._stop_flag = False
        self._batch, self._last_emit = [], 0.0

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._analyze()
        except Exception as e:
            self.error.emit(str(e))

    def _on_frame(self, data):
        # 프레임마다 시그널을 보내면 스레드 간 전환 비용이 분석보다 커서(90초 영상: 엔진 28초 / GUI 178초)
        # 0.1초 분량씩 묶어 보낸다. 그래프를 다시 그리는 동안 분석 스레드도 멈춰서(GIL) 간격이 짧을수록
        # 느려지지만(0.25초 30초 / 0.1초 38초), 0.25초는 그래프가 뚝뚝 끊겨 보였다.
        self._batch.append(data)
        now = time.monotonic()
        if now - self._last_emit >= 0.1:
            self._flush()
            self._last_emit = now

    def _flush(self):
        if self._batch:
            last = self._batch[-1]
            self.progress.emit(last["processed"], last["total"])
            self.frame_result.emit(self._batch)
            self._batch = []

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
            engine=self.engine,
        )
        if metrics is None:  # 사용자 중단
            return
        self._flush()

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
        summary["lum_risk"] = metrics["lum_risk"].tolist()
        summary["red_risk"] = metrics["red_risk"].tolist()
        self.finished.emit(summary)
