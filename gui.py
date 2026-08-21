"""
gui.py — PEAT 분석 GUI (PyQt5 + pyqtgraph)
실시간 차트와 WCAG 2.3.1 판정을 제공하는 데스크탑 애플리케이션.
Nuitka로 exe 빌드 가능.
"""

import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QStatusBar, QFrame,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
import numpy as np

from worker import AnalysisWorker
from peat import FAIL_TRANSITIONS, AREA_THRESHOLD, WINDOW_SECONDS, _rolling_max


# ──────────────────────────────────────────────────────────────────────────
# 스타일 상수
# ──────────────────────────────────────────────────────────────────────────
STYLE_PASS = "background-color: #a6e3a1; color: #1e1e2e; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 8px;"
STYLE_CAUTION_PASS = "background-color: #f9e2af; color: #1e1e2e; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 8px;"
STYLE_CAUTION_FAIL = "background-color: #fab387; color: #1e1e2e; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 8px;"
STYLE_FAIL = "background-color: #f38ba8; color: #1e1e2e; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 8px;"
STYLE_IDLE = "background-color: #45475a; color: #a6adc8; font-size: 18px; font-weight: bold; padding: 12px; border-radius: 8px;"


class DownloadWorker(QThread):
    """YouTube 영상 다운로드 워커 (yt-dlp 사용)"""
    progress = pyqtSignal(str)       # 상태 메시지
    finished = pyqtSignal(str)       # 다운로드 완료 → 파일 경로
    error = pyqtSignal(str)          # 에러 메시지

    def __init__(self, url, output_dir):
        super().__init__()
        self.url = url
        self.output_dir = output_dir

    def run(self):
        try:
            import subprocess
            import tempfile

            os.makedirs(self.output_dir, exist_ok=True)
            output_template = os.path.join(self.output_dir, "%(title).50s.%(ext)s")

            self.progress.emit("다운로드 시작...")

            # yt-dlp로 720p 이하 다운로드 (ffmpeg 없이 단일 파일로)
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--format", "best[height<=720]/best[height<=1080]/best",
                "--extractor-args", "youtube:player_client=mediaconnect",
                "--no-playlist",
                "--no-check-certificates",
                "--output", output_template,
                "--print", "after_move:filepath",
                "--no-simulate",
                self.url,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10분 타임아웃
            )

            if result.returncode != 0:
                # yt_dlp 모듈 방식 실패 시 직접 import 시도
                self.progress.emit("yt-dlp 직접 호출 시도...")
                filepath = self._download_with_lib()
            else:
                # stdout 마지막 줄에서 파일 경로 추출
                lines = result.stdout.strip().split("\n")
                filepath = lines[-1].strip() if lines else ""

            if filepath and os.path.exists(filepath):
                self.finished.emit(filepath)
            else:
                self.error.emit(f"다운로드 실패: 파일을 찾을 수 없습니다.\nstdout: {result.stdout}\nstderr: {result.stderr}")

        except Exception as e:
            self.error.emit(f"다운로드 오류: {str(e)}")

    def _download_with_lib(self):
        """yt-dlp를 라이브러리로 직접 사용"""
        import yt_dlp

        output_template = os.path.join(self.output_dir, "%(title).50s.%(ext)s")
        filepath_holder = {}

        def progress_hook(d):
            if d["status"] == "downloading":
                pct = d.get("_percent_str", "?")
                self.progress.emit(f"다운로드 중... {pct}")
            elif d["status"] == "finished":
                filepath_holder["path"] = d.get("filename", "")
                self.progress.emit("변환 중...")

        ydl_opts = {
            "format": "best[height<=720]/best[height<=1080]/best",
            "extractor_args": {"youtube": {"player_client": ["mediaconnect"]}},
            "noplaylist": True,
            "outtmpl": output_template,
            "progress_hooks": [progress_hook],
            "quiet": True,
            "nocheckcertificate": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url])

        # 파일 경로 찾기
        if filepath_holder.get("path"):
            path = filepath_holder["path"]
            # .webm → .mp4 변환된 경우
            mp4_path = os.path.splitext(path)[0] + ".mp4"
            if os.path.exists(mp4_path):
                return mp4_path
            if os.path.exists(path):
                return path

        # output_dir에서 가장 최근 파일 찾기
        files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir)]
        if files:
            return max(files, key=os.path.getmtime)
        return ""


class PEATMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WCAG 2.3.1 Flash Analysis")
        self.setMinimumSize(1100, 700)
        self.worker = None

        # 데이터 버퍼
        self.times = []
        self.lum_window_counts = []
        self.red_window_counts = []
        self.lum_areas = []
        self.red_areas = []
        self.sat_areas = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # ── 상단: 파일 선택 + 옵션 ──
        top_group = QGroupBox("설정")
        top_layout = QVBoxLayout(top_group)

        # 첫 줄: 파일 선택 + 옵션
        file_row = QHBoxLayout()

        self.btn_file = QPushButton("📁 영상 선택")
        self.btn_file.setFixedWidth(120)
        self.btn_file.clicked.connect(self._select_file)
        file_row.addWidget(self.btn_file)

        self.lbl_path = QLabel("선택된 파일 없음")
        self.lbl_path.setStyleSheet("color: #555;")
        file_row.addWidget(self.lbl_path, 1)

        # 옵션
        file_row.addWidget(QLabel("Downsample:"))
        self.spin_downsample = QSpinBox()
        self.spin_downsample.setRange(1, 10)
        self.spin_downsample.setValue(1)
        self.spin_downsample.setToolTip("N번째 프레임만 처리 (속도↑, 정밀도↓)")
        file_row.addWidget(self.spin_downsample)

        self.chk_pattern = QCheckBox("줄무늬 패턴 분석")
        self.chk_pattern.setChecked(True)
        file_row.addWidget(self.chk_pattern)

        self.btn_start = QPushButton("▶ 분석 시작")
        self.btn_start.setFixedWidth(120)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start_analysis)
        file_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹ 중지")
        self.btn_stop.setFixedWidth(80)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_analysis)
        file_row.addWidget(self.btn_stop)

        top_layout.addLayout(file_row)

        # 둘째 줄: YouTube URL 입력
        url_row = QHBoxLayout()

        url_row.addWidget(QLabel("🔗 YouTube URL:"))
        self.input_url = QLineEdit()
        self.input_url.setPlaceholderText("https://www.youtube.com/watch?v=... 또는 https://youtu.be/...")
        url_row.addWidget(self.input_url, 1)

        self.btn_download = QPushButton("⬇ 다운로드 후 분석")
        self.btn_download.setFixedWidth(160)
        self.btn_download.clicked.connect(self._download_and_analyze)
        url_row.addWidget(self.btn_download)

        top_layout.addLayout(url_row)

        layout.addWidget(top_group)

        # ── 중앙: 실시간 차트 ──
        chart_group = QGroupBox("실시간 분석 차트")
        chart_layout = QVBoxLayout(chart_group)

        # pyqtgraph 설정
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#cfd2d6")
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self.plot_widget.setYRange(0, 10)
        self.plot_widget.hideAxis("left")  # PEAT 스타일: Y축 숫자 없음

        # X축 스타일: 진한 색 + 눈금 표시
        x_axis = self.plot_widget.getAxis("bottom")
        x_axis.setLabel("Time (s)", color="#222")
        x_axis.setTickFont(QFont("Malgun Gothic", 10, QFont.Bold))
        x_axis.setPen(pg.mkPen(color="#333", width=2))
        x_axis.setTextPen(pg.mkPen(color="#222"))
        x_axis.setStyle(tickLength=-10)

        # 마우스 줌/패닝 비활성화 (빈 데이터일 때 nan 에러 방지 + PEAT 스타일 고정 뷰)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getViewBox().setAutoVisible(x=False, y=False)
        self.plot_widget.getViewBox().enableAutoRange(axis='xy', enable=False)
        self.plot_widget.setXRange(0, 1)  # 초기값, 분석 시 갱신됨

        # 4단계 밴드 배경 (하단 30%)
        band_top = 3.0  # ymax(10) * 0.30
        h = band_top / 4
        band_colors = ["#c8cbcf", "#bfc3c7", "#b6babe", "#adb1b5"]
        band_labels = ["PASS", "CAUTION (PASS)", "CAUTION (FAIL)", "FAIL"]
        for i, (color, label) in enumerate(zip(band_colors, band_labels)):
            region = pg.LinearRegionItem(
                values=[i * h, (i + 1) * h],
                orientation="horizontal",
                brush=pg.mkBrush(color),
                movable=False,
            )
            region.setZValue(-10)
            self.plot_widget.addItem(region)
            # 밴드 라벨
            text = pg.TextItem(text=f"  {label}", color="#777", anchor=(0, 0.5))
            text.setPos(0, i * h + h / 2)
            text.setZValue(-5)
            self.plot_widget.addItem(text)

        # 시계열 플롯 — PEAT 스타일
        # 범례 (커브 생성 전에 추가해야 인식됨)
        legend = self.plot_widget.addLegend(offset=(10, 10))
        legend.setBrush(pg.mkBrush("#cfd2d6"))
        legend.setFlag(legend.GraphicsItemFlag.ItemIsMovable, False)

        self.curve_lum_act = self.plot_widget.plot(
            pen=pg.mkPen(color="white", width=2, style=Qt.DashLine),
            name="Luminance flash"
        )
        self.curve_red_act = self.plot_widget.plot(
            pen=pg.mkPen(color="#FF5252", width=2, style=Qt.DashLine),
            name="Red flash"
        )
        self.curve_lum_diag = self.plot_widget.plot(
            pen=pg.mkPen(color="white", width=1),
            name="Lum flash diag"
        )
        self.curve_red_diag = self.plot_widget.plot(
            pen=pg.mkPen(color="darkred", width=1),
            name="Red flash diag"
        )
        self.curve_extended = self.plot_widget.plot(
            pen=pg.mkPen(color="blue", width=1.2),
            name="Extended Flash"
        )

        chart_layout.addWidget(self.plot_widget)

        # 프레임 미리보기 (차트 클릭 시 해당 시간의 영상 프레임 표시)
        self.frame_preview = QLabel("차트를 클릭하면 해당 시간의 프레임이 표시됩니다")
        self.frame_preview.setAlignment(Qt.AlignCenter)
        self.frame_preview.setFixedHeight(200)
        self.frame_preview.setStyleSheet(
            "background-color: #181825; color: #6c7086; font-size: 11px; border-radius: 8px; border: 1px solid #313244;"
        )
        chart_layout.addWidget(self.frame_preview)

        # 차트 클릭 시그널 연결
        self.plot_widget.scene().sigMouseClicked.connect(self._on_chart_click)
        # 클릭 위치 표시용 수직선
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(color="#FFEB3B", width=1, style=Qt.DashLine))
        self.vline.setVisible(False)
        self.plot_widget.addItem(self.vline)

        layout.addWidget(chart_group, 1)

        # ── 하단: 진행률 + 판정 ──
        bottom_layout = QHBoxLayout()

        # 진행률
        progress_frame = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m 프레임 (%p%)")
        progress_frame.addWidget(self.progress_bar)

        self.lbl_status = QLabel("대기 중")
        self.lbl_status.setStyleSheet("color: #666; font-size: 12px;")
        progress_frame.addWidget(self.lbl_status)
        bottom_layout.addLayout(progress_frame, 1)

        # 판정 표시
        verdict_frame = QVBoxLayout()
        verdict_frame.setAlignment(Qt.AlignCenter)

        self.lbl_verdict_title = QLabel("판정 결과")
        self.lbl_verdict_title.setAlignment(Qt.AlignCenter)
        self.lbl_verdict_title.setStyleSheet("font-size: 11px; color: #666;")
        verdict_frame.addWidget(self.lbl_verdict_title)

        self.lbl_verdict = QLabel("—")
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setFixedWidth(250)
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)
        verdict_frame.addWidget(self.lbl_verdict)

        bottom_layout.addLayout(verdict_frame)
        layout.addLayout(bottom_layout)

        # 상세 결과
        self.lbl_details = QLabel("")
        self.lbl_details.setStyleSheet("font-size: 11px; color: #444; padding: 4px;")
        self.lbl_details.setWordWrap(True)
        layout.addWidget(self.lbl_details)

        # 상태바
        self.statusBar().showMessage("영상을 선택한 후 분석을 시작하세요.")

    # ──────────────────────────────────────────────────────────────────
    # 파일 선택
    # ──────────────────────────────────────────────────────────────────
    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v *.mpg *.mpeg *.3gp *.ts);;All Files (*)"
        )
        if path:
            self.video_path = path
            self.lbl_path.setText(os.path.basename(path))
            self.lbl_path.setToolTip(path)
            self.btn_start.setEnabled(True)
            self.statusBar().showMessage(f"선택됨: {path}")

    # ──────────────────────────────────────────────────────────────────
    # YouTube 다운로드 후 분석
    # ──────────────────────────────────────────────────────────────────
    def _download_and_analyze(self):
        url = self.input_url.text().strip()
        if not url:
            self.statusBar().showMessage("URL을 입력해주세요.")
            return

        if "youtube.com" not in url and "youtu.be" not in url:
            self.statusBar().showMessage("유효한 YouTube URL을 입력해주세요.")
            return

        # UI 상태 변경
        self.btn_download.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.lbl_status.setText("YouTube 영상 다운로드 중...")
        self.statusBar().showMessage("다운로드 중...")
        self.lbl_verdict.setText("다운로드 중...")
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)

        # temp 폴더
        import tempfile
        self._temp_dir = os.path.join(tempfile.gettempdir(), "wcag_flash_temp")

        self.dl_worker = DownloadWorker(url, self._temp_dir)
        self.dl_worker.progress.connect(self._on_dl_progress)
        self.dl_worker.finished.connect(self._on_dl_finished)
        self.dl_worker.error.connect(self._on_dl_error)
        self.dl_worker.start()

    def _on_dl_progress(self, msg):
        self.lbl_status.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_dl_finished(self, filepath):
        self.video_path = filepath
        self.lbl_path.setText(f"🎬 {os.path.basename(filepath)}")
        self.lbl_path.setToolTip(filepath)
        self.btn_download.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.lbl_status.setText(f"다운로드 완료: {os.path.basename(filepath)}")
        self.statusBar().showMessage("다운로드 완료 — 분석을 시작하세요.")
        self.lbl_verdict.setText("—")
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)

        # 자동으로 분석 시작
        self._start_analysis()

    def _on_dl_error(self, msg):
        self.btn_download.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.lbl_status.setText("다운로드 실패")
        self.lbl_verdict.setText("오류")
        self.lbl_verdict.setStyleSheet(STYLE_FAIL)
        self.lbl_details.setText(f"다운로드 오류: {msg}")
        self.statusBar().showMessage("다운로드 실패")

    # ──────────────────────────────────────────────────────────────────
    # 분석 시작/중지
    # ──────────────────────────────────────────────────────────────────
    def _start_analysis(self):
        if not hasattr(self, "video_path"):
            return

        # 초기화
        self.times.clear()
        self.lum_window_counts.clear()
        self.red_window_counts.clear()
        self.lum_areas.clear()
        self.red_areas.clear()
        self.sat_areas.clear()
        self.curve_lum_act.setData([], [])
        self.curve_red_act.setData([], [])
        self.curve_lum_diag.setData([], [])
        self.curve_red_diag.setData([], [])
        self.lbl_verdict.setText("분석 중...")
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)
        self.lbl_details.setText("")
        self.progress_bar.setValue(0)

        # 분석 중에는 차트 조작 비활성화
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)

        # 버튼 상태
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_file.setEnabled(False)

        # 워커 생성
        downsample = self.spin_downsample.value()
        enable_pattern = self.chk_pattern.isChecked()

        # emit_every: 모든 프레임 데이터를 받되, 차트 렌더링은 GUI에서 throttle
        emit_every = 1

        self.worker = AnalysisWorker(
            video_path=self.video_path,
            downsample=downsample,
            enable_pattern=enable_pattern,
            emit_every=emit_every,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.frame_result.connect(self._on_frame_result)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.lbl_status.setText("분석 진행 중...")
        self.statusBar().showMessage("분석 중...")

    def _stop_analysis(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self._reset_buttons()
        self.lbl_status.setText("중지됨")
        self.lbl_verdict.setText("중지됨")
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)
        self.statusBar().showMessage("분석이 중지되었습니다.")

    def _reset_buttons(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_file.setEnabled(True)

    # ──────────────────────────────────────────────────────────────────
    # 워커 시그널 핸들러
    # ──────────────────────────────────────────────────────────────────
    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(current)

    def _on_frame_result(self, data):
        self.times.append(data["time"])
        self.lum_window_counts.append(data["lum_window_count"])
        self.red_window_counts.append(data["red_window_count"])
        self.sat_areas.append(data.get("sat_area", 0.0))

        # 차트 렌더링: 30포인트마다 (UI 성능 유지)
        n = len(self.times)
        if n % 30 != 0:
            return

        t_arr = np.array(self.times)
        lum_window = np.array(self.lum_window_counts, dtype=np.float32)
        red_window = np.array(self.red_window_counts, dtype=np.float32)
        sat_area = np.array(self.sat_areas, dtype=np.float32)

        eff_fps = n / (t_arr[-1] + 1e-9)  # 현재까지의 effective fps 추정
        win = max(1, int(round(eff_fps)))

        ymax = 10.0
        band_top = ymax * 0.30
        _floor = FAIL_TRANSITIONS - 4

        # peat.py plot_unified와 동일한 로직
        lum_act = np.clip(lum_window - _floor, 0, None)
        lum_act = _rolling_max(lum_act, max(1, win // 2))
        act_scale = (ymax * 0.92) / (lum_act.max() + 1e-9)

        red_act = _rolling_max(sat_area, win)
        red_scale = (band_top * 0.5) / (red_act.max() + 1e-9)

        diag_scale = (band_top * 0.9) / (lum_window.max() + 1e-9)

        self.curve_lum_act.setData(t_arr, lum_act * act_scale)
        self.curve_red_act.setData(t_arr, red_act * red_scale)
        self.curve_lum_diag.setData(t_arr, lum_window * diag_scale)
        self.curve_red_diag.setData(t_arr, red_window * diag_scale)
        self.plot_widget.setXRange(0, t_arr[-1] + 0.5, padding=0)

        # 실시간 상태
        lum_max = int(lum_window.max())
        red_max = int(red_window.max())
        self.lbl_status.setText(
            f"분석 중... | 시간: {data['time']:.1f}s | "
            f"휘도 최대: {lum_max} | 적색 최대: {red_max}"
        )

    def _on_finished(self, summary):
        self._reset_buttons()
        verdict = summary["verdict"]

        # ── 최종 차트: peat.py plot_unified와 완전 동일한 로직 ──
        if "times" in summary and len(summary["times"]) > 0:
            t = np.array(summary["times"])
            lum_window = np.array(summary["lum_window"], dtype=np.float32)
            red_window = np.array(summary["red_window"], dtype=np.float32)
            sat_area = np.array(summary.get("sat_area_series", [0.0] * len(t)), dtype=np.float32)
            ext = np.array(summary["extended_series"], dtype=np.float32)
            eff_fps = summary["effective_fps"]
            win = max(1, int(round(eff_fps)))  # 1초 윈도우 프레임수

            ymax = 10.0
            band_top = ymax * 0.30
            _floor = FAIL_TRANSITIONS - 4  # = 3

            # ── Luminance activity (점선, 메인) — peat.py 동일 ──
            lum_act = np.clip(lum_window - _floor, 0, None)
            lum_act = _rolling_max(lum_act, max(1, win // 2))
            act_scale = (ymax * 0.92) / (lum_act.max() + 1e-9)

            # ── Red activity (점선) — sat_area의 rolling_max ──
            red_act = _rolling_max(sat_area, win)
            red_scale = (band_top * 0.5) / (red_act.max() + 1e-9)

            # ── Diag (실선, 하단) — 카운트 스케일 ──
            diag_scale = (band_top * 0.9) / (lum_window.max() + 1e-9)

            # ── Extended Flash (파란선) — ≥0.8일 때만 표시 ──
            ext_disp = np.where(ext >= 0.8, ext * band_top, 0.0)

            # 차트 그리기
            self.curve_lum_act.setData(t, lum_act * act_scale)
            self.curve_red_act.setData(t, red_act * red_scale)
            self.curve_lum_diag.setData(t, lum_window * diag_scale)
            self.curve_red_diag.setData(t, red_window * diag_scale)
            self.curve_extended.setData(t, ext_disp)
            self.plot_widget.setXRange(0, t[-1] + 0.5, padding=0)

            # 분석 완료 후 줌/슬라이드 활성화
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_widget.setMenuEnabled(True)
            # 축 범위 제한: 음수 영역 안 보이게
            self.plot_widget.setLimits(xMin=0, yMin=0)

        # 판정 스타일 적용
        if verdict == "PASS":
            style = STYLE_PASS
        elif verdict == "CAUTION (PASS)":
            style = STYLE_CAUTION_PASS
        elif verdict == "CAUTION (FAIL)":
            style = STYLE_CAUTION_FAIL
        else:
            style = STYLE_FAIL

        self.lbl_verdict.setText(verdict)
        self.lbl_verdict.setStyleSheet(style)

        # 상세 결과
        details = (
            f"FPS: {summary['fps']:.1f} | "
            f"처리 프레임: {summary['frames_processed']} | "
            f"1초내 최대 전환: {summary['flash_max_per_window']} (FAIL≥{FAIL_TRANSITIONS})\n"
            f"휘도 FAIL: {summary['luminance_fail']} | "
            f"적색 FAIL: {summary['red_fail']} | "
            f"패턴 FAIL: {summary['pattern_fail']} | "
            f"확장 플래시 경고: {summary['extended_flash_warn']}"
        )
        self.lbl_details.setText(details)
        self.lbl_status.setText("분석 완료")
        self.statusBar().showMessage(f"분석 완료 — {verdict}")

        self.worker = None

    def _on_error(self, msg):
        self._reset_buttons()
        self.lbl_verdict.setText("오류")
        self.lbl_verdict.setStyleSheet(STYLE_FAIL)
        self.lbl_details.setText(f"오류: {msg}")
        self.lbl_status.setText("오류 발생")
        self.statusBar().showMessage(f"오류: {msg}")
        self.worker = None

    # ──────────────────────────────────────────────────────────────────
    # 차트 클릭 → 영상 프레임 표시
    # ──────────────────────────────────────────────────────────────────
    def _on_chart_click(self, event):
        """차트 클릭 시 해당 시간의 영상 프레임을 미리보기에 표시"""
        if not hasattr(self, "video_path") or not self.times:
            return

        # 클릭 좌표를 데이터 좌표로 변환
        pos = event.scenePos()
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
        click_time = mouse_point.x()

        if click_time < 0:
            return

        # 수직선 표시
        self.vline.setPos(click_time)
        self.vline.setVisible(True)

        # 해당 시간의 프레임 추출
        self._show_frame_at_time(click_time)

    def _show_frame_at_time(self, time_sec):
        """영상에서 특정 시간의 프레임을 추출하여 미리보기에 표시"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            # 해당 시간으로 seek
            frame_num = int(time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.frame_preview.setText(f"프레임 읽기 실패 (t={time_sec:.2f}s)")
                return

            # BGR → RGB 변환 후 QPixmap으로
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w

            from PyQt5.QtGui import QImage, QPixmap
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # 미리보기 크기에 맞게 스케일
            scaled = pixmap.scaled(
                self.frame_preview.width(),
                self.frame_preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.frame_preview.setPixmap(scaled)
            self.statusBar().showMessage(f"프레임 표시: {time_sec:.2f}s (frame #{frame_num})")

        except Exception as e:
            self.frame_preview.setText(f"오류: {e}")


def main():
    # 고DPI 스케일링 (Windows에서 차트 선명하게)
    import os
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 폰트 설정 (Windows에서 선명하게)
    import platform
    if platform.system() == "Windows":
        font = QFont("Malgun Gothic", 9)  # 맑은 고딕
    elif platform.system() == "Darwin":
        font = QFont("Apple SD Gothic Neo", 12)
    else:
        font = QFont("Noto Sans", 10)
    font.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(font)

    # 다크 모던 테마
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e2e;
        }
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-size: 12px;
        }
        QGroupBox {
            border: 1px solid #45475a;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
            font-weight: bold;
            color: #cdd6f4;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        QPushButton {
            background-color: #45475a;
            color: #cdd6f4;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #585b70;
        }
        QPushButton:pressed {
            background-color: #6c7086;
        }
        QPushButton:disabled {
            background-color: #313244;
            color: #6c7086;
        }
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: #313244;
            text-align: center;
            color: #cdd6f4;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #89b4fa;
            border-radius: 4px;
        }
        QLabel {
            color: #cdd6f4;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
        }
        QCheckBox {
            color: #cdd6f4;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 1px solid #45475a;
            background-color: #313244;
        }
        QCheckBox::indicator:checked {
            background-color: #89b4fa;
            border-color: #89b4fa;
        }
        QStatusBar {
            background-color: #181825;
            color: #a6adc8;
            border-top: 1px solid #313244;
        }
    """)

    window = PEATMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
