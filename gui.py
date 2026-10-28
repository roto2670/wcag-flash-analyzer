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
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QSpinBox, QCheckBox, QFrame,
    QLineEdit, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import numpy as np

from worker import AnalysisWorker


def fmt_mmss(sec, decimals=0):
    """초 → 분:초 문자열 (예: 79.5 → '1:19.5')."""
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60.0)
    if decimals > 0:
        s = round(s, decimals)
        if s >= 60.0:
            m, s = m + 1, 0.0
        return f"{int(m)}:{s:0{3 + decimals}.{decimals}f}"
    s = int(round(s))
    if s == 60:
        m, s = m + 1, 0
    return f"{int(m)}:{s:02d}"


class TimeAxisItem(pg.AxisItem):
    """x축 눈금을 분:초로 표시. 눈금 간격도 시간 단위(1·2·5·10·15·30초, 1·2·5·10분)로 잡는다."""
    STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600]

    def tickSpacing(self, minVal, maxVal, size):
        span = maxVal - minVal
        if span <= 0 or size <= 0:
            return super().tickSpacing(minVal, maxVal, size)
        target = span / max(2.0, size / 80.0)   # 눈금 사이 최소 ~80px
        major = next((st for st in self.STEPS if st >= target), self.STEPS[-1])
        i = self.STEPS.index(major)
        minor = self.STEPS[i - 1] if i > 0 else major / 2
        return [(major, 0), (minor, 0)]

    def tickStrings(self, values, scale, spacing):
        dec = 1 if spacing < 1 else 0
        return [fmt_mmss(v, dec) for v in values]
from engine import (
    FAIL_TRANSITIONS, RISK_FAIL, CHART_YMAX, CHART_BANDS, CHART_PASS_LINE,
    count_to_chart_y, risk_to_chart_y,
    flash_mask, red_mask,
)


# ──────────────────────────────────────────────────────────────────────────
# 스타일 상수
# ──────────────────────────────────────────────────────────────────────────
STYLE_PASS = "background-color: #9ece6a; color: #1a1b26; font-size: 20px; font-weight: bold; padding: 14px; border-radius: 10px;"
STYLE_CAUTION_PASS = "background-color: #e0af68; color: #1a1b26; font-size: 20px; font-weight: bold; padding: 14px; border-radius: 10px;"
STYLE_CAUTION_FAIL = "background-color: #ff9e64; color: #1a1b26; font-size: 20px; font-weight: bold; padding: 14px; border-radius: 10px;"
STYLE_FAIL = "background-color: #f7768e; color: #1a1b26; font-size: 20px; font-weight: bold; padding: 14px; border-radius: 10px;"
STYLE_IDLE = "background-color: #24283b; color: #565f89; font-size: 20px; font-weight: bold; padding: 14px; border-radius: 10px; border: 1px solid #414868;"


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

            os.makedirs(self.output_dir, exist_ok=True)
            output_template = os.path.join(self.output_dir, "%(title).50s.%(ext)s")

            self.progress.emit("다운로드 시작...")

            # yt-dlp로 720p 이하 다운로드 (비디오만, H.264 우선)
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--format", "bv[height<=720][vcodec^=avc]/bv[height<=720]/bv[height<=1080][vcodec^=avc]/bv/b",
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
            "format": "bv[height<=720][vcodec^=avc]/bv[height<=720]/bv[height<=1080][vcodec^=avc]/bv/b",
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
        self.setAcceptDrops(True)
        self.worker = None
        self._current_preview_time = None  # 현재 미리보기 프레임 시간
        self._video_fps = 30.0             # 영상 fps (프레임 이동용)

        # 데이터 버퍼
        self.times = []
        self.lum_window_counts = []
        self.red_window_counts = []
        self.lum_raw_counts = []
        self.lum_risk = []
        self.red_risk = []
        self.red_raw_counts = []
        self.lum_act_counts = []
        self.red_act_counts = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # ── 상단: 입력 영역 ──
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 12, 16, 12)
        input_layout.setSpacing(10)

        # 첫 줄: 파일 선택 + 분석 버튼
        file_row = QHBoxLayout()
        file_row.setSpacing(8)

        self.btn_file = QPushButton("📁  영상 파일 선택")
        self.btn_file.setObjectName("primaryBtn")
        self.btn_file.setFixedHeight(36)
        self.btn_file.clicked.connect(self._select_file)
        file_row.addWidget(self.btn_file)

        self.lbl_path = QLabel("파일을 선택하거나 드래그하세요")
        self.lbl_path.setObjectName("pathLabel")
        file_row.addWidget(self.lbl_path, 1)

        self.btn_start = QPushButton("▶  분석")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.setFixedSize(100, 36)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start_analysis)
        file_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setFixedSize(36, 36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_analysis)
        file_row.addWidget(self.btn_stop)

        input_layout.addLayout(file_row)

        # 둘째 줄: YouTube URL
        url_row = QHBoxLayout()
        url_row.setSpacing(8)

        self.input_url = QLineEdit()
        self.input_url.setObjectName("urlInput")
        self.input_url.setPlaceholderText("YouTube URL을 붙여넣으세요...")
        self.input_url.setFixedHeight(36)
        self.input_url.textEdited.connect(self._on_url_edited)
        url_row.addWidget(self.input_url, 1)

        self.btn_download = QPushButton("⬇  URL 분석")
        self.btn_download.setObjectName("downloadBtn")
        self.btn_download.setFixedHeight(36)
        self.btn_download.clicked.connect(self._download_and_analyze)
        url_row.addWidget(self.btn_download)

        input_layout.addLayout(url_row)

        # 셋째 줄: 옵션
        opt_row = QHBoxLayout()
        opt_row.setSpacing(12)

        opt_row.addWidget(QLabel("Downsample:"))
        self.spin_downsample = QSpinBox()
        self.spin_downsample.setRange(1, 10)
        self.spin_downsample.setValue(1)
        self.spin_downsample.setFixedWidth(60)
        self.spin_downsample.setToolTip("N번째 프레임만 처리 (속도↑, 정밀도↓)")
        opt_row.addWidget(self.spin_downsample)

        self.chk_pattern = QCheckBox("줄무늬 패턴 분석")
        self.chk_pattern.setChecked(True)
        opt_row.addWidget(self.chk_pattern)

        # PEAT 'Show warnings'처럼 경고 수준(위험도 0.1~0.4) 표시를 끌 수 있게 (2차 QA 기능 외 #7)
        self.chk_warnings = QCheckBox("경고(CAUTION) 표시")
        self.chk_warnings.setChecked(True)
        self.chk_warnings.setToolTip("끄면 FAIL 구간만 목록에 나오고, 그래프 점선의 경고 수준(0.5 미만)을 숨깁니다")
        self.chk_warnings.toggled.connect(self._on_warnings_toggled)
        opt_row.addWidget(self.chk_warnings)

        opt_row.addStretch()

        self.btn_export = QPushButton("💾  결과 내보내기")
        self.btn_export.setObjectName("exportBtn")
        self.btn_export.setFixedHeight(30)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_result)
        opt_row.addWidget(self.btn_export)

        input_layout.addLayout(opt_row)
        layout.addWidget(input_frame)

        # ── 중앙: 차트 + 프레임 미리보기 ──
        center_layout = QHBoxLayout()
        center_layout.setSpacing(12)

        # 차트
        chart_frame = QFrame()
        chart_frame.setObjectName("chartFrame")
        chart_inner = QVBoxLayout(chart_frame)
        chart_inner.setContentsMargins(8, 8, 8, 8)

        # segmentedLineMode: 굵은 점선을 선분 단위로 그림. 한 덩어리 경로로 그리면 깜빡임이 많은 긴 영상
        # (12분)에서 전체 보기 다시 그리기가 2.7초 걸려 확대·축소가 버벅였다 (선분 단위 0.1초).
        pg.setConfigOptions(antialias=True, useOpenGL=False, segmentedLineMode="on")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.plot_widget.setBackground("#c9ccd0")
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self.plot_widget.setYRange(0, CHART_YMAX, padding=0)
        self.plot_widget.hideAxis("left")

        x_axis = self.plot_widget.getAxis("bottom")
        x_axis.setLabel("Time (m:ss)", color="#222")
        x_axis.setTickFont(QFont("Malgun Gothic", 10, QFont.Bold))
        x_axis.setPen(pg.mkPen(color="#333", width=2))
        x_axis.setTextPen(pg.mkPen(color="#222"))
        x_axis.setStyle(tickLength=-10)

        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getViewBox().setAutoVisible(x=False, y=False)
        self.plot_widget.getViewBox().enableAutoRange(axis='xy', enable=False)
        self.plot_widget.setXRange(0, 1)

        # 밴드 — 공식 PEAT 결과창 실측(engine.CHART_BANDS): PASS·FAIL은 밝게, CAUTION 두 칸은 어둡게
        lo = 0.0
        for hi, color, label in zip(CHART_BANDS, ["#c9ccd0", "#b3b7bb", "#b3b7bb"],
                                    ["PASS", "CAUTION (PASS)", "CAUTION (FAIL)"]):
            region = pg.LinearRegionItem(values=[lo, hi], orientation="horizontal",
                                         brush=pg.mkBrush(color), movable=False)
            region.setZValue(-10)
            for ln in region.lines:
                ln.setPen(pg.mkPen(None))
            self.plot_widget.addItem(region)
            text = pg.TextItem(text=f"  {label}", color="#888", anchor=(0, 0.5))
            text.setPos(0, (lo + hi) / 2)
            text.setZValue(-5)
            self.plot_widget.addItem(text)
            lo = hi
        text = pg.TextItem(text="  FAIL", color="#888", anchor=(0, 0.5))
        text.setPos(0, CHART_BANDS[2] + 0.1)
        text.setZValue(-5)
        self.plot_widget.addItem(text)
        pass_line = pg.InfiniteLine(pos=CHART_PASS_LINE, angle=0, pen=pg.mkPen(color="black", width=1))
        pass_line.setZValue(-4)
        self.plot_widget.addItem(pass_line)

        # 커브
        # 범례는 그래프 위쪽 바깥에 한 줄로 — 안쪽에 두면 높은 위험도 선을 가린다 (2차 QA 기능 외 #2)
        plot_item = self.plot_widget.getPlotItem()
        legend = pg.LegendItem(colCount=5, horSpacing=10, labelTextColor="#222")
        plot_item.legend = legend                    # plot(name=...)이 이 범례에 항목을 추가
        plot_item.layout.removeItem(plot_item.titleLabel)   # 안 쓰는 제목 칸 자리에 범례
        plot_item.titleLabel.hide()
        plot_item.layout.addItem(legend, 0, 1, alignment=Qt.AlignLeft)

        self.curve_lum_act = self.plot_widget.plot(
            pen=pg.mkPen(color="white", width=3, style=Qt.DotLine), name="Luminance flash")
        self.curve_red_act = self.plot_widget.plot(
            pen=pg.mkPen(color="#FF5252", width=3, style=Qt.DotLine), name="Red flash")
        self.curve_lum_diag = self.plot_widget.plot(
            pen=pg.mkPen(color="white", width=1), name="Lum flash diag")
        self.curve_red_diag = self.plot_widget.plot(
            pen=pg.mkPen(color="darkred", width=1), name="Red flash diag")
        self.curve_extended = self.plot_widget.plot(
            pen=pg.mkPen(color="#89b4fa", width=2), name="Extended Flash")

        # 줌 리셋: pyqtgraph의 'A' 버튼은 마우스를 올려야 보이고 뜻이 안 전해져서 숨기고,
        # 항상 보이는 [전체 보기] 버튼을 그래프 위에 둔다 (2차 QA 기능 외 #10)
        self.plot_widget.getPlotItem().hideButtons()
        chart_bar = QHBoxLayout()
        chart_bar.addStretch()
        self.btn_full_view = QPushButton("🔍  전체 보기")
        self.btn_full_view.setObjectName("exportBtn")
        self.btn_full_view.setFixedHeight(26)
        self.btn_full_view.setToolTip("확대·이동한 그래프를 영상 전체 범위로 되돌립니다")
        self.btn_full_view.setEnabled(False)
        self.btn_full_view.clicked.connect(self._full_view)
        chart_bar.addWidget(self.btn_full_view)
        chart_inner.addLayout(chart_bar)
        chart_inner.addWidget(self.plot_widget)

        # 차트 클릭 시그널
        self.plot_widget.scene().sigMouseClicked.connect(self._on_chart_click)
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(color="#FFEB3B", width=3))
        self.vline.setVisible(False)
        self.plot_widget.addItem(self.vline)
        # 선택한 FAIL 구간 띠 (2차 QA 기능 외 #4)
        self.sel_region = pg.LinearRegionItem(values=(0, 0), orientation="vertical", movable=False,
                                              brush=pg.mkBrush(255, 235, 59, 70))
        for ln in self.sel_region.lines:
            ln.setPen(pg.mkPen(None))
        self.sel_region.setZValue(-3)
        self.sel_region.setVisible(False)
        self.plot_widget.addItem(self.sel_region)
        self.fail_segs, self.fail_selected = [], None

        center_layout.addWidget(chart_frame, 3)

        # 우측: 프레임 미리보기 + 판정
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 프레임 미리보기
        self.frame_preview = QLabel("차트 클릭 시\n프레임 표시")
        self.frame_preview.setAlignment(Qt.AlignCenter)
        self.frame_preview.setMinimumSize(280, 180)
        self.frame_preview.setObjectName("framePreview")
        right_panel.addWidget(self.frame_preview)

        # 위반 영역 하이라이트 토글 (직전 프레임 대비 유해 변화 픽셀)
        self.chk_highlight = QCheckBox("위반 영역 표시 (노랑=휘도, 청록=적색)")
        self.chk_highlight.setChecked(True)
        self.chk_highlight.toggled.connect(self._refresh_preview)
        right_panel.addWidget(self.chk_highlight)

        # 판정 결과
        self.lbl_verdict = QLabel("—")
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setObjectName("verdict")
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)
        right_panel.addWidget(self.lbl_verdict)

        # FAIL 구간 (클릭하면 해당 시간으로 이동)
        self.lbl_fail_segments = QLabel("")
        self.lbl_fail_segments.setObjectName("failSegments")
        self.lbl_fail_segments.setWordWrap(True)
        self.lbl_fail_segments.setOpenExternalLinks(False)
        self.lbl_fail_segments.linkActivated.connect(self._on_fail_segment_click)
        self.lbl_fail_segments.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # FAIL 구간이 많으면 패널 아래로 잘리던 문제 — 남는 공간을 채우고 넘치면 스크롤 (2차 QA 기능 외 #3)
        self.fail_scroll = QScrollArea()
        self.fail_scroll.setObjectName("failScroll")
        self.fail_scroll.setWidgetResizable(True)
        self.fail_scroll.setFrameShape(QFrame.NoFrame)
        self.fail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.fail_scroll.setWidget(self.lbl_fail_segments)
        right_panel.addWidget(self.fail_scroll, 1)
        center_layout.addLayout(right_panel, 1)

        layout.addLayout(center_layout, 1)

        # ── 하단: 진행률 + 상세 ──
        bottom_frame = QFrame()
        bottom_frame.setObjectName("bottomFrame")
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(12, 8, 12, 8)

        progress_col = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m 프레임 (%p%)")
        self.progress_bar.setFixedHeight(18)
        progress_col.addWidget(self.progress_bar)

        self.lbl_status = QLabel("대기 중")
        self.lbl_status.setObjectName("statusLabel")
        progress_col.addWidget(self.lbl_status)

        self.lbl_details = QLabel("")
        self.lbl_details.setObjectName("detailsLabel")
        self.lbl_details.setWordWrap(True)
        progress_col.addWidget(self.lbl_details)

        bottom_layout.addLayout(progress_col)
        layout.addWidget(bottom_frame)

        # 상태바
        self.statusBar().showMessage("영상을 선택하거나 드래그하세요.")

    # ──────────────────────────────────────────────────────────────────
    # 파일 선택
    # ──────────────────────────────────────────────────────────────────
    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v *.mpg *.mpeg *.3gp *.ts);;All Files (*)"
        )
        if path:
            self._set_local_video(path)
            self.statusBar().showMessage(f"선택됨: {path}")

    def _set_local_video(self, path):
        """로컬 영상 파일을 분석 대상으로 — URL 입력칸은 비운다 (2차 QA 기능 외 #8)."""
        self.video_path = path
        self.lbl_path.setText(os.path.basename(path))
        self.lbl_path.setToolTip(path)
        self.input_url.clear()
        self.btn_start.setEnabled(True)

    def _on_url_edited(self, text):
        """URL을 입력하기 시작하면 이전 파일 표시를 지우고 분석 버튼을 끈다. 비우면 원래대로.
        video_path는 그대로 둬서 이전 결과의 그래프 클릭·미리보기는 계속 동작한다."""
        if text.strip():
            self.lbl_path.setText("YouTube URL → [URL 분석]을 누르세요")
            self.lbl_path.setToolTip("")
            self.btn_start.setEnabled(False)
        elif getattr(self, "video_path", None):
            self.lbl_path.setText(os.path.basename(self.video_path))
            self.lbl_path.setToolTip(self.video_path)
            self.btn_start.setEnabled(True)
        else:
            self.lbl_path.setText("파일을 선택하거나 드래그하세요")

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
        self._xrange_set = False
        self.lum_window_counts.clear()
        self.red_window_counts.clear()
        self.lum_raw_counts.clear()
        self.lum_risk.clear()
        self.red_risk.clear()
        self.red_raw_counts.clear()
        self.lum_act_counts.clear()
        self.red_act_counts.clear()
        self.curve_lum_act.setData([], [])
        self.curve_red_act.setData([], [])
        self.curve_lum_diag.setData([], [])
        self.curve_red_diag.setData([], [])
        self.lbl_verdict.setText("분석 중...")
        self.lbl_verdict.setStyleSheet(STYLE_IDLE)
        self.lbl_details.setText("")
        self.progress_bar.setValue(0)
        self.fail_segs, self.fail_selected = [], None
        self.lbl_fail_segments.setText("")
        self.btn_full_view.setEnabled(False)
        self.sel_region.setVisible(False)

        # 분석 중에는 차트 조작 비활성화
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)

        # 버튼 상태
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_file.setEnabled(False)

        # 워커 생성
        self.worker = AnalysisWorker(
            video_path=self.video_path,
            downsample=self.spin_downsample.value(),
            enable_pattern=self.chk_pattern.isChecked(),
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

    def _draw_band_curves(self, t, lum_act, red_act, lum_raw, red_raw, lum_risk=None, red_risk=None):
        """곡선 갱신 — engine.py plot_unified와 같은 규칙(공식 PEAT 결과창 실측).
        점선 = 위험도 그대로, 실선 = 전환 수 × 0.5/7. 세로축은 위험도 0~3.5."""
        zero = np.zeros(len(t), dtype=np.float32)
        lum_risk = zero if lum_risk is None else np.asarray(lum_risk, dtype=np.float32)
        red_risk = zero if red_risk is None else np.asarray(red_risk, dtype=np.float32)
        if not self.chk_warnings.isChecked():       # 경고 표시 끔: FAIL 기준 미만은 바닥으로
            lum_risk = np.where(lum_risk >= RISK_FAIL, lum_risk, 0.0)
            red_risk = np.where(red_risk >= RISK_FAIL, red_risk, 0.0)
        self.curve_lum_act.setData(t, risk_to_chart_y(lum_risk))
        self.curve_red_act.setData(t, risk_to_chart_y(red_risk))
        self.curve_lum_diag.setData(t, count_to_chart_y(lum_raw))
        self.curve_red_diag.setData(t, count_to_chart_y(red_raw))

    def _on_frame_result(self, batch):
        """워커가 0.1초마다 묶어 보낸 프레임 결과를 쌓고 그래프를 한 번 갱신."""
        for data in batch:
            self.times.append(data["time"])
            self.lum_window_counts.append(data["lum_window_count"])
            self.red_window_counts.append(data["red_window_count"])
            self.lum_raw_counts.append(data.get("lum_window_raw_count", 0))
            self.red_raw_counts.append(data.get("red_window_raw_count", 0))
            self.lum_act_counts.append(data.get("lum_act_count", data["lum_window_count"]))
            self.red_act_counts.append(data.get("red_act_count", data["red_window_count"]))
            self.lum_risk.append(data.get("lum_risk", 0.0))
            self.red_risk.append(data.get("red_risk", 0.0))
        data = batch[-1]
        # 가로축은 영상 전체 길이로 한 번만 잡는다 (PEAT처럼 왼쪽부터 채워짐). 매번 setXRange하면
        # 눈금·격자 재계산이 분석 스레드를 붙잡아 90초 영상이 28초 → 140초가 됐다.
        if not self._xrange_set and data["processed"] > 1:
            dt = data["time"] / (data["processed"] - 1)
            self.plot_widget.setXRange(0, max(1, data["total"] - 1) * dt + 0.5, padding=0)
            self._xrange_set = True

        t_arr = np.array(self.times)
        lum_window = np.array(self.lum_window_counts, dtype=np.float32)
        red_window = np.array(self.red_window_counts, dtype=np.float32)
        lum_raw = np.array(self.lum_raw_counts, dtype=np.float32)
        red_raw = np.array(self.red_raw_counts, dtype=np.float32)
        lum_act = np.array(self.lum_act_counts, dtype=np.float32)
        red_act = np.array(self.red_act_counts, dtype=np.float32)
        self._draw_band_curves(t_arr, lum_act, red_act, lum_raw, red_raw,
                               np.array(self.lum_risk, dtype=np.float32),
                               np.array(self.red_risk, dtype=np.float32))

        # 실시간 상태
        self.lbl_status.setText(
            f"분석 중... | 시간: {fmt_mmss(data['time'], 1)} | "
            f"휘도 최대: {int(lum_window.max())} | 적색 최대: {int(red_window.max())}"
        )

    def _on_finished(self, summary):
        self._reset_buttons()
        verdict = summary["verdict"]

        # ── 최종 차트: engine.py plot_unified와 완전 동일한 로직 ──
        if "times" in summary and len(summary["times"]) > 0:
            t = np.array(summary["times"])
            lum_window = np.array(summary["lum_window"], dtype=np.float32)
            red_window = np.array(summary["red_window"], dtype=np.float32)
            # 실시간 누적값을 최종값으로 교체 — FAIL 구간 링크·PDF가 판정과 같은 값을 쓰도록
            self.times = list(summary["times"])
            self.lum_window_counts = lum_window.tolist()
            self.red_window_counts = red_window.tolist()
            lum_raw = np.array(summary.get("lum_window_raw", summary["lum_window"]), dtype=np.float32)
            red_raw = np.array(summary.get("red_window_raw", summary["red_window"]), dtype=np.float32)
            lum_act = np.array(summary.get("lum_window_act", summary["lum_window"]), dtype=np.float32)
            red_act = np.array(summary.get("red_window_act", summary["red_window"]), dtype=np.float32)
            ext = np.array(summary["extended_series"], dtype=np.float32)
            lum_risk, red_risk = summary.get("lum_risk"), summary.get("red_risk")
            self._draw_band_curves(t, lum_act, red_act, lum_raw, red_raw,
                                              None if lum_risk is None else np.asarray(lum_risk),
                                              None if red_risk is None else np.asarray(red_risk))

            # Extended Flash (파란선) — ≥0.8일 때만 표시
            self.curve_extended.setData(t, np.where(ext >= 0.8, ext * CHART_BANDS[0], 0.0))

            # 분석 완료 후 줌/슬라이드 활성화
            self.plot_widget.setMouseEnabled(x=True, y=True)
            self.plot_widget.setMenuEnabled(True)
            # 축 범위 제한: 음수 영역 + 영상 길이 밖으로 줌아웃 안 되게
            self.plot_widget.setLimits(xMin=0, xMax=t[-1] + 0.5, yMin=0, yMax=CHART_YMAX)
            self.plot_widget.setXRange(0, t[-1] + 0.5, padding=0)

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

        # FAIL 구간 타임스탬프 표시
        if summary.get("lum_risk") is not None:
            self.lum_risk, self.red_risk = list(summary["lum_risk"]), list(summary["red_risk"])
        self._refresh_risk_segments()
        self.fail_scroll.verticalScrollBar().setValue(0)   # 새 결과는 맨 위부터

        # 내보내기 버튼 활성화
        self.btn_export.setEnabled(True)
        self.btn_full_view.setEnabled(True)

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
    # 키보드: 좌우 방향키로 1프레임 이동
    # ──────────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if self._current_preview_time is None or not hasattr(self, "video_path"):
            super().keyPressEvent(event)
            return

        frame_duration = 1.0 / self._video_fps

        if event.key() == Qt.Key_Left:
            new_time = max(0, self._current_preview_time - frame_duration)
            self.vline.setPos(new_time)
            self._show_frame_at_time(new_time)
        elif event.key() == Qt.Key_Right:
            new_time = self._current_preview_time + frame_duration
            self.vline.setPos(new_time)
            self._show_frame_at_time(new_time)
        else:
            super().keyPressEvent(event)

    # ──────────────────────────────────────────────────────────────────
    # 드래그 앤 드롭
    # ──────────────────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._set_local_video(path)
                self.statusBar().showMessage(f"드롭됨: {path}")
                # 자동 분석 시작
                self._start_analysis()

    # ──────────────────────────────────────────────────────────────────
    # 결과 이미지/PDF 내보내기
    # ──────────────────────────────────────────────────────────────────
    def _export_result(self):
        """차트 + 판정 결과를 PNG/PDF로 저장"""
        if not self.times:
            self.statusBar().showMessage("내보낼 분석 결과가 없습니다.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "결과 내보내기", "flash_analysis_result",
            "PNG Image (*.png);;PDF Document (*.pdf)"
        )
        if not path:
            return

        import pyqtgraph.exporters as exporters

        if path.endswith(".pdf"):
            # PDF: matplotlib로 재생성
            self._export_pdf(path)
        else:
            # PNG: pyqtgraph 차트 캡처
            if not path.endswith(".png"):
                path += ".png"
            exporter = exporters.ImageExporter(self.plot_widget.plotItem)
            exporter.parameters()["width"] = 1920
            exporter.export(path)
            self.statusBar().showMessage(f"이미지 저장: {path}")

    def _export_pdf(self, path):
        """matplotlib로 PDF 생성 (차트 + 판정 정보 포함)"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.ticker import FuncFormatter

        t = np.array(self.times)
        lum_counts = np.array(self.lum_window_counts, dtype=np.float32)
        red_counts = np.array(self.red_window_counts, dtype=np.float32)

        with PdfPages(path) as pdf:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_facecolor("#cfd2d6")
            ax.plot(t, lum_counts, color="white", linewidth=1.5, label="Luminance flash")
            ax.plot(t, red_counts, color="red", linewidth=1.5, label="Red flash")
            ax.axhline(y=FAIL_TRANSITIONS, color="black", linestyle="--", label=f"FAIL ({FAIL_TRANSITIONS})")
            ax.set_xlabel("Time (m:ss)")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_mmss(v)))
            ax.set_ylabel("Transitions / 1s window")
            ax.set_title(f"WCAG 2.3.1 Flash Analysis — {self.lbl_verdict.text()}")
            ax.legend()
            ax.set_ylim(0, max(FAIL_TRANSITIONS + 2, lum_counts.max() + 2))
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # 두 번째 페이지: 판정 정보 + FAIL 구간
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.axis("off")
            info_text = self.lbl_details.text()
            fail_text = self._get_fail_timestamps_text(t, lum_counts, red_counts)
            full_text = f"판정: {self.lbl_verdict.text()}\n\n{info_text}\n\n{fail_text}"
            ax2.text(0.05, 0.95, full_text, transform=ax2.transAxes,
                     fontsize=11, verticalalignment="top", fontfamily="monospace")
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

        self.statusBar().showMessage(f"PDF 저장: {path}")

    # ──────────────────────────────────────────────────────────────────
    # FAIL 구간 타임스탬프
    # ──────────────────────────────────────────────────────────────────
    def _peak_time(self, start, end):
        """구간 안에서 휘도·적색 위험도가 가장 높은 프레임의 시간 (동률이면 앞쪽).
        PEAT 'jump to next defect'처럼 대표 프레임을 보여주기 위함 (2차 QA 결과 #1)."""
        t = np.asarray(self.times)
        risk = np.maximum(np.asarray(self.lum_risk, dtype=np.float32), np.asarray(self.red_risk, dtype=np.float32))
        inside = np.flatnonzero((t >= start) & (t <= end))
        if len(inside) == 0 or len(risk) != len(t):
            return start
        return float(t[inside[np.argmax(risk[inside])]])

    def _full_view(self):
        """그래프를 영상 전체 길이 × 위험도 축 전체로 되돌림."""
        if self.times:
            self.plot_widget.setXRange(0, self.times[-1] + 0.5, padding=0)
        self.plot_widget.setYRange(0, CHART_YMAX, padding=0)

    def _get_fail_timestamps(self, times, lum_counts, red_counts):
        """FAIL 임계값을 넘는 구간의 시작~끝 타임스탬프 리스트 반환"""
        threshold = FAIL_TRANSITIONS
        combined = np.maximum(lum_counts, red_counts)
        in_fail = combined >= threshold

        segments = []
        start = None
        for i, flag in enumerate(in_fail):
            if flag and start is None:
                start = times[i]
            elif not flag and start is not None:
                segments.append((start, times[i - 1]))
                start = None
        if start is not None:
            segments.append((start, times[-1]))

        return segments

    @staticmethod
    def _runs(times, mask):
        """mask가 True인 연속 구간의 (시작, 끝) 시간 리스트."""
        segs, start = [], None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = times[i]
            elif not flag and start is not None:
                segs.append((start, times[i - 1]))
                start = None
        if start is not None:
            segs.append((start, times[-1]))
        return segs

    def _get_risk_segments(self, times, lum_counts, red_counts, lum_risk, red_risk, warnings=True):
        """[(시작, 끝, 'FAIL'|'WARN')]. FAIL = 판정 전환 수 ≥7, 경고 = 위험도 0.1~0.4(FAIL 아님).
        warnings=False면 FAIL만 (PEAT Show warnings 끔)."""
        fail = np.maximum(lum_counts, red_counts) >= FAIL_TRANSITIONS
        segs = [(s, e, "FAIL") for s, e in self._runs(times, fail)]
        if warnings:
            warn = (np.maximum(lum_risk, red_risk) > 0) & ~fail
            segs += [(s, e, "WARN") for s, e in self._runs(times, warn)]
        return segs

    @staticmethod
    def _fail_list_html(segments, selected=None):
        """구간 목록 HTML — FAIL·경고 묶음마다 제목에 개수, 항목마다 번호, 선택한 구간은 배경 강조."""
        if not segments:
            return "위험 구간 없음"
        lines = []
        for kind, title, color in (("FAIL", "🚨 FAIL 구간", "#f7768e"), ("WARN", "⚠️ 경고(CAUTION) 구간", "#e0af68")):
            items = [(i, s, e) for i, (s, e, k) in enumerate(segments) if k == kind]
            if not items:
                continue
            if lines:
                lines.append("<br>")
            lines.append(f"<b>{title} ({len(items)}개):</b><br>")
            for n, (i, s, e) in enumerate(items, 1):
                text = f"#{n} ▶ {fmt_mmss(s, 2)} ~ {fmt_mmss(e, 2)}"
                if i == selected:
                    text = f'<span style="background-color:{color}; color:#1a1b26;">&nbsp;{text}&nbsp;</span>'
                lines.append(f'<a href="seg:{i}" style="color:{color}; text-decoration:none;">{text}</a><br>')
        return "".join(lines)

    def _get_fail_timestamps_text(self, times, lum_counts, red_counts):
        """구간 목록 HTML — PDF 내보내기용 (화면과 같은 경고 표시 설정)"""
        return self._fail_list_html(self._get_risk_segments(
            times, lum_counts, red_counts, np.asarray(self.lum_risk), np.asarray(self.red_risk),
            self.chk_warnings.isChecked()))

    def _refresh_risk_segments(self):
        """현재 결과와 경고 표시 설정으로 구간 목록을 다시 만든다 (선택은 해제)."""
        t = np.array(self.times)
        self.fail_segs = self._get_risk_segments(
            t, np.array(self.lum_window_counts, dtype=np.float32), np.array(self.red_window_counts, dtype=np.float32),
            np.asarray(self.lum_risk, dtype=np.float32), np.asarray(self.red_risk, dtype=np.float32),
            self.chk_warnings.isChecked())
        self._select_fail_segment(None)

    def _on_warnings_toggled(self, _checked):
        """경고 표시 체크박스 — 목록과 점선을 다시 그림 (분석 결과가 있을 때만)."""
        if self.worker is not None or not self.times or len(self.lum_risk) != len(self.times):
            return
        self._refresh_risk_segments()
        self._draw_band_curves(np.array(self.times), None, None,
                               np.array(self.lum_raw_counts, dtype=np.float32),
                               np.array(self.red_raw_counts, dtype=np.float32),
                               np.asarray(self.lum_risk, dtype=np.float32),
                               np.asarray(self.red_risk, dtype=np.float32))

    def _select_fail_segment(self, idx):
        """구간 선택 상태를 목록 강조와 그래프 띠에 반영 (None이면 선택 해제)."""
        self.fail_selected = idx
        bar = self.fail_scroll.verticalScrollBar()
        pos = bar.value()                       # 다시 그려도 스크롤 위치 유지
        self.lbl_fail_segments.setText(self._fail_list_html(self.fail_segs, idx))
        bar.setValue(pos)
        if idx is None:
            self.sel_region.setVisible(False)
        else:
            s, e, _ = self.fail_segs[idx]
            dt = self.times[1] - self.times[0] if len(self.times) > 1 else 0.0
            self.sel_region.setRegion((s, e + dt))      # 끝 프레임까지 포함 (1프레임 구간도 보이게)
            self.sel_region.setVisible(True)

    # ──────────────────────────────────────────────────────────────────
    # FAIL 구간 클릭 → 해당 시간 프레임 표시 + 차트 이동
    # ──────────────────────────────────────────────────────────────────
    def _on_fail_segment_click(self, link):
        """구간 링크 클릭 시 구간 강조 + 구간 안 위험도 최고 프레임으로 수직선·미리보기 이동"""
        if not link.startswith("seg:"):
            return
        idx = int(link[4:])
        self._select_fail_segment(idx)
        seg_start, seg_end, _ = self.fail_segs[idx]
        time_sec = self._peak_time(seg_start, seg_end)

        # 수직선 표시
        self.vline.setPos(time_sec)
        self.vline.setVisible(True)

        # 차트를 해당 구간으로 이동 (구간 앞뒤 2초, 최소 7초 폭)
        self.plot_widget.setXRange(max(0, seg_start - 2), max(seg_end + 2, seg_start + 5), padding=0)

        # 프레임 미리보기
        self._show_frame_at_time(time_sec)

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

        # 클릭한 시간이 든 구간을 선택 (밖이면 해제)
        if self.fail_segs:
            hit = next((i for i, (s, e, _) in enumerate(self.fail_segs) if s <= click_time <= e), None)
            if hit != self.fail_selected:
                self._select_fail_segment(hit)

        # 해당 시간의 프레임 추출
        self._show_frame_at_time(click_time)

    def _refresh_preview(self):
        """하이라이트 토글 시 현재 미리보기 프레임 다시 그리기"""
        if self._current_preview_time is not None and hasattr(self, "video_path"):
            self._show_frame_at_time(self._current_preview_time)

    @staticmethod
    def _violation_masks(frame, prev_frame):
        """직전 프레임 대비 위반 픽셀 마스크 (휘도·적색 플래시, fpaCore 문턱).
        분석은 9프레임 변동폭 기준이지만, 시각화는 프레임간 변화로도 위반 위치를 충분히 보여준다."""
        return flash_mask(frame, prev_frame), red_mask(frame, prev_frame)

    def _show_frame_at_time(self, time_sec):
        """영상에서 특정 시간의 프레임을 추출하여 미리보기에 표시.
        하이라이트 켜짐 + 직전 프레임 존재 시 위반 픽셀을 색으로 표시."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            self._video_fps = fps
            self._current_preview_time = time_sec

            # 직전 프레임부터 순차로 읽어 seek 1회로 두 프레임 확보
            frame_num = int(time_sec * fps)
            prev_frame = None
            if frame_num > 0 and self.chk_highlight.isChecked():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret_p, prev_frame = cap.read()
                if not ret_p:
                    prev_frame = None
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.frame_preview.setText(f"프레임 읽기 실패 (t={time_sec:.2f}s)")
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 위반 영역 오버레이: 노랑=휘도 유해 변화, 청록=적색 전환
            status_extra = ""
            if prev_frame is not None and prev_frame.shape == frame.shape:
                lum_mask, red_mask = self._violation_masks(frame, prev_frame)
                if lum_mask.any() or red_mask.any():
                    out = frame_rgb.astype(np.float32)
                    if lum_mask.any():
                        out[lum_mask] = out[lum_mask] * 0.55 + np.float32([255, 235, 59]) * 0.45
                    if red_mask.any():
                        out[red_mask] = out[red_mask] * 0.55 + np.float32([0, 229, 255]) * 0.45
                    frame_rgb = np.ascontiguousarray(out.astype(np.uint8))
                    total = lum_mask.size
                    status_extra = (f" | 위반 픽셀: 휘도 {lum_mask.sum() / total:.0%}, "
                                    f"적색 {red_mask.sum() / total:.0%}")

            h, w, ch = frame_rgb.shape
            from PyQt5.QtGui import QImage, QPixmap
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # 미리보기 크기에 맞게 스케일
            scaled = pixmap.scaled(
                self.frame_preview.width(),
                self.frame_preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.frame_preview.setPixmap(scaled)
            self.statusBar().showMessage(
                f"프레임 표시: {time_sec:.2f}s (frame #{frame_num}){status_extra}")

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
        QMainWindow, QWidget {
            background-color: #1a1b26;
            color: #c0caf5;
        }
        #inputFrame {
            background-color: #24283b;
            border-radius: 10px;
            border: 1px solid #414868;
        }
        #chartFrame {
            background-color: #24283b;
            border-radius: 10px;
            border: 1px solid #414868;
        }
        #bottomFrame {
            background-color: #24283b;
            border-radius: 8px;
            border: 1px solid #414868;
        }
        #framePreview {
            background-color: #1a1b26;
            color: #565f89;
            font-size: 11px;
            border-radius: 8px;
            border: 1px solid #414868;
        }
        #pathLabel {
            color: #7aa2f7;
            font-size: 12px;
            padding-left: 8px;
        }
        #urlInput {
            background-color: #1a1b26;
            color: #c0caf5;
            border: 1px solid #414868;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
        }
        #urlInput:focus {
            border-color: #7aa2f7;
        }
        #primaryBtn {
            background-color: #7aa2f7;
            color: #1a1b26;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            font-weight: bold;
            font-size: 12px;
        }
        #primaryBtn:hover { background-color: #89b4fa; }
        #primaryBtn:pressed { background-color: #6a92e7; }
        #startBtn {
            background-color: #9ece6a;
            color: #1a1b26;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 13px;
        }
        #startBtn:hover { background-color: #a9d87a; }
        #startBtn:disabled { background-color: #414868; color: #565f89; }
        #stopBtn {
            background-color: #f7768e;
            color: #1a1b26;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
        }
        #stopBtn:hover { background-color: #ff899e; }
        #stopBtn:disabled { background-color: #414868; color: #565f89; }
        #downloadBtn {
            background-color: #bb9af7;
            color: #1a1b26;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }
        #downloadBtn:hover { background-color: #c8a8ff; }
        #exportBtn {
            background-color: transparent;
            color: #7aa2f7;
            border: 1px solid #7aa2f7;
            border-radius: 6px;
            padding: 4px 12px;
            font-size: 11px;
        }
        #exportBtn:hover { background-color: #7aa2f720; }
        #exportBtn:disabled { border-color: #414868; color: #565f89; }
        #statusLabel {
            color: #565f89;
            font-size: 11px;
        }
        #detailsLabel {
            color: #a9b1d6;
            font-size: 11px;
        }
        #failScroll, #failScroll > QWidget > QWidget {
            background: transparent;
        }
        #failSegments {
            color: #f7768e;
            font-size: 14px;
            font-weight: bold;
        }
        #verdict {
            font-size: 20px;
            font-weight: bold;
            padding: 14px;
            border-radius: 10px;
        }
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: #1a1b26;
            text-align: center;
            color: #c0caf5;
            font-size: 10px;
        }
        QProgressBar::chunk {
            background-color: #7aa2f7;
            border-radius: 4px;
        }
        QSpinBox {
            background-color: #1a1b26;
            color: #c0caf5;
            border: 1px solid #414868;
            border-radius: 4px;
            padding: 4px;
        }
        QCheckBox {
            color: #a9b1d6;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 16px; height: 16px;
            border-radius: 4px;
            border: 1px solid #414868;
            background-color: #1a1b26;
        }
        QCheckBox::indicator:checked {
            background-color: #7aa2f7;
            border-color: #7aa2f7;
        }
        QLabel { color: #a9b1d6; }
        QStatusBar {
            background-color: #16161e;
            color: #565f89;
            border-top: 1px solid #414868;
            font-size: 11px;
        }
    """)

    window = PEATMainWindow()
    window.show()
    # 명령줄에 영상 경로를 주면 드롭과 같이 바로 분석 시작
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window._set_local_video(os.path.abspath(sys.argv[1]))
        window._start_analysis()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
