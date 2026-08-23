# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WCAG 2.3.1 비디오 플래시 분석 도구 — PEAT(Photosensitive Epilepsy Analysis Tool) 유사 도구. 비디오에서 휘도(luminance) 및 적색(red) 플래시를 프레임 단위로 분석하여 광과민성 발작 위험을 평가한다.

구성:

- **`peat.py`** — 분석 엔진 + CLI. Harding/ITU-R BT.1702-3 충실 재구현. cd/m² 절대 휘도 기준, 적색 WCAG 작업정의 공식, 줄무늬 패턴 분석, 페이싱 면제, 장면전환(scene cut) 제외까지 포함.
- **`gui.py`** — PyQt5 + pyqtgraph 데스크탑 앱(`PEATMainWindow`). 실시간 차트, 4단계 판정 표시, YouTube URL 다운로드 분석(yt-dlp), 드래그앤드롭, PNG/PDF 내보내기, FAIL 구간 클릭/차트 클릭 → 프레임 미리보기, 좌우 방향키 1프레임 이동.
- **`worker.py`** — `AnalysisWorker(QThread)`. 프레임 단위 분석을 백그라운드에서 수행하며 시그널(`progress`/`frame_result`/`finished`/`error`)로 GUI에 전달.
- **`build.sh`** — Nuitka standalone/onefile 빌드 스크립트 (macOS/Windows/Linux 분기).

> 공식 PEAT의 분석 코어는 Cambridge Research Systems의 Harding FPA 엔진을 라이선스한 **독점 모듈**이라 코드 복제가 불가능하다. `peat.py`는 PEAT가 따르는 **표준(WCAG 2.3.1 + ITU-R BT.1702-3 + Harding/Wilkins)**을 직접 구현한 것이다.

## Environment Setup

Windows 개발 환경. venv는 Python 3.14 (Windows 레이아웃):

```bash
source venv/Scripts/activate   # Git Bash
# venv\Scripts\activate        # cmd/PowerShell
pip install -r requirements.txt
```

의존성: `PyQt5`, `pyqtgraph`, `opencv-python-headless`, `numpy`, `matplotlib`(PDF 내보내기·CLI 그래프), `yt-dlp`(YouTube 다운로드), `nuitka`(exe 빌드).

## Running

```bash
python gui.py                                       # GUI 앱 (주 사용 경로)
python peat.py --video ./temp/<영상파일>            # CLI 분석 + Matplotlib 그래프
python peat.py --video ./temp/<영상파일> --no_show  # 그래프 없이 콘솔 요약만
./build.sh                                          # Nuitka로 PEAT_Analyzer 실행파일 빌드 (venv 활성화 필요)
```

테스트 비디오는 `temp/` 디렉터리에 있음 (gitignore됨). 큰 영상은 `--max_frames`/`--downsample`로 제한해서 돌릴 것.

`peat.py` CLI 인자: `--area_threshold`(기본 0.25), `--window`(기본 1.0초), `--downsample N`, `--max_frames N`, `--no_pattern`(줄무늬 패턴 분석 끄기), `--no_show`.

`peat.py`는 항상 콘솔에 휘도/적색/패턴별 PASS·FAIL 요약을 출력한다(`print_summary`).

## Architecture

### 분석 로직 이중화 — 중요

**프레임 분석 루프와 최종 판정 로직이 두 곳에 존재한다**: `peat.py:analyze_video`(CLI·일괄 처리)와 `worker.py:AnalysisWorker._analyze`(GUI·실시간 시그널 방출). worker는 peat의 헬퍼 함수·상수를 import하지만 루프 자체는 복제본이다. **분석 알고리즘(플래시 감지, 방향 추적, 판정 규칙)을 수정할 때는 반드시 두 파일을 함께 수정할 것.**

차트 y좌표 매핑(`count_to_band_y`, `area_to_band_y`, `_rolling_max`)은 `peat.py`에 정의되어 GUI가 import하므로 이 부분은 단일 소스다.

### 분석 파이프라인 — `peat.py`

공통 파이프라인: `cv2.VideoCapture` 프레임 읽기 → sRGB 선형화(breakpoint **0.04045**) → 절대 휘도 `cd/m² = Y_rel × DISPLAY_PEAK_CD`(기본 200, SDR 가정).

1. **휘도 플래시**: 극값(피크/밸리) 프레임 대비 **유해 전환 픽셀을 방향별(밝아짐/어두워짐)로 분리**해 각각의 **로컬 10° 시야각 면적**(`local_area_fraction`: 화면 1/3×1/3 박스의 최대 깜빡임 밀도, 화면 안에 완전히 들어가는 박스만)을 계산. 지배 방향의 면적이 ≥25%이고 방향이 직전과 반대(opposing)면 전환 1회로 카운트. 방향 판정 규칙:
   - up/down **양쪽 모두** ≥25%면 `signs_reversed`(방향 블록 지도 상관)로 **같은 영역이 실제 반전했는지** 확인 — 반전이면 opposing(역위상 점멸), 아니면 지배 방향 유지(모션/줌은 up/down이 공존해도 각 영역 부호가 유지되므로 제외)
   - 유해 픽셀이 있으면 지배 방향(면적 큰 쪽), 없으면 전역 평균 부호로 극값 드리프트만 추적
   - ※ 전체 화면 25%가 아니라 로컬 영역 25% — WCAG "10° 시야각" 기준이라 국소 플래시 포착. 방향도 로컬(방향별 면적) 기준 — 전역 평균 방향은 화면 구석의 국소 플래시를 놓침(test-2에서 수정됨).
   - 유해 전환 = `harmful_luminance_mask`: 어두운쪽 <160 cd/m²면 차이 ≥**20 cd/m²**, ≥160 cd/m²면 **Michelson 대비 ≥1/17**.
   - **장면전환 제외**: 직전 프레임 대비 로컬 면적 >0.7 급변이 1초 내 ≤2회로 '고립'되면 scene cut으로 보고 카운트 제외. 반복 점멸(급변 다수)은 유지.
2. **적색 플래시**: WCAG 작업정의 — 한쪽 상태가 `R/(R+G+B) ≥ 0.8`(포화 적색) **AND** `(R−G−B)×320` 변화 > **20**(`red_flash_value`, 음수는 0)인 픽셀을 적색 **진입(+)/이탈(−) 방향별 로컬 면적**으로 판정 — 휘도와 동일 구조(지배 방향, 양방향 동시 유의 시 `signs_reversed` 동일 영역 반전 검사). 휘도와 독립된 방향/윈도우 추적.
3. **줄무늬 패턴**: `detect_stripe_pattern` — 행/열 평균 프로파일 FFT로 지배 주파수 검출. **>5 광-암 쌍** + 광-암차 ≥20 cd/m² + 최고휘도 >50 cd/m²이면 패턴 후보, **≥0.5초 지속** 시 FAIL.
4. **전환 카운트**: 1초 슬라이딩 윈도우 내 휘도/적색 각각 전환 수. **페이싱 면제**: leading edge 간격 ≥334ms(60Hz)/≥360ms(50Hz) 전환은 안전으로 보고 누적 제외.
5. **Extended Flash**(`extended_series`): 5초 윈도우의 80% 프레임에서 실패의 1/3(=전환 `FAIL_TRANSITIONS//3`) 이상 플래시가 지속되는 정도(0~1). 공식 PEAT 그래프의 파란 계단선.
6. **4단계 판정**(공식 PEAT 밴드 모방, `summary["verdict"]`):
   - **FAIL**: 휘도/적색 전환 ≥7(=>3 flash) 또는 패턴 FAIL — 명백한 WCAG 위반
   - **CAUTION (FAIL)**: Extended Flash 경고 발생 또는 최대 전환 ≥6 (pass/fail 라인 직전)
   - **CAUTION (PASS)**: 최대 전환 ≥ 실패의 1/3 — 플래시 활동 경고
   - **PASS**: 그 외

### 차트 매핑 (CLI `plot_unified` / GUI 공통)

4단계 밴드는 y축 하단 30%(`band_top = ymax×0.30`)에 얇게 배치. **선 높이는 자기 최댓값 정규화가 아니라 전환수 임계를 밴드 경계에 고정**(`count_to_band_y`): 0→PASS 바닥, `FAIL_TRANSITIONS//3`→1/4, `FAIL_TRANSITIONS-1`→2/4, `FAIL_TRANSITIONS`→3/4, `2×FAIL_TRANSITIONS`→천장. 따라서 선이 놓인 밴드가 그대로 그 시점의 등급이다.

- **Luminance flash / Red flash (점선)** = 메인 활동선. 휘도는 `lum_window` 카운트의 밴드 매핑; 적색은 `sat_area`(포화적색 존재 면적, 전환 무관)의 1초 rolling max를 `area_to_band_y`로 — 지속 적색도 표시되도록.
- **Lum/Red flash diag (실선)** = 윈도우 카운트(보조).
- **Extended Flash (파란선)** = `extended_series` ≥0.8(경고 발생)일 때만 표시.

### GUI 흐름 — `gui.py`

- 입력 3경로: 파일 선택 / 드래그앤드롭 / YouTube URL(`DownloadWorker` — yt-dlp subprocess 우선, 실패 시 라이브러리 직접 호출; `%TEMP%/wcag_flash_temp`에 720p 이하 H.264 우선 비디오만 다운로드 후 자동 분석 시작).
- 분석 중: `frame_result` 시그널마다 데이터 누적 + 매 프레임 차트 갱신. 완료 후 summary의 전체 시계열로 최종 차트를 다시 그리고 줌/팬 활성화.
- 상호작용: FAIL 구간 타임스탬프 링크·차트 클릭 → `_show_frame_at_time`으로 해당 프레임 미리보기 + 수직선(`vline`); 좌우 방향키로 1프레임씩 이동(`keyPressEvent`).
- 내보내기: PNG은 pyqtgraph `ImageExporter`, PDF는 matplotlib(Agg)로 차트+판정 정보 2페이지 재생성.
- OpenGL은 꺼져 있음 — pyqtgraph OpenGL이 DashLine 스타일을 무시하는 문제(commit c0a17ad). 코덱은 H.264 우선(AV1 디코더의 OS간 차이 회피).

### 핵심 상수 (`peat.py` 상단)

`DISPLAY_PEAK_CD=200`, `FLASH_DELTA_CD=20`, `DARK_BOUND_CD=160`, `MICHELSON_THRESH=1/17`, `AREA_THRESHOLD=0.25`(로컬 10° 시야각 영역의 25%), `RED_RATIO_THRESH=0.80`, `RED_DELTA_THRESH=20`, `FAIL_TRANSITIONS=7`, `PATTERN_MIN_PAIRS=5`, `EXTENDED_SECONDS=5.0`, `PATTERN_MIN_SECONDS=0.5`.

### 알려진 근사/한계

- **패턴 면적**: `detect_stripe_pattern`은 검출 시 면적을 1.0으로 근사(블록 단위 정밀 면적 산출 미구현). 정지(>40%)/움직임(>25%) 구분은 프레임 차분 기반 근사. 자연 영상에서 위양성 가능 → `--no_pattern`(CLI) / 체크박스(GUI)로 끌 수 있음.
- **cd/m² 환산**: 디스플레이 백색 200 cd/m² 가정에 의존(HDR 미지원). HLG/PQ 영상은 부정확.
- **공식 PEAT/Harding 엔진은 독점**이라 수치 동일성은 검증 불가 — 표준 기준 재현일 뿐.
