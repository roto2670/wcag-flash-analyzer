# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WCAG 2.3.1 비디오 플래시 분석 도구 — PEAT(Photosensitive Epilepsy Analysis Tool) 재현 도구. 비디오에서 휘도(luminance)·적색(red) 플래시와 줄무늬 패턴을 프레임 단위로 분석해 광과민성 발작 위험을 판정한다.

구성:

- **`peat.py`** — 분석 엔진 + CLI. 휘도·적색은 공식 PEAT 결과 파일(.fpa)을 역산해 맞춘 **PEAT식 픽셀 엔진**(`PeatEngine`), 줄무늬 패턴은 Harding/Wilkins 기준 근사.
- **`gui.py`** — PyQt5 + pyqtgraph 데스크탑 앱(`PEATMainWindow`). 실시간 차트, 4단계 판정 표시, YouTube URL 다운로드 분석(yt-dlp), 드래그앤드롭, PNG/PDF 내보내기, FAIL 구간 클릭/차트 클릭 → 프레임 미리보기, 좌우 방향키 1프레임 이동.
- **`worker.py`** — `AnalysisWorker(QThread)`. `peat.analyze_video`를 콜백(`on_frame`/`should_stop`)으로 구동하는 얇은 래퍼. 시그널(`progress`/`frame_result`/`finished`/`error`)로 GUI에 전달만 하며 분석 로직은 없다.
- **`build.sh`** — Nuitka standalone/onefile 빌드 스크립트 (macOS/Windows/Linux 분기).

> 공식 PEAT의 분석 코어는 Cambridge Research Systems의 Harding FPA 엔진을 라이선스한 **독점 모듈**이다. 코드를 들여다보거나(디컴파일) 복제하지 않고, PEAT를 **블랙박스로 돌려 얻은 결과(FPA)**로 규칙을 역산한다.

## Environment Setup

macOS(현재 주 개발 환경, Python 3.11)와 Windows(Python 3.14) 모두 사용:

```bash
source venv/bin/activate        # macOS/Linux
# source venv/Scripts/activate  # Windows Git Bash
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

`peat.py` CLI 인자: `--area_threshold`(10° 박스 면적 임계, 기본 0.25 초과), `--window`(기본 1.0초), `--downsample N`, `--max_frames N`, `--no_pattern`(줄무늬 패턴 분석 끄기), `--no_show`. 콘솔에는 항상 휘도/적색/패턴별 PASS·FAIL 요약을 출력한다(`print_summary`).

테스트 데이터(전부 gitignore):

- `test_vid/` — QA 원본 mp4와 PEAT 녹화본 AVI(`PEAT_*.avi`, 288×216·30fps rawvideo, 브라우저 화면 포함).
- `test_fpa/` — 위 AVI들의 PEAT 결과 FPA 14개(+ Knox PEAT PDF 보고서). 정답 데이터.
- `temp/` — 회귀용 `PEAT_wuwa.avi`(FAIL 5.5~6.3초), `pass.avi`(PASS), `PEAT_how to monitor your heart rate - browser.avi`.

## Architecture

### 분석 로직 단일 소스

프레임 분석 루프·카운트·판정은 **`peat.py:analyze_video` 한 곳에만 존재**한다. CLI는 콜백 없이 호출하고, GUI는 `worker.py:AnalysisWorker`가 `on_frame`(매 프레임 실시간 데이터 dict 수신 → 시그널 emit)과 `should_stop`(중단 버튼) 콜백을 주입해 같은 함수를 구동한다. `should_stop`이 True를 반환하면 `analyze_video`는 `None`을 반환한다(→ worker는 `finished` 미발생). 분석 알고리즘 수정은 `peat.py`만 고치면 GUI/CLI 모두 반영된다.

차트 y좌표 매핑(`count_to_band_y`, `count_to_activity_y`)과 미리보기 강조 마스크(`peat_flash_mask`, `peat_red_mask`)도 `peat.py` 정의를 GUI가 import한다.

### PEAT 정답 데이터와 역산 방법

PEAT는 분석 결과를 `.fpa`로 저장할 수 있다. 형식(little-endian): 0x118 u32 프레임 수(그 뒤 4바이트는 쓰레기값일 수 있음), 0x120 double fps, 0x142부터 프레임당 16바이트 `<BBBBHHHHHH>`:

| 필드 | 의미 |
|---|---|
| byte0 | 적색 전환 수 |
| byte1 | 휘도 전환 수 (1프레임 늦게 기록) |
| u16@4 / u16@6 | 휘도 세기(w2) / 적색 세기(w3) |
| u16@8 / u16@10 | 적색 위험도(w4) / 휘도 위험도(w5), ×1000 = PEAT 보고서의 Lum/Red 0.0~3.4 |

PEAT의 FAIL 프레임 = 위험도 ≥ 500(0.5). QA PC(Windows, 원격 접속)에서 PEAT를 돌리고, 파일 복사는 원격 앱에서 깨지므로 PowerShell `[Convert]::ToBase64String([IO.File]::ReadAllBytes(...)) | Set-Clipboard`로 옮긴다. 규칙은 **조건을 하나씩 바꾼 합성 테스트 AVI**(면적·밝기·횟수·이동·무늬·해상도·적색, 8차까지 약 120개)를 PEAT로 돌려 역산했다. 생성 스크립트·결과·실험 기록은 메모리 `peat-fpa-format` 참고.

### 분석 파이프라인 — `peat.py`

`cv2.VideoCapture` 프레임 → `PeatEngine.update(frame)`(휘도·적색) + 줄무늬 패턴(감마 2.4 cd/m², `linearize_video`).

**PeatEngine 공통 구조** (`_PeatChannel` 2개 — 휘도 Y, 적색값):

- 입력을 폭 `PEAT_WIDTH`=288로 축소(인코딩 공간, INTER_AREA) → **sRGB 선형화**(`peat_rgb`). 기준 거리·면적이 화면 크기에 비례하기 때문(576폭 테스트에서 이동 허용 거리가 정확히 2배).
- 채널마다 픽셀별 극값·방향 추적기 2개(`_PixelExtremes`): **전환 수용**(낮은 기준)과 **플래시용**(높은 기준). 극값 대비 반대 방향으로 기준 이상 변하면 전환.
- **전환 수**(진단 실선, `*_window_raw`) = 10° 박스(화면 1/3×1/3, 화면 안에 완전히 들어가는 위치만)에서 '최근 1초 전환 ≥k' 픽셀 면적이 25%를 **초과**하는 최대 k. 정확히 25% 패치는 PEAT도 0회.
- **움직임 허용**: 지금 값이 직전 프레임 이웃(십자 ±`PEAT_MOTION_R`=5px) 값 범위 안이고 그 반대도 성립하면 이동으로 보고 전환으로 세지 않음(극값은 현재값으로 흡수). 부드러운 사인 줄무늬는 같은 이동에도 PEAT FAIL → 블록 추적이 아니라 '주변 값 존재' 방식에 가깝다.
- **FAIL** = 전환 수 ≥7 AND 최근 7프레임(`PEAT_RECENT_S`, `age <= recent`) 안에 플래시 기준으로 전환한 픽셀의 10° 면적 > `PEAT_RECENT_AREA` 0.10. 깜빡임이 멈추면 1초 창 카운트가 남아도 0.23초 뒤 FAIL 종료(합성 테스트에서 PEAT와 프레임 단위 일치). 판정 카운트 `*_window`는 FAIL이 아닌 ≥7 프레임을 6으로 클램프한 값.
- **위험도**(`*_risk`, 0~3.4) = `_risk_from_area`: 0.459×(A/0.25)^3.07, A = 최근 플래시 픽셀 중 1초 전환 ≥7인 10° 면적(합성 실측 25.7%→0.5, 30%→0.8, 40%→1.9, 54%↑→3.4, 밝기와 무관). 7회 미만이면 5회 0.3·6회 0.4로 상한.

1. **휘도** (`PeatEngine.lum`): Y = Rec.709 상대휘도(sRGB 선형). 전환 기준 `PEAT_COUNT_DY` 0.048(0↔60 안 셈, 64 셈), 플래시 기준 `PEAT_FLASH_DY` 0.095 ≈ 20cd(0↔86 안 셈, 88 셈), 어두운 쪽 Y ≥ `PEAT_DARK_Y` 0.8이면 플래시 아님(230↔255 FAIL, 235↔255 면제). 감마 2.4는 0↔96(PEAT FAIL)을 19cd로 환산해 놓쳤다. PEAT 휘도 세기(w2) = 545×ΔY+334(sRGB, 오차 0.3%).
2. **적색** (`PeatEngine.red`): 값 = WCAG 적색값 `max(R−G−B,0)×320`(sRGB 선형). 플래시 기준 `PEAT_RED_FLASH` 20(r70=19.6 안 셈, r75=22.5 셈), 전환 기준 `PEAT_RED_COUNT` 10(r60=14.5도 셈). 지금 또는 극값 상태가 **포화 적색**(`R/(R+G+B) ≥ PEAT_RED_SAT` 0.84)일 때만 전환 — PEAT는 0.8보다 엄격((255,80,80)=0.86 셈, (255,95,95)=0.81 안 셈). PEAT 적색 세기(w3) = 1.8×적색값 변화+229. 이전의 감마 2.4·0.8 기준은 r75·r80 FAIL을 놓쳤다.
3. **줄무늬 패턴**: `detect_stripe_pattern` — 행/열 평균 프로파일 FFT로 지배 주파수 검출. **>5 광-암 쌍** + 광-암차 ≥20 cd/m² + 최고휘도 >50 cd/m²이면 패턴 후보, **≥0.5초 지속** 시 FAIL. PEAT에는 없는 축(Harding/Ofcom 기준).
4. **Extended Flash**(`extended_series`): HardingFPA 정의 — FAIL 직전 '경고 수준'(1초 전환 ≥3, FAIL 미만) 플래시가 최근 5초의 80% 이상 지속된 정도(0~1). 실제 FAIL 중이거나 진단선이 CAUTION 밴드(전환 2) 아래면 0. 공식 PEAT 그래프의 파란 계단선. FPA로는 아직 검증 안 됨.
5. **4단계 판정**(`summary["verdict"]`):
   - **FAIL**: 휘도/적색 FAIL 프레임 존재 또는 패턴 FAIL. 단 `summary["wcag_2_3_1_fail"]`은 휘도/적색만 반영(패턴은 WCAG 2.3.1 범주 밖)
   - **CAUTION (FAIL)**: Extended Flash 경고 또는 최대 판정 카운트 ≥6
   - **CAUTION (PASS)**: 최대 판정 카운트 ≥2(실패의 1/3)
   - **PASS**: 그 외

**검증 현황** (2026-10-27):

- 합성 테스트: 휘도·적색 깜빡임은 FAIL 구간·프레임·전환 수가 PEAT와 일치. 움직임 테스트에서 프레임당 6~8px 이동 줄무늬 등 경계 사례 6개는 우리만 FAIL.
- 실제 PEAT 녹화본 14개(test_fpa): 판정 14/14 일치, PEAT 휘도 FAIL 32구간 전부 검출(누락 0), 우리만 잡은 짧은 구간 10개, 전환 수 프레임 평균 오차 약 0.35, 적색 전환 수 오차 ≤0.17.
- 남은 과민 구간(Super Speed 5.3~6.0초 세탁조 물 소용돌이, Flipsuit 6초·29.8초 등)은 **움직임 판정**이 PEAT와 달라서다. 7차까지 역산했지만 정확한 방식은 미해결: 줄무늬 간격 24는 FAIL·32는 통과·64는 FAIL(모두 8px 이동), 체커보드·잔 무작위 무늬는 16~32px 이동도 통과, 크고 흐린 무늬는 12px부터 FAIL. 범위검사·허용오차·공간평균·블록매칭·위상상관 모두 불일치. 판정 기준값(면적·최근면적)을 바꿔도 과민 구간은 안 줄고 진짜 FAIL부터 놓친다.

### 차트 매핑 (CLI `plot_unified` / GUI 공통)

4단계 밴드는 y축 하단 `BAND_TOP_FRAC`(0.22)에 얇게 배치. **선 높이는 전환수 임계를 밴드 경계에 고정**(`count_to_band_y`): 0→PASS 바닥, `FAIL_TRANSITIONS//3`→1/4, `FAIL_TRANSITIONS-1`→2/4, `FAIL_TRANSITIONS`→3/4, `2×FAIL_TRANSITIONS`→천장. 선이 놓인 밴드가 그 시점의 등급이다.

- **Luminance / Red flash (점선)** = PEAT 위험도(`lum_risk`/`red_risk`)를 그대로 그림(`count_to_activity_y(counts, band_top, ymax, risk)`): 0.2 이하는 바닥, 0.3/0.4는 CAUTION 밴드, 0.5 이상은 FAIL 경계~천장(3.4)에 비례. 판정 카운트가 FAIL 미만이면 FAIL 밴드에 올리지 않음. FAIL 최소 높이 `RISK_MIN_H` 0.10. 실제 영상에서 점선 높이는 아직 근사(FAIL 구간 평균 오차 0.5~1.6 / 3.4 척도).
- **Lum/Red flash diag (실선)** = 전환 수(`*_window_raw`, PEAT FPA 횟수와 같은 정의)의 `count_to_band_y`. GUI는 '플래시 수'(전환÷2) 스케일로 그린다.
- **Extended Flash (파란선)** = `extended_series` ≥0.8일 때만 표시.

### GUI 흐름 — `gui.py`

- 입력 3경로: 파일 선택 / 드래그앤드롭 / YouTube URL(`DownloadWorker` — yt-dlp subprocess 우선, 실패 시 라이브러리 직접 호출; `%TEMP%/wcag_flash_temp`에 720p 이하 H.264 우선 비디오만 다운로드 후 자동 분석 시작).
- 분석 중: `frame_result` 시그널마다 데이터 누적 + 매 프레임 차트 갱신(실시간 dict의 `lum_risk`/`red_risk`로 최종 차트와 같은 점선). 완료 후 summary의 전체 시계열로 최종 차트를 다시 그리고 줌/팬 활성화 — `setLimits(xMin=0, xMax=영상길이+0.5, yMin=0, yMax=10)`.
- 상호작용: FAIL 구간 타임스탬프 링크·차트 클릭 → `_show_frame_at_time`으로 프레임 미리보기 + 수직선(`vline`); 좌우 방향키로 1프레임씩 이동(`keyPressEvent`). **위반 영역 하이라이트**(`chk_highlight`, 기본 켜짐): 직전 프레임 대비 휘도 플래시 픽셀은 노랑(`peat_flash_mask`), 적색 플래시 픽셀은 청록(`peat_red_mask`) 반투명 오버레이 — 분석은 극값 대비지만 시각화는 프레임간 변화로 근사, 원본 해상도. 상태바에 위반 픽셀 비율 표시.
- 내보내기: PNG은 pyqtgraph `ImageExporter`, PDF는 matplotlib(Agg)로 차트+판정 정보 2페이지 재생성.
- OpenGL은 꺼져 있음 — pyqtgraph OpenGL이 DashLine 스타일을 무시하는 문제(commit c0a17ad). 코덱은 H.264 우선(AV1 디코더의 OS간 차이 회피).

### 핵심 상수 (`peat.py` 상단)

- 휘도·적색 엔진: `PEAT_WIDTH=288`, `PEAT_COUNT_DY=0.048`, `PEAT_FLASH_DY=0.095`, `PEAT_DARK_Y=0.8`, `PEAT_RED_COUNT=10`, `PEAT_RED_FLASH=20`, `PEAT_RED_SAT=0.84`, `PEAT_MOTION_R=5`, `PEAT_RECENT_S=7/30`, `PEAT_RECENT_AREA=0.10`, `AREA_THRESHOLD=0.25`, `FAIL_TRANSITIONS=7`, `RISK_FAIL=0.5`, `RISK_MAX=3.4`, `RISK_MIN_H=0.10`.
- 패턴·Extended: `DISPLAY_GAMMA=2.4`, `DISPLAY_PEAK_CD=200`, `FLASH_DELTA_CD=20`, `PATTERN_MIN_PAIRS=5`, `PATTERN_LIGHT_CD=50`, `PATTERN_MIN_SECONDS=0.5`, `EXTENDED_SECONDS=5.0`.

### 알려진 근사/한계

- **움직임 판정**: 위 '검증 현황' 참고 — 큰 움직임(물보라·회전·빠른 패닝)에서 PEAT보다 예민할 수 있다(안전 쪽 오차).
- **포화 적색 경계**: 0.81~0.86 사이까지만 확인해 0.84를 씀.
- **mp4 vs PEAT 녹화본**: PEAT 정답은 브라우저 화면 녹화(288×216, 브라우저 UI·여백 포함, 24fps 영상은 30fps로 녹화)에 대한 것이다. 원본 mp4를 직접 분석하면 10° 박스·이동 허용 거리의 영상 대비 비율과 프레임 간격이 달라 결과가 조금 다를 수 있다. 엔진은 AVI(PEAT 입력) 기준으로 맞춘다.
- **패턴 면적**: `detect_stripe_pattern`은 검출 시 면적을 1.0으로 근사. 자연 영상에서 위양성 가능 → `--no_pattern`(CLI) / 체크박스(GUI)로 끌 수 있음.
- **HDR 미지원**: SDR sRGB 가정. HLG/PQ 영상은 부정확.
