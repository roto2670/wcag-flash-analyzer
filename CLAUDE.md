# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WCAG 2.3.1 비디오 플래시 분석 도구 — PEAT(Photosensitive Epilepsy Analysis Tool) 유사 도구. 비디오에서 휘도(luminance) 및 적색(red) 플래시를 프레임 단위로 분석하여 광과민성 발작 위험을 평가한다.

구성:

- **`peat.py`** — 분석 엔진 + CLI. Harding/ITU-R BT.1702-3 충실 재구현. cd/m² 절대 휘도 기준, 적색 WCAG 작업정의 공식, 줄무늬 패턴 분석, 페이싱 면제, 장면전환(scene cut) 제외까지 포함.
- **`gui.py`** — PyQt5 + pyqtgraph 데스크탑 앱(`PEATMainWindow`). 실시간 차트, 4단계 판정 표시, YouTube URL 다운로드 분석(yt-dlp), 드래그앤드롭, PNG/PDF 내보내기, FAIL 구간 클릭/차트 클릭 → 프레임 미리보기, 좌우 방향키 1프레임 이동.
- **`worker.py`** — `AnalysisWorker(QThread)`. `peat.analyze_video`를 콜백(`on_frame`/`should_stop`)으로 구동하는 얇은 래퍼. 시그널(`progress`/`frame_result`/`finished`/`error`)로 GUI에 전달만 하며 분석 로직은 없다.
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

### 분석 로직 단일 소스

프레임 분석 루프·카운트·판정은 **`peat.py:analyze_video` 한 곳에만 존재**한다. CLI는 콜백 없이 호출하고, GUI는 `worker.py:AnalysisWorker`가 `on_frame`(매 프레임 실시간 데이터 dict 수신 → 시그널 emit)과 `should_stop`(중단 버튼) 콜백을 주입해 같은 함수를 구동한다. `should_stop`이 True를 반환하면 `analyze_video`는 `None`을 반환한다(→ worker는 `finished` 미발생). 분석 알고리즘 수정은 `peat.py`만 고치면 GUI/CLI 모두 반영된다.

차트 y좌표 매핑(`count_to_band_y`, `count_to_activity_y`)도 `peat.py` 정의를 GUI가 import한다.

### 분석 파이프라인 — `peat.py`

공통 파이프라인: `cv2.VideoCapture` 프레임 읽기 → 휘도·적색은 **PEAT식 픽셀 엔진**(`PeatEngine` — 채널별 `_PeatChannel`, sRGB 선형값), 줄무늬 패턴만 **BT.1886 감마 2.4 선형화**(`linearize_video`, `DISPLAY_GAMMA`) → `cd/m² = Y_rel × DISPLAY_PEAK_CD`(기본 200).

> 휘도 규칙은 공식 PEAT 결과 파일(.fpa) 역산으로 맞췄다(2026-10-27): 면적·밝기·횟수·이동·해상도를 하나씩 바꾼 합성 테스트 AVI 63개를 QA PC의 PEAT로 돌려 FPA를 받고, 실제 QA 영상 7개 FPA와 함께 프레임 단위로 대조. FPA 형식·실험 기록은 메모리 `peat-fpa-format` 참고.

1. **휘도 플래시** (`PeatEngine.lum`, `peat.py` 상단 `PEAT_*` 상수):
   - 입력을 폭 `PEAT_WIDTH`=288로 축소(인코딩 공간, INTER_AREA) → sRGB 선형화 + Rec.709 상대휘도 Y(0~1). 감마 2.4는 0↔96 점멸(PEAT FAIL)을 19cd로 환산해 놓쳤고, PEAT 위험도 세기도 sRGB ΔY에 1차식으로 정확히 맞음.
   - 픽셀별 극값 추적 2개: 전환 수용(ΔY ≥ `PEAT_COUNT_DY` 0.048 — 0↔60 안 셈, 64 셈), 플래시용(ΔY ≥ `PEAT_FLASH_DY` 0.095 ≈ 20cd — 0↔86 안 셈, 88 셈, 어두운 쪽 Y ≥ `PEAT_DARK_Y` 0.8이면 면제 — 230↔255 FAIL, 235↔255 면제).
   - **전환 수(진단 실선, `lum_window_raw`)** = 10° 박스(화면 1/3×1/3, 화면 안에 완전히 들어가는 위치만)에서 '최근 1초 전환 ≥k' 픽셀 면적이 25%를 **초과**하는 최대 k (정확히 25% 패치는 PEAT도 0회). PEAT FPA의 휘도 횟수와 같은 값(PEAT는 1프레임 늦게 기록).
   - **움직임 허용**: 지금 값이 직전 프레임 이웃(십자 ±`PEAT_MOTION_R`=5px) 값 범위 안이고 그 반대도 성립하면 이동으로 보고 전환으로 세지 않음(극값은 현재값으로 흡수). 이동 줄무늬 실험: 288폭 기준 프레임당 7px 이하 무시·9px 이상 FAIL, 해상도에 비례(576폭에서 2배 — 그래서 입력을 288폭으로 축소). 경계가 부드러운 사인 줄무늬는 같은 이동에도 FAIL → 블록 추적이 아니라 '주변 값 존재' 방식. 정지 무늬 위 밝기 깜빡임은 셈. 이웃 표본 모양은 합성(흑백 경계)으로 구분 불가 — 실제 7개 FPA에서 십자 r=5가 누락 0·과민 최소(사각 r3: 과민 8, 십자 r7: 누락 5).
   - **FAIL** = 전환 수 ≥7 AND 최근 7프레임(`PEAT_RECENT_S`) 안에 플래시 기준으로 전환한 픽셀의 10° 면적 > `PEAT_RECENT_AREA` 0.10. 깜빡임이 멈추면 1초 창 카운트가 남아 있어도 0.23초 뒤 FAIL 종료(합성 테스트에서 PEAT와 프레임 단위로 일치). 판정 카운트 `lum_window`는 FAIL이 아닌 ≥7 프레임을 6으로 클램프한 값.
   - **위험도**(`lum_risk`, PEAT 보고서 Lum 0.0~3.4, ≥0.5 FAIL) = `_risk_from_area`: 0.459×(A/0.25)^3.07, A = 최근 플래시 픽셀 중 1초 전환 ≥7인 면적(합성 실측: 25.7%→0.5, 30%→0.8, 40%→1.9, 54%↑→3.4, 밝기와 무관). 7회 미만은 5회 0.3·6회 0.4로 상한. 실제 영상의 점선 높이 오차는 FAIL 구간 평균 0.5~1.6(0~3.4 척도)로 아직 근사.
   - 검증(2026-10-27): 실제 QA AVI 7개 FPA — PEAT FAIL 구간 25/25(누락 0), 우리만 잡은 구간 6, 전환 수 프레임 평균 오차 0.33(이전 엔진 24/25, 과민 9, 오차 ≈1.0). 합성 63개 — FAIL 구간 43/43, 과민 6(전부 프레임당 6~8px 이동 줄무늬 경계 사례).
   - ※ 이전의 프레임 단위 카운트(`signs_reversed`·버스트 보정·장면전환 제외·데드존 구제)와 보정 게이트 4종(픽셀 진단·세기·움직임 구제·FAIL 검증)은 **제거됨** — 전부 PEAT 픽셀 규칙과 다른 근사였음. 적색은 아직 이전 방식(아래 2번).
2. **적색 플래시** (`PeatEngine.red`, `PEAT_RED_*` — 합성 테스트 8차로 역산, 2026-10-27): 휘도와 같은 픽셀 엔진 구조(전환 수·FAIL 종료·면적 25% 초과·움직임 허용·위험도 공식 동일). 채널 값 = WCAG 적색값 `max(R−G−B,0)×320`을 **sRGB 선형값**으로 계산 — 감마 2.4는 r75·r80(PEAT FAIL)을 기준 20 아래로 환산해 놓쳤음(sRGB: r70=19.6 안 셈, r75=22.5 셈). 플래시 기준 `PEAT_RED_FLASH`=20, 전환 수 기준 `PEAT_RED_COUNT`=10(r60=14.5도 PEAT가 셈). 전환은 지금 또는 극값 상태가 **포화 적색**일 때만: `R/(R+G+B) ≥ PEAT_RED_SAT` 0.84 — PEAT는 0.8보다 엄격((255,80,80)=0.86 셈, (255,95,95)=0.81 안 셈, 경계는 그 사이). PEAT 적색 세기(FPA w3) = 1.8×적색값 변화+229(혼합색 포함 오차 1% 안). 검증: 합성 적색 21개 FAIL 구간·프레임·전환 수 전부 일치, 실제 14개 적색 전환 수 오차 ≤0.17(적색 FAIL 없음, PEAT도 없음). ※ 이전 프레임 단위 적색 판정(`signs_reversed` 반전 검사·버스트 보정·페이싱 면제)은 제거됨.
3. **줄무늬 패턴**: `detect_stripe_pattern` — 행/열 평균 프로파일 FFT로 지배 주파수 검출. **>5 광-암 쌍** + 광-암차 ≥20 cd/m² + 최고휘도 >50 cd/m²이면 패턴 후보, **≥0.5초 지속** 시 FAIL.
4. **전환 카운트**: 픽셀 엔진의 1초 창 픽셀별 전환 수(위 1·2번). ※ 과거 방식 기록 — **페이싱 면제**: 앞뒤 간격이 **모두** ≥334ms(60Hz)/≥360ms(50Hz)인 '고립' 전환만 안전으로 면제 — 뒤 간격은 알 수 없으므로 잠정 면제 후 다음 이벤트가 간격 안에 오면 소급 카운트(`lum_pending_idx`/`red_pending_idx`). 직전 간격만으로 면제하면 혼합 간격 시퀀스(버스트+갭)에서 실제 4플래시/초가 PASS로 빠지는 위음성이 있었음.
5. **Extended Flash**(`extended_series`): HardingFPA 정의 — FAIL 직전 '경고 수준'(1초 전환 ≥3, FAIL 미만) 플래시가 최근 5초의 80% 이상 지속된 정도(0~1). 실제 FAIL 중이거나 진단선이 CAUTION 밴드(전환 2) 아래면 0. 이전엔 FAIL 프레임·1 flash까지 쳐서 Knox 0:15/1:40, Ballie 40~50초에 파란선이 떴지만 QA 1차 PEAT는 전 영상 Extended 없음. 공식 PEAT 그래프의 파란 계단선.
6. **4단계 판정**(공식 PEAT 밴드 모방, `summary["verdict"]`):
   - **FAIL**: 휘도/적색 전환 ≥7(=>3 flash) 또는 패턴 FAIL. 단 `summary["wcag_2_3_1_fail"]`은 휘도/적색만 반영 — 줄무늬 패턴은 Harding/Ofcom 기준이라 WCAG 2.3.1 범주 밖(verdict에만 반영)
   - **CAUTION (FAIL)**: Extended Flash 경고 발생 또는 최대 전환 ≥6 (pass/fail 라인 직전)
   - **CAUTION (PASS)**: 최대 전환 ≥ 실패의 1/3 — 플래시 활동 경고
   - **PASS**: 그 외

### 차트 매핑 (CLI `plot_unified` / GUI 공통)

4단계 밴드는 y축 하단 30%(`band_top = ymax×0.30`)에 얇게 배치. **선 높이는 자기 최댓값 정규화가 아니라 전환수 임계를 밴드 경계에 고정**(`count_to_band_y`): 0→PASS 바닥, `FAIL_TRANSITIONS//3`→1/4, `FAIL_TRANSITIONS-1`→2/4, `FAIL_TRANSITIONS`→3/4, `2×FAIL_TRANSITIONS`→천장. 따라서 선이 놓인 밴드가 그대로 그 시점의 등급이다.

- **Luminance flash (점선)** = PEAT 위험도(`lum_risk`)를 그대로 그림(`count_to_activity_y(..., risk)`): 0.3/0.4는 CAUTION 밴드, 0.5 이상은 FAIL 경계~천장(3.4) 비례, FAIL 최소 높이 `RISK_MIN_H` 0.10. 아래 '적색/이전' 설명 중 휘도 부분은 과거 방식 기록.
- **Red flash (점선)** = 적색 위험도(`red_risk`)를 휘도와 같은 방식으로 그림. 이 아래는 과거 점선 방식 기록. PEAT 설명서: 점선 높이는 "깜빡임의 밝기와 면적"으로 정해지고 횟수는 실선이 센다. 0.75초 창 카운트의 초당 환산(`lum_window_act`/`red_window_act`)으로 **≤4회는 바닥(0)**, 5·6회는 CAUTION 밴드, 7회 미만은 FAIL 밴드 하단 아래. **FAIL(≥7) 높이** h(0=FAIL 경계, 1=천장 0.95×ymax) = (x − `RISK_CONC_OFFSET` 0.51)/`RISK_CONC_SCALE` 0.23 + `RISK_COUNT_GAIN` 0.075 × (픽셀 진단 횟수 − 7), x = `lum_concurrent` = 1초 창 안 **'동시 전환 면적'**(픽셀 추적기가 그 프레임에 함께 전환한 픽셀의 10° 박스 면적, `PixelTransitionTracker.concurrent`) 상위 `RISK_TOPK`=7개 평균. WCAG "동시에 발생하는 플래시의 결합 면적" 그대로 — 물보라·가는 빛줄기·잔무늬 카메라 이동은 픽셀마다 제각각 깜빡여 1초 누적 면적은 크지만 동시 면적이 작아 낮게, 프로젝터 켜짐·컷처럼 한꺼번에 바뀌면 높게. QA 1차 PEAT 스크린샷 20개 피크 픽셀 실측 대조: 평균 오차 0.120(이전 1초 누적 면적 방식 0.158) — Super Speed 7초 0.15(PEAT 0.14), JET FIT 6~7초 0.15(0.15), Product care 0.95(0.95). 남는 차이: Freestyle+ 12·13번 0.94/1.00(PEAT 0.65/0.68), GCE 3:32 0.58(0.97), ViewFinity 1:19 0.40(0.08). FAIL 구간 최소 높이는 `RISK_MIN_H` 0.10 — PEAT는 딱 7회 FAIL(Lint 19초 0.17, Smart Motion 6초 0.06)도 선 위로 보이게 솟음(이전 0.03은 선에 붙어 낮아 보였음). 적색은 (횟수−7)/9. 실시간 그리기도 `on_frame`의 `lum_concurrent`(최근 1초 동시 전환 면적 상위 7개 평균)와 세기 게이트를 받아 최종 차트와 같은 높이로 그린다(이전엔 횟수만으로 그려 분석 중엔 높게 솟았다가 완료 후 낮아짐). ※ PEAT 실제 차트는 합격/불합격 선이 CAUTION(PASS)와 CAUTION(FAIL) 사이(우리는 FAIL 밴드 하단).
- **Lum/Red flash diag (실선)** = 휘도는 **PEAT식 전환 수**(`lum_window_raw`, PEAT FPA 휘도 횟수와 동일 정의), 적색은 게이트 적용 전 원시 반전 후보 카운트(`red_window_raw`)의 `count_to_band_y` — 공식 PEAT처럼 낮은 활동 구간까지 길게 이어지는 진단선. 판정에는 영향 없음.
- **Extended Flash (파란선)** = `extended_series` ≥0.8(경고 발생)일 때만 표시.

### GUI 흐름 — `gui.py`

- 입력 3경로: 파일 선택 / 드래그앤드롭 / YouTube URL(`DownloadWorker` — yt-dlp subprocess 우선, 실패 시 라이브러리 직접 호출; `%TEMP%/wcag_flash_temp`에 720p 이하 H.264 우선 비디오만 다운로드 후 자동 분석 시작).
- 분석 중: `frame_result` 시그널마다 데이터 누적 + 매 프레임 차트 갱신. 완료 후 summary의 전체 시계열로 최종 차트를 다시 그리고 줌/팬 활성화 — `setLimits(xMin=0, xMax=영상길이+0.5, yMin=0, yMax=10)`로 영상 길이·y축 범위 밖 줌아웃 차단.
- 상호작용: FAIL 구간 타임스탬프 링크·차트 클릭 → `_show_frame_at_time`으로 해당 프레임 미리보기 + 수직선(`vline`); 좌우 방향키로 1프레임씩 이동(`keyPressEvent`). **위반 영역 하이라이트**(`chk_highlight`, 기본 켜짐): 직전 프레임 대비 휘도 유해 변화 픽셀은 노랑, 적색 전환 픽셀은 청록 반투명 오버레이(`_violation_masks` — 분석 루프는 극값 대비지만 시각화는 프레임간 변화로 근사), 상태바에 위반 픽셀 비율 표시. 휘도 강조는 `peat_flash_mask`, 적색 강조는 `peat_red_mask`(PEAT 기준, 원본 해상도).
- 내보내기: PNG은 pyqtgraph `ImageExporter`, PDF는 matplotlib(Agg)로 차트+판정 정보 2페이지 재생성.
- OpenGL은 꺼져 있음 — pyqtgraph OpenGL이 DashLine 스타일을 무시하는 문제(commit c0a17ad). 코덱은 H.264 우선(AV1 디코더의 OS간 차이 회피).

### 핵심 상수 (`peat.py` 상단)

휘도·적색: `PEAT_RED_FLASH=20`, `PEAT_RED_COUNT=10`, `PEAT_RED_SAT=0.84`, `PEAT_WIDTH=288`, `PEAT_COUNT_DY=0.048`, `PEAT_FLASH_DY=0.095`, `PEAT_DARK_Y=0.8`, `PEAT_MOTION_R=5`, `PEAT_RECENT_S=7/30`, `PEAT_RECENT_AREA=0.10`, `RISK_FAIL=0.5`, `RISK_MAX=3.4`. 패턴: `DISPLAY_PEAK_CD=200`, `FLASH_DELTA_CD=20`, `DARK_BOUND_CD=160`, `MICHELSON_THRESH=1/17`, `AREA_THRESHOLD=0.25`(로컬 10° 시야각 영역의 25%), `RED_RATIO_THRESH=0.80`, `RED_DELTA_THRESH=20`, `FAIL_TRANSITIONS=7`, `PATTERN_MIN_PAIRS=5`, `EXTENDED_SECONDS=5.0`, `PATTERN_MIN_SECONDS=0.5`.

### 알려진 근사/한계

- **패턴 면적**: `detect_stripe_pattern`은 검출 시 면적을 1.0으로 근사(블록 단위 정밀 면적 산출 미구현). 정지(>40%)/움직임(>25%) 구분은 프레임 차분 기반 근사. 자연 영상에서 위양성 가능 → `--no_pattern`(CLI) / 체크박스(GUI)로 끌 수 있음.
- **cd/m² 환산**: 디스플레이 백색 200 cd/m² 가정에 의존(HDR 미지원). HLG/PQ 영상은 부정확. `DISPLAY_PEAK_CD`를 바꾸면 `FLASH_DELTA_CD`(20/200=10%)·`DARK_BOUND_CD`(160/200=0.8)도 같은 비율로 조정해야 WCAG 등가 유지.
- **Michelson 분기**: `harmful_luminance_mask`(적색·패턴 보조용으로만 남음)는 어두운 쪽 ≥160 cd/m²에서 Michelson ≥1/17을 유해로 유지하지만, 휘도 판정(픽셀 엔진)은 PEAT 실측대로 완전 면제.
- **mp4 vs PEAT 녹화본**: QA의 PEAT 결과는 브라우저 화면 녹화(288×216, 브라우저 UI·여백 포함)를 분석한 것이라 원본 mp4와 10° 박스·이동 거리의 영상 대비 비율이 다르다. 원본 mp4 분석 시 이전 엔진과 FAIL 구간이 달라진 곳(Super Speed 5.7초 추가, JET FIT Main02 10.5초 추가, YT_5SScN 31.6초 사라짐)은 해당 영상의 PEAT FPA로 확인 필요.
- **공식 PEAT/Harding 엔진은 독점**이라 수치 동일성은 검증 불가 — 표준 기준 재현일 뿐.
