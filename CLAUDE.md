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

공통 파이프라인: `cv2.VideoCapture` 프레임 읽기 → **BT.1886 감마 2.4 선형화**(`linearize_video`, `DISPLAY_GAMMA`) → 절대 휘도 `cd/m² = Y_rel × DISPLAY_PEAK_CD`(기본 200, SDR 가정). 감마 2.4는 ITU-R BT.1702-3 Table 1(신호 0.52 → 41.7 cd/m²) 그대로 — 이전의 웹용 sRGB(선형 토+≈2.2)는 암부를 밝게 환산해 어두운 장면 밝기 차이를 과장했고(QA 1차 Super Speed 5.7초 과민), 2.4로 바꾼 뒤 QA 1차 12개 영상 판정·FAIL 지점이 PEAT와 전부 일치(누락 0 / 과민 0, 2026-10-15). 적색 공식도 같은 선형 RGB를 쓴다.

1. **휘도 플래시**: 극값(피크/밸리) 프레임 대비 **유해 전환 픽셀을 방향별(밝아짐/어두워짐)로 분리**해 각각의 **로컬 10° 시야각 면적**(`local_area_fraction`: 화면 1/3×1/3 박스의 최대 깜빡임 밀도, 화면 안에 완전히 들어가는 박스만)을 계산. **유의성은 결합(전체 유해) 면적 ≥25%로 판정**(WCAG "동시 발생 플래시의 결합 면적" — 역위상 점멸은 방향별로 25% 미만일 수 있음)하고, 방향이 직전과 반대(opposing)면 전환 1회로 카운트. 역위상 분기는 양방향 각각 ≥12.5%(=임계의 절반) + 결합 ≥25%일 때 진입. 방향 판정 규칙:
   - **모든 전환 카운트는 `signs_reversed`(방향 블록 지도 상관 < −0.02) 필수** — WCAG의 "동일 영역의 opposing change" 그대로. 지배 박스가 근소한 차이로 뒤바뀌는 모션/줌/제품 회전은 부호 지도가 유지되어(상관 양수) 제외되고, 진짜 점멸은 같은 블록의 부호가 반전되어(상관 음수) 통과. 임계가 −0.2처럼 깊으면 장면 전체가 바뀌며 점멸하는 컷 점멸(게임 트레일러)의 희석된 상관(−0.2~−0.02)을 놓침 — PEAT_wuwa 실측으로 −0.02 확정(진짜 점멸 −0.03~−0.85, 모션 +0.02~+0.48). 반전 후보였으나 검사 탈락 시 카운트 없이 방향/극값만 전환(다음 실제 반전을 놓치지 않기 위함). **겹침 없음 폴백**: 두 지도의 활성 블록이 겹치지 않으면(플래시 영역 이동/점프 — 교차 스트로브) 각 지도의 지배 부호 곱 < −0.36으로 판정 — 점멸은 지도가 모노 방향 + 서로 반대, 모션은 전연(상승)/후연(하강)이 한 지도에 섞여 지배 부호가 약해 False 유지
   - up/down 양쪽 모두 ≥25%(역위상 동시 점멸 후보)일 때도 동일 검사로 판정
   - 유해 픽셀이 있으면 지배 방향(면적 큰 쪽), 없으면 전역 평균 부호로 극값 드리프트만 추적
   - ※ 전체 화면 25%가 아니라 로컬 영역 25% — WCAG "10° 시야각" 기준이라 국소 플래시 포착. 방향도 로컬(방향별 면적) 기준 — 전역 평균 방향은 화면 구석의 국소 플래시를 놓침(test-2에서 수정됨).
   - 유해 전환 = `harmful_luminance_mask`: 어두운쪽 <160 cd/m²면 차이 ≥**20 cd/m²**, ≥160 cd/m²면 **Michelson 대비 ≥1/17**.
   - **버스트 시작 보정(오프바이원)**: 1초 이상 유의 변화가 없다가 나타난 첫 유의 전환도 1회로 카운트(`lum_burst_start`/`red_burst_start`, `signs_reversed` 불요 — 직전 기록이 없으므로). 반전만 세면 물리 전환 N회가 N−1회로 집계돼 FAIL에 8전환이 필요해지는 문제 보정 — Harding 관행은 1초 내 전환 ≥7회 FAIL.
   - **장면전환 제외**: 직전 프레임 대비 로컬 면적 >0.7 급변이 1초 내 ≤2회로 '고립'되면 scene cut으로 보고 잠정 제외(`lum_cut_pending`). 급변이 1초 내 3회 이상으로 늘어나면 반복 점멸로 판명 → 잠정 제외분 소급 복원(버스트 초반 1~2 전환 누락 방지). **순서 중요**: 페이싱 면제가 장면전환 검사보다 먼저 — 복원 대상을 '페이싱을 통과한' 이벤트로 한정해야 저속 점멸이 복원 경로로 면제를 우회하지 않는다.
   - **corr 데드존 구제**: `signs_reversed` 상관이 ±0.02 안(모션/점멸 판별 불능 구간)인 반전 후보는 10° 로컬 평균 휘도 변화 ≥20 cd(`local_mean_map`, `LOCAL_MEAN_GRID` 12×16, WCAG 플래시 정의 상당)이면 전환 인정 — Lint filter 19초(티슈 연속 투하 = 진짜 점멸 상당)에서 PEAT는 FAIL인데 corr +0.009 프레임 하나가 잘려 6전환에 그치던 위음성 보정.
   - **Harding식 픽셀 진단 게이트**(`PixelTransitionTracker`, `PIX_W`=160, `PIX_TOL_S`=0.3): HardingFPA 매뉴얼 원문 — "각 픽셀이 최근 1초 동안 겪은 전환 수(움직임 허용 후) … 이미지의 1/4 이상이 7회 이상이면 FAIL", 진단값 = "가장 활발한 25%가 겪은 최소 전환 수". 160폭 저해상도로 픽셀별 극값/방향/1초 전환 이력을 추적하고, 10° 박스(`local_area_fraction`) 안 25% 이상이 k회 이상인 최대 k를 진단값으로 계산. 프레임 카운트 FAIL이라도 ±0.3초 안에 픽셀 진단 FAIL이 없으면 6으로 클램프 — 매 프레임 '다른' 픽셀이 바뀌는 카메라 줌/텍스처를 걸러냄(Ballie 40초 제거). 진단선(실선 `lum_window_raw`)도 이 값으로 그림(Harding 진단선 정의). 추적기 휘도도 `DISPLAY_GAMMA` 2.4로 환산하고 축소는 인코딩 공간에서 먼저 — sRGB 대비 암부가 더 어두워 어두운 배경의 작은 반짝임(Knox 0:37 경기장)이 빠짐(해상도 96/240, 감마 2.6/2.8은 모두 악화). **움직임 보정은 넣지 않음**(전역 워프·크기 상한·구조 정규화 플로우·DIS·양방향 일치 검사 모두 시도): 밀집 옵티컬 플로우(Farneback/DIS, 구조 정규화 포함)로 픽셀 상태를 워프하면 PEAT가 FAIL로 잡는 빠른 카메라 이동+플래시(ViewFinity 1:12·1:19, Ballie 5.5·25초, Knox 2:31)까지 지워짐 — PEAT의 '움직임 허용'은 매우 약한 것으로 판단. 화면 전체 25% 기준(Harding 방송용)은 누락 10으로 과둔감 — PEAT는 341×256(10°) 박스 기준이 맞음.
   - **플래시 세기 게이트**(`LUM_MIN_WINDOW_INTENSITY_CD`=11, `_window_mean_delta`): PEAT 설명서 "점선(위험도) 높이는 깜빡임의 밝기와 면적으로 결정", Harding "위험도 선이 합격선을 넘으면 FAIL" — 횟수만이 아니라 세기도 필요. 전환마다 극값 대비 10° 영역 평균 휘도 변화를 기록하고, FAIL 창(1초) 안 카운트된 전환들의 **중간값**이 11cd 미만이면 6으로 클램프. 전환을 개별로 걸러내면(이전 시도) 정확히 7회인 PEAT FAIL(Lint 19초)이 깨지지만 창 단위 중간값은 횟수를 건드리지 않음. QA 1차 실측(창 중간 세기): PEAT 무이슈 Ballie 30초 10.7 / Super Speed 5.7초 11.1, PEAT FAIL Knox 2:55 11.6 · Ballie 25초 13.6 · 나머지 15~58. 여유가 0.3~0.6cd로 얇음 — 2차 QA에서 재검증 필요. 평균·75퍼센타일 통계는 더 나쁨.
   - ※ 과거의 '휘도 FAIL 검증 게이트'(10° 로컬 평균 반전 ≥6 또는 스윙 ≥70cd 없으면 클램프)는 **제거됨**(2026-10-15) — 픽셀 진단·세기 게이트가 같은 역할을 하게 된 뒤로 과민 차단 효과가 0이었고, QA #12 Freestyle+ 2초(PEAT FAIL)를 지우는 누락만 일으켰음. 제거 후 QA 1차 12개 영상: 누락 0 / 과민 1(Super Speed 5.7초).
2. **적색 플래시**: WCAG 작업정의 — 한쪽 상태가 `R/(R+G+B) ≥ 0.8`(포화 적색) **AND** `(R−G−B)×320` 변화 > **20**(`red_flash_value`, 음수는 0)인 픽셀을 적색 **진입(+)/이탈(−) 방향별 로컬 면적**으로 판정 — 휘도와 동일 구조(지배 방향, 양방향 동시 유의 시 `signs_reversed` 동일 영역 반전 검사). 휘도와 독립된 방향/윈도우 추적. **색공간은 선형 RGB가 정답**(W3C 원문이 R,G,B를 상대 휘도 정의의 선형화 값으로 위임 — 2026-08 검증 완료). ×320 방식은 WCAG 2.0 시대 정의 = 공식 PEAT 구현이며, 2022년 갱신 정의(CIE UCS 색차 >0.2)와 둘 다 W3C 인정 — PEAT 재현 목표상 ×320 유지.
3. **줄무늬 패턴**: `detect_stripe_pattern` — 행/열 평균 프로파일 FFT로 지배 주파수 검출. **>5 광-암 쌍** + 광-암차 ≥20 cd/m² + 최고휘도 >50 cd/m²이면 패턴 후보, **≥0.5초 지속** 시 FAIL.
4. **전환 카운트**: 1초 슬라이딩 윈도우 내 휘도/적색 각각 전환 수. **페이싱 면제**: 앞뒤 간격이 **모두** ≥334ms(60Hz)/≥360ms(50Hz)인 '고립' 전환만 안전으로 면제 — 뒤 간격은 알 수 없으므로 잠정 면제 후 다음 이벤트가 간격 안에 오면 소급 카운트(`lum_pending_idx`/`red_pending_idx`). 직전 간격만으로 면제하면 혼합 간격 시퀀스(버스트+갭)에서 실제 4플래시/초가 PASS로 빠지는 위음성이 있었음.
5. **Extended Flash**(`extended_series`): HardingFPA 정의 — FAIL 직전 '경고 수준'(1초 전환 ≥3, FAIL 미만) 플래시가 최근 5초의 80% 이상 지속된 정도(0~1). 실제 FAIL 중이거나 진단선이 CAUTION 밴드(전환 2) 아래면 0. 이전엔 FAIL 프레임·1 flash까지 쳐서 Knox 0:15/1:40, Ballie 40~50초에 파란선이 떴지만 QA 1차 PEAT는 전 영상 Extended 없음. 공식 PEAT 그래프의 파란 계단선.
6. **4단계 판정**(공식 PEAT 밴드 모방, `summary["verdict"]`):
   - **FAIL**: 휘도/적색 전환 ≥7(=>3 flash) 또는 패턴 FAIL. 단 `summary["wcag_2_3_1_fail"]`은 휘도/적색만 반영 — 줄무늬 패턴은 Harding/Ofcom 기준이라 WCAG 2.3.1 범주 밖(verdict에만 반영)
   - **CAUTION (FAIL)**: Extended Flash 경고 발생 또는 최대 전환 ≥6 (pass/fail 라인 직전)
   - **CAUTION (PASS)**: 최대 전환 ≥ 실패의 1/3 — 플래시 활동 경고
   - **PASS**: 그 외

### 차트 매핑 (CLI `plot_unified` / GUI 공통)

4단계 밴드는 y축 하단 30%(`band_top = ymax×0.30`)에 얇게 배치. **선 높이는 자기 최댓값 정규화가 아니라 전환수 임계를 밴드 경계에 고정**(`count_to_band_y`): 0→PASS 바닥, `FAIL_TRANSITIONS//3`→1/4, `FAIL_TRANSITIONS-1`→2/4, `FAIL_TRANSITIONS`→3/4, `2×FAIL_TRANSITIONS`→천장. 따라서 선이 놓인 밴드가 그대로 그 시점의 등급이다.

- **Luminance flash / Red flash (점선)** = PEAT '위험도 선' 근사(`count_to_activity_y`). PEAT 설명서: 점선 높이는 "깜빡임의 밝기와 면적"으로 정해지고 횟수는 실선이 센다. 0.75초 창 카운트의 초당 환산(`lum_window_act`/`red_window_act`)으로 **≤4회는 바닥(0)**, 5·6회는 CAUTION 밴드, 7회 미만은 FAIL 밴드 하단 아래. **FAIL(≥7) 높이** h(0=FAIL 경계, 1=천장 0.95×ymax) = (x − `RISK_CONC_OFFSET` 0.51)/`RISK_CONC_SCALE` 0.23 + `RISK_COUNT_GAIN` 0.075 × (픽셀 진단 횟수 − 7), x = `lum_concurrent` = 1초 창 안 **'동시 전환 면적'**(픽셀 추적기가 그 프레임에 함께 전환한 픽셀의 10° 박스 면적, `PixelTransitionTracker.concurrent`) 상위 `RISK_TOPK`=7개 평균. WCAG "동시에 발생하는 플래시의 결합 면적" 그대로 — 물보라·가는 빛줄기·잔무늬 카메라 이동은 픽셀마다 제각각 깜빡여 1초 누적 면적은 크지만 동시 면적이 작아 낮게, 프로젝터 켜짐·컷처럼 한꺼번에 바뀌면 높게. QA 1차 PEAT 스크린샷 20개 피크 픽셀 실측 대조: 평균 오차 0.120(이전 1초 누적 면적 방식 0.158) — Super Speed 7초 0.15(PEAT 0.14), JET FIT 6~7초 0.15(0.15), Product care 0.95(0.95). 남는 차이: Freestyle+ 12·13번 0.94/1.00(PEAT 0.65/0.68), GCE 3:32 0.58(0.97), ViewFinity 1:19 0.40(0.08). FAIL 구간 최소 높이는 `RISK_MIN_H` 0.10 — PEAT는 딱 7회 FAIL(Lint 19초 0.17, Smart Motion 6초 0.06)도 선 위로 보이게 솟음(이전 0.03은 선에 붙어 낮아 보였음). 적색은 (횟수−7)/9. 실시간 그리기도 `on_frame`의 `lum_concurrent`(최근 1초 동시 전환 면적 상위 7개 평균)와 세기 게이트를 받아 최종 차트와 같은 높이로 그린다(이전엔 횟수만으로 그려 분석 중엔 높게 솟았다가 완료 후 낮아짐). ※ PEAT 실제 차트는 합격/불합격 선이 CAUTION(PASS)와 CAUTION(FAIL) 사이(우리는 FAIL 밴드 하단).
- **Lum/Red flash diag (실선)** = 휘도는 **Harding식 픽셀 진단값**(가장 활발한 25%의 픽셀별 1초 전환 수, `lum_window_raw`), 적색은 게이트 적용 전 원시 반전 후보 카운트(`red_window_raw`)의 `count_to_band_y` — 공식 PEAT처럼 낮은 활동 구간까지 길게 이어지는 진단선. 판정에는 영향 없음.
- **Extended Flash (파란선)** = `extended_series` ≥0.8(경고 발생)일 때만 표시.

### GUI 흐름 — `gui.py`

- 입력 3경로: 파일 선택 / 드래그앤드롭 / YouTube URL(`DownloadWorker` — yt-dlp subprocess 우선, 실패 시 라이브러리 직접 호출; `%TEMP%/wcag_flash_temp`에 720p 이하 H.264 우선 비디오만 다운로드 후 자동 분석 시작).
- 분석 중: `frame_result` 시그널마다 데이터 누적 + 매 프레임 차트 갱신. 완료 후 summary의 전체 시계열로 최종 차트를 다시 그리고 줌/팬 활성화 — `setLimits(xMin=0, xMax=영상길이+0.5, yMin=0, yMax=10)`로 영상 길이·y축 범위 밖 줌아웃 차단.
- 상호작용: FAIL 구간 타임스탬프 링크·차트 클릭 → `_show_frame_at_time`으로 해당 프레임 미리보기 + 수직선(`vline`); 좌우 방향키로 1프레임씩 이동(`keyPressEvent`). **위반 영역 하이라이트**(`chk_highlight`, 기본 켜짐): 직전 프레임 대비 휘도 유해 변화 픽셀은 노랑, 적색 전환 픽셀은 청록 반투명 오버레이(`_violation_masks` — 분석 루프는 극값 대비지만 시각화는 프레임간 변화로 근사), 상태바에 위반 픽셀 비율 표시.
- 내보내기: PNG은 pyqtgraph `ImageExporter`, PDF는 matplotlib(Agg)로 차트+판정 정보 2페이지 재생성.
- OpenGL은 꺼져 있음 — pyqtgraph OpenGL이 DashLine 스타일을 무시하는 문제(commit c0a17ad). 코덱은 H.264 우선(AV1 디코더의 OS간 차이 회피).

### 핵심 상수 (`peat.py` 상단)

`DISPLAY_PEAK_CD=200`, `FLASH_DELTA_CD=20`, `DARK_BOUND_CD=160`, `MICHELSON_THRESH=1/17`, `AREA_THRESHOLD=0.25`(로컬 10° 시야각 영역의 25%), `RED_RATIO_THRESH=0.80`, `RED_DELTA_THRESH=20`, `FAIL_TRANSITIONS=7`, `PATTERN_MIN_PAIRS=5`, `EXTENDED_SECONDS=5.0`, `PATTERN_MIN_SECONDS=0.5`.

### 알려진 근사/한계

- **패턴 면적**: `detect_stripe_pattern`은 검출 시 면적을 1.0으로 근사(블록 단위 정밀 면적 산출 미구현). 정지(>40%)/움직임(>25%) 구분은 프레임 차분 기반 근사. 자연 영상에서 위양성 가능 → `--no_pattern`(CLI) / 체크박스(GUI)로 끌 수 있음.
- **cd/m² 환산**: 디스플레이 백색 200 cd/m² 가정에 의존(HDR 미지원). HLG/PQ 영상은 부정확. `DISPLAY_PEAK_CD`를 바꾸면 `FLASH_DELTA_CD`(20/200=10%)·`DARK_BOUND_CD`(160/200=0.8)도 같은 비율로 조정해야 WCAG 등가 유지.
- **Michelson 분기(의도적 보수성)**: WCAG/ITU는 어두운 쪽 ≥160 cd/m²(상대휘도 0.8)면 완전 면제하지만, `harmful_luminance_mask`는 이 영역에서 Michelson ≥1/17을 유해로 유지 — 표준 초과이며 위양성 방향으로만 작용.
- **공식 PEAT/Harding 엔진은 독점**이라 수치 동일성은 검증 불가 — 표준 기준 재현일 뿐.
