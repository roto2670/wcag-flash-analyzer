# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WCAG 2.3.1 비디오 플래시 분석 프로토타입 — PEAT(Photosensitive Epilepsy Analysis Tool) 유사 도구. 비디오에서 휘도(luminance) 및 적색(red) 플래시를 프레임 단위로 분석하여 광과민성 발작 위험을 평가한다.

두 가지 엔진이 있다:

- **`main.py`** — 초기 프로토타입. 상대휘도 0.1 임계값 기반, 평균 휘도 피크/밸리 추적.
- **`peat.py`** — Harding/ITU-R BT.1702-3 충실 재구현(권장). cd/m² 절대 휘도 기준, 적색 UCS 색차, 줄무늬 패턴 분석, 페이싱 면제까지 포함.

> 공식 PEAT의 분석 코어는 Cambridge Research Systems의 Harding FPA 엔진을 라이선스한 **독점 모듈**이라 코드 복제가 불가능하다. `peat.py`는 PEAT가 따르는 **표준(WCAG 2.3.1 + ITU-R BT.1702-3 + Harding/Wilkins)**을 직접 구현한 것이다.

## Environment Setup

```bash
source venv/bin/activate   # Python 3.8 가상환경 활성화
```

의존성: `opencv-python`, `numpy`, `matplotlib` (venv에 설치됨)

## Running the Script

```bash
python peat.py --video ./temp/<영상파일>            # 권장: Harding/ITU 엔진
python peat.py --video ./temp/<영상파일> --no_show  # 그래프 없이 콘솔 요약만
python main.py --video ./temp/<영상파일>            # 구 프로토타입
```

테스트 비디오는 `temp/` 디렉터리에 있음 (gitignore됨). `temp/test_10.avi`는 5.7GB로 크니 `--max_frames`/`--downsample`로 제한해서 돌릴 것.

공통 선택 인자: `--area_threshold` (기본 0.25), `--window` (기본 1.0초), `--downsample N`, `--max_frames N`.
`peat.py` 전용: `--no_pattern` (줄무늬 패턴 분석 끄기), `--no_show`.
`main.py` 전용: `--lum_delta_threshold` (기본 0.10).

`peat.py`는 항상 콘솔에 휘도/적색/패턴별 PASS·FAIL 요약을 출력한다(`print_summary`). `main.py`는 콘솔 출력 없이 Matplotlib 창만 띄운다.

## Architecture — `peat.py` (Harding/ITU 엔진, 권장)

공통 파이프라인: `cv2.VideoCapture` 프레임 읽기 → sRGB 선형화(breakpoint **0.04045**) → 절대 휘도 `cd/m² = Y_rel × DISPLAY_PEAK_CD`(기본 200, SDR 가정).

1. **휘도 플래시**: 평균 휘도의 피크/밸리(극값) 추적. 극값 프레임 대비 **유해 전환 픽셀**의 **로컬 10° 시야각 면적**(`local_area_fraction`: 화면 1/3×1/3 박스의 최대 깜빡임 밀도)이 ≥25%이고 방향이 직전과 반대(opposing)면 전환 1회로 카운트. ※ 전체 화면 25%가 아니라 로컬 영역 25% — 공식 PEAT/WCAG의 "10° 시야각" 기준이라 국소 플래시를 포착(전체평균 방식보다 ~9배 민감).
   - 유해 전환 = `harmful_luminance_mask`: 어두운쪽 <160 cd/m²면 차이 ≥**20 cd/m²**, ≥160 cd/m²면 **Michelson 대비 ≥1/17**.
2. **적색 플래시**: `R/(R+G+B) ≥ 0.8`(포화 적색) **AND** 두 상태의 **CIE 1976 UCS (u',v') 거리 > 0.2**인 픽셀이 로컬 면적(동일 `local_area_fraction`) ≥25%일 때 전환 카운트. (휘도와 독립된 방향 추적)
3. **줄무늬 패턴**: `detect_stripe_pattern` — 행/열 평균 프로파일 FFT로 지배 주파수 검출. **>5 광-암 쌍** + 광-암차 ≥20 cd/m² + 최고휘도 >50 cd/m²이면 패턴 후보, **≥0.5초 지속** 시 FAIL.
4. **전환 카운트**: 1초 슬라이딩 윈도우 내 휘도/적색 각각 전환 수. **페이싱 면제**: leading edge 간격 ≥334ms(60Hz)/≥360ms(50Hz) 전환은 안전으로 보고 누적 제외.
5. **Extended Flash**(`extended_series`): 5초 윈도우의 80% 프레임에서 실패의 1/3(=전환 `FAIL_TRANSITIONS//3`) 이상 플래시가 지속되는 정도(0~1). 공식 PEAT 그래프의 파란 계단선.
6. **4단계 판정**(공식 PEAT 밴드 모방, `summary["verdict"]`):
   - **FAIL**: 휘도/적색 전환 ≥7(=>3 flash) 또는 패턴 FAIL — 명백한 WCAG 위반
   - **CAUTION (FAIL)**: Extended Flash 경고 발생 또는 최대 전환 ≥6 (pass/fail 라인 직전)
   - **CAUTION (PASS)**: 최대 전환 ≥ 실패의 1/3 — 플래시 활동 경고
   - **PASS**: 그 외
7. **출력**: `print_summary`(4단계 등급) + `plot_unified`(공식 PEAT ex.png 모방). 레이아웃: 회색 배경, **4밴드는 화면 하단 30%에 얇게**. 공식 PEAT 범례대로 시리즈 매핑:
   - **Luminance flash / Red flash (점선) = 플래시 활동(메인, 솟음)**. 휘도 활동은 `lum_window`(우리 데이터상 활동≈윈도우 카운트) 정규화로 천장까지; 적색 활동은 `sat_area`(포화적색 존재 면적, 전환 무관)의 1초 윈도우 max — 지속 적색도 표시되도록.
   - **Lum/Red flash diag (실선) = 카운트(보조, 흐리게)**.
   - **Extended Flash (파란선) = `extended_series`가 ≥0.8(경고 발생)일 때만 표시** — 미발생 영상은 안 보임.
   ※ ex.png 모양 재현을 위해 메인 선은 정규화 표시이며, 절대 카운트는 `summary["flash_max_per_window"]` 참조.

### `peat.py` 핵심 상수 (파일 상단)

`DISPLAY_PEAK_CD=200`, `FLASH_DELTA_CD=20`, `DARK_BOUND_CD=160`, `MICHELSON_THRESH=1/17`, `AREA_THRESHOLD=0.25`(로컬 10° 시야각 영역의 25%), `RED_RATIO_THRESH=0.80`, `RED_UV_DIST_THRESH=0.20`, `FAIL_TRANSITIONS=7`, `PATTERN_MIN_PAIRS=5`.

### 알려진 근사/한계

- **패턴 면적**: `detect_stripe_pattern`은 검출 시 면적을 1.0으로 근사(블록 단위 정밀 면적 산출 미구현). 정지(>40%)/움직임(>25%) 구분은 프레임 차분 기반 근사. 자연 영상에서 위양성 가능 → `--no_pattern`으로 끌 수 있음.
- **cd/m² 환산**: 디스플레이 백색 200 cd/m² 가정에 의존(HDR 미지원). HLG/PQ 영상은 부정확.
- **공식 PEAT/Harding 엔진은 독점**이라 수치 동일성은 검증 불가 — 표준 기준 재현일 뿐.

## Architecture — `main.py` (구 프로토타입)

1. **휘도 계산**: sRGB 선형화(`linearize_srgb`) 후 `Y = 0.2126R + 0.7152G + 0.0722B`. `red_mask`는 선형화 전 원본 sRGB로 판정.
2. **플래시 감지**: 평균 휘도 피크/밸리 추적, 극값 대비 누적 변화 `|ΔY| ≥ lum_delta_threshold(0.1)` AND 면적 ≥25%. 적색은 `R/(R+G+B)≥0.8` 상태변화 비율 ≥25%(UCS 색차 없음).
3. **판정**: 1초 윈도우 방향 전환 ≥7회면 FAIL. 출력은 Matplotlib 그래프(`"FAKE PEAT"`)뿐.

`peat.py` 대비 차이: cd/m² 절대 기준·적색 UCS 색차·패턴 분석·페이싱 면제가 모두 없음.
