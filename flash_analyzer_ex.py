"""
flash_analyzer.py — 영상 깜빡임(광과민성 위험) 분석 엔진

주석의 "§n"은 계산식 문서의 절 번호입니다.

구현 범위
  - mode 1 (25fps), mode 2 (30fps) 진행형 영상: 전체 판정 경로 구현
  - mode 3, 4 (인터레이스 필드 모드): 미구현 (NotImplementedError)
  - 패턴 판정: 비활성, 항상 0

참고
  - 실수 연산은 64비트 double을 씁니다.
  - ROI는 화면 전체로 가정합니다.

사용 예 1) 명령줄에서 영상 분석 (pip install numpy opencv-python)
    python flash_analyzer.py input.mp4 result.csv            # fps를 보고 mode 자동 선택
    python flash_analyzer.py input.mp4 result.csv --mode 1   # 25fps 규칙 강제

사용 예 2) 다른 코드에서 import
    from flash_analyzer import FlashAnalyzer
    fa = FlashAnalyzer(mode=2)                # 30fps
    for frame_rgb in frames:                  # (H, W, 3) uint8, RGB 순서
        r = fa.process_frame(frame_rgb)
        print(r.verdict, r.lum_score, r.red_score)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 정수 연산 도우미 (C식 정수 나눗셈)
# ---------------------------------------------------------------------------

def tdiv(a, b: int):
    """0 쪽으로 버리는 정수 나눗셈. b > 0 가정."""
    a = np.asarray(a, dtype=np.int64)
    return np.sign(a) * (np.abs(a) // b)


def rint_even(x):
    """짝수 쪽 반올림. 4.5 -> 4, 5.5 -> 6."""
    return np.rint(x).astype(np.int64)


# ---------------------------------------------------------------------------
# 설정값 (기본값)
# ---------------------------------------------------------------------------

@dataclass
class FlashConfig:
    flash_lum_delta: int = 20    # 9프레임 변동폭 문턱, cd/m²
    flash_area_pct: int = 25     # 면적 기준, %
    transitions_sec: int = 6     # 1초 창 전환 수 문턱 (> 6)
    reserved: int = 20           # 사용하지 않음
    pattern_area_pct: int = 40   # 패턴(비활성)
    pattern_pairs: int = 5       # 패턴(비활성)
    red_saturation_pct: int = 80 # 포화 적색 비율 문턱, %
    scale: int = 10              # 분석 해상도 배율 (10 -> 96x72)


# ---------------------------------------------------------------------------
# §1 휘도 변환 표
# ---------------------------------------------------------------------------

def build_lut() -> np.ndarray:
    """픽셀값 0..255 -> 선형 휘도 0..200 cd/m² (정수)."""
    lut = np.zeros(256, dtype=np.int64)
    for p in range(256):
        if p <= 10:
            v = (p / 255.0 * 200.0) / 12.92
        else:
            v = ((p / 255.0 + 0.055) / 1.055) ** 2.4 * 200.0
        lut[p] = int(np.rint(v))
    return lut


def build_inv_lut() -> np.ndarray:
    """선형 휘도 0..200 -> 픽셀값 0..255. 0만 선형 구간으로 처리합니다."""
    inv = np.zeros(201, dtype=np.int64)
    for v in range(201):
        if v <= 0:
            x = (v / 200.0) * 12.92 * 255.0
        else:
            x = ((v / 200.0) ** (1.0 / 2.4) * 1.055 - 0.055) * 255.0
        inv[v] = int(np.rint(x))
    return inv


LUT = build_lut()
INV_LUT = build_inv_lut()

# Rec.709 가중치를 65536 단위로 나타낸 값 (합 = 65536)
W_R, W_G, W_B = 13933, 46871, 4732


# ---------------------------------------------------------------------------
# §1 축소: 선형 휘도 공간 가우시안 리샘플링
# ---------------------------------------------------------------------------

def _gauss_weights(k: int, a: float, s: int) -> np.ndarray:
    c = k // 2
    i = np.arange(k)
    return np.trunc(np.exp((i - c) ** 2 * a) * s).astype(np.int64)


def downscale(bgr: np.ndarray, wd: int, hd: int) -> np.ndarray:
    """(Hs, Ws, 3) BGR uint8 -> (hd, wd, 3) BGR uint8.

    각 채널을 LUT로 선형화해 가우시안 가중 합을 낸 뒤 INV_LUT로 되돌립니다.
    커널은 입력 픽셀 위치 r*Hs//hd 에서 시작해 아래/오른쪽으로 K칸을 덮습니다(중심 정렬 아님).
    """
    hs, ws = bgr.shape[:2]
    if ws == wd and hs == hd:
        return bgr.copy()                       # 같은 크기면 그대로 복사

    kx = int(2 * ws / wd) + 1
    ky = int(2 * hs / hd) + 1
    ax = -1.0 / ((2 * kx) / 4.0)
    ay = -1.0 / ((2 * ky) / 4.0)

    # 가중치 합이 2^23 미만이 될 때까지 배율 s를 절반씩 줄임
    s = 2000
    while True:
        s >>= 1
        wx = _gauss_weights(kx, ax, s)
        wy = _gauss_weights(ky, ay, s)
        total = int(wx.sum() * wy.sum())
        if total < 0x800000:
            break

    lin = LUT[bgr]                              # (hs, ws, 3) 선형값
    # 각 행 오른쪽에 마지막 픽셀을 kx번 덧붙임
    pad = np.repeat(lin[:, -1:, :], kx, axis=1)
    lin_ext = np.concatenate([lin, pad], axis=1)   # (hs, ws+kx, 3)

    out = np.zeros((hd, wd, 3), dtype=np.uint8)
    col_start = (np.arange(wd) * ws) // wd
    for r in range(hd):
        # 세로 방향: 행 r*Hs//hd 부터 ky개 행을 가중 합
        acc = np.zeros((ws + kx, 3), dtype=np.int64)
        last_row = None
        for j in range(ky):
            srow = (r * hs) // hd + j
            if srow < hs:
                last_row = lin_ext[srow]
            # 범위를 벗어나면 직전에 읽은 행을 다시 씀
            acc += last_row * wy[j]
        # 가로 방향: 열 q*Ws//wd 부터 kx개 열을 가중 합
        win = np.lib.stride_tricks.sliding_window_view(acc, kx, axis=0)  # (ws+1, 3, kx)
        ssum = (win[col_start] * wx).sum(axis=2)                         # (wd, 3)
        out[r] = INV_LUT[ssum // total]
    return out


# ---------------------------------------------------------------------------
# 프레임 하나의 휘도·적색 데이터
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    LB: np.ndarray   # 선형 파랑 (H, W) int
    LG: np.ndarray   # 선형 초록
    LR: np.ndarray   # 선형 빨강
    Y: np.ndarray    # §1 픽셀 휘도

    @staticmethod
    def from_bgr(small_bgr: np.ndarray) -> "FrameData":
        lin = LUT[small_bgr]
        LB, LG, LR = lin[..., 0], lin[..., 1], lin[..., 2]
        Y = (W_R * LR + W_G * LG + W_B * LB) >> 16
        return FrameData(LB, LG, LR, Y)

    @property
    def red_diff(self):
        return self.LR - self.LB - self.LG

    def rex(self):
        """§2 적색 지표 (블록 히스토그램용)."""
        return np.maximum(0, rint_even(self.red_diff * 1.6))

    def red_q(self):
        """적색 전환 판정용 정수 신호 (410/768 ≈ 1.6/3)."""
        return np.maximum(0, tdiv(410 * self.red_diff + 128, 768))

    def red_energy_signal(self):
        """적색 에너지 계산용 9프레임 신호 r_k."""
        return np.maximum(0, 3 * rint_even(self.red_diff * 1.6 / 3.0))

    def red_ratio(self):
        """포화 적색 비율 σ = floor(100·R / (R+G+B+1))."""
        return (100 * self.LR) // (self.LR + self.LG + self.LB + 1)


# ---------------------------------------------------------------------------
# §3 움직임과 깜빡임 구분: 블록 히스토그램 게이트
# ---------------------------------------------------------------------------

def block_gates(cur: FrameData, prev: FrameData):
    """블록별 실제 밝기 변화 여부를 픽셀 해상도 맵으로 돌려줍니다 (휘도, 적색)."""
    h, w = cur.Y.shape
    b = w >> 3
    nby, nbx = h // b, w // b
    thr_lum = int(np.rint(36 * b * b / 100.0))
    thr_red = int(np.rint(44 * b * b / 100.0))

    # 직전 프레임 맵은 x = 0 열을 0으로 둡니다
    prevY = prev.Y.copy();     prevY[:, 0] = 0
    prevR = prev.rex().copy(); prevR[:, 0] = 0
    curY, curR = cur.Y, cur.rex()

    g_lum = np.zeros((h, w), dtype=bool)
    g_red = np.zeros((h, w), dtype=bool)
    for by in range(nby):
        for bx in range(nbx):
            sl = (slice(by * b, by * b + b), slice(bx * b, bx * b + b))
            hp = np.bincount((prevY[sl] // 10).ravel(), minlength=61)[:61]
            hc = np.bincount((curY[sl] // 10).ravel(), minlength=61)[:61]
            if np.abs(hc - hp).sum() > thr_lum:
                g_lum[sl] = True
            rp = np.bincount((prevR[sl] // 10).ravel(), minlength=61)[:61]
            rc = np.bincount((curR[sl] // 10).ravel(), minlength=61)[:61]
            if np.abs(rc - rp).sum() > thr_red:
                g_red[sl] = True
    return g_lum, g_red


# ---------------------------------------------------------------------------
# §4 전환과 1초 창: 픽셀별 상태 링 버퍼
# ---------------------------------------------------------------------------

class TransitionRing:
    """픽셀마다 (flag, expect, counter)를 N프레임 링 버퍼로 보관합니다."""

    def __init__(self, n: int, shape):
        self.n = n
        self.flag = np.zeros((n,) + shape, dtype=np.int8)     # 이번 프레임 전환 (+1/-1/0)
        self.expect = np.zeros((n,) + shape, dtype=np.int8)   # 다음 기대 방향
        self.counter = np.zeros((n,) + shape, dtype=np.int64) # 최근 1초 전환 수

    def update(self, f: int, d: np.ndarray, gate: np.ndarray, thr: int = 6) -> np.ndarray:
        cur, prv = f % self.n, (f - 1) % self.n
        old_flag = self.flag[cur].copy()          # N프레임 전(1초 전) 기록
        p_exp, p_cnt = self.expect[prv], self.counter[prv]

        rise = (d > thr) & gate & (p_exp >= 0)
        fall = ~rise & (d < -thr) & gate & (p_exp <= 0)
        trans = rise | fall

        cnt = p_cnt.copy()
        # 전환 발생: 1초 전 칸이 비어 있었으면 +1, 아니면 하나 들어오고 하나 나가 그대로
        cnt = np.where(trans & (old_flag == 0), np.minimum(p_cnt + 1, 255), cnt)
        # 전환 없음: 1초 전 전환이 빠져나가면 -1
        leave = ~trans & (old_flag != 0) & (p_cnt > 0)
        cnt = np.where(leave, p_cnt - 1, cnt)

        exp = p_exp.copy()
        exp = np.where(rise, -1, exp)
        exp = np.where(fall, +1, exp)
        exp = np.where(leave & (cnt == 0), 0, exp)          # 창이 비면 기대 방향 초기화

        fl = np.zeros_like(old_flag)
        fl = np.where(rise, 1, fl)
        fl = np.where(fall, -1, fl)

        self.flag[cur], self.expect[cur], self.counter[cur] = fl, exp, cnt
        return cnt


# ---------------------------------------------------------------------------
# §6 깜빡임 에너지 E: 8점 월시 변환
# ---------------------------------------------------------------------------

TAPER = np.array([2, 3, 4, 5, 6, 7, 6, 5], dtype=np.int64)

# 월시 계수 패턴 (a1 .. a8)
WALSH = {
    1:  [+1, +1, +1, +1, -1, -1, -1, -1],
    2:  [+1, +1, -1, -1, -1, -1, +1, +1],
    3:  [+1, +1, -1, -1, +1, +1, -1, -1],
    4:  [+1, -1, -1, +1, +1, -1, -1, +1],
    5:  [+1, -1, -1, +1, -1, +1, +1, -1],
    6:  [+1, -1, +1, -1, -1, +1, -1, +1],
    7:  [+1, -1, +1, -1, +1, -1, +1, -1],
    8:  [+1, +1, +1, -1, -1, -1, -1, +1],
    9:  [+1, -1, -1, -1, -1, +1, +1, +1],
    10: [-1, -1, +1, -1, +1, +1, -1, +1],
    11: [-1, +1, -1, -1, +1, -1, +1, +1],
}
WALSH_M = {j: np.array(v, dtype=np.int64) for j, v in WALSH.items()}


def energy(hist: np.ndarray, delta_thr: int) -> np.ndarray:
    """hist: (9, P) 정수, 오래된 프레임부터. 반환: 픽셀별 E (P,)."""
    m = hist.sum(axis=0) // 9                       # §6.1 평균 (값이 음수가 아니므로 floor = trunc)
    d = hist - m
    delta = d.max(axis=0) - d.min(axis=0)
    ok = delta >= delta_thr

    a = tdiv(TAPER[:, None] * d[1:9] + 4, 8)        # §6.2 시간 창, d2..d9 사용
    W = {j: (v[:, None] * a).sum(axis=0) for j, v in WALSH_M.items()}  # §6.3

    B1 = tdiv(np.abs(W[1]) + np.abs(W[2]) + np.abs(W[8]) + np.abs(W[9]) + 4, 8)   # §6.4
    B2 = (np.abs(W[3]) + np.abs(W[4] + 2)) // 4
    B3 = tdiv(np.abs(W[5]) + np.abs(W[6]) + np.abs(W[10]) + np.abs(W[11]) + 4, 8)
    B4 = np.abs(tdiv(W[7], 2))

    E = np.zeros_like(m)                            # §6.5
    for b, B in enumerate((B1, B2, B3, B4), start=1):
        E += np.where(B >= 1, b * B + 50 * b, 0)
    return np.where(ok, E, 0)


# ---------------------------------------------------------------------------
# §7 위험도 곡선
# ---------------------------------------------------------------------------

def f_small(x: float) -> float:
    xs = min(x, 8.3)
    return (23.0 * xs - xs * xs - 60.0) / 14.0


def f_large(x: float) -> float:
    xl = min(x, 23.0)
    return (46.0 * xl - xl * xl - 120.0) / 12.0


def risk_curve(x: float, area: float, a25: float) -> float:
    """area < a25 작은 곡선, area >= 2·a25 큰 곡선, 그 사이 선형 보간."""
    if area < a25:
        f = f_small(x)
    elif area < 2 * a25:
        t = (area - a25) / a25
        f = t * f_large(x) + (1.0 - t) * f_small(x)
    else:
        f = f_large(x)
    return max(f, 0.0)


def classify(R: float) -> int:
    """§8 0 = PASS, 1 = 경고, 2 = FAIL."""
    if R < 0.05:
        return 0
    if R <= 0.05 + 0.4:
        return 1
    return 2


# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------

VERDICT = {0: "PASS", 1: "WARN", 2: "FAIL"}


@dataclass
class FrameResult:
    frame: int
    lum_score: float        # R (휘도), 0.1 단위
    red_score: float        # R (적색)
    lum_class: int
    red_class: int
    extended: bool          # §9 확장 경고
    lum_x: float            # §6.6 깜빡임 강도
    red_x: float
    lum_area_pct: float     # §5 위험 면적 비율
    red_area_pct: float
    lum_f: float            # §7 위험도 (반올림 전)
    red_f: float
    lum_k_count: int        # §5 F(k)가 1인 k의 개수
    red_k_count: int

    @property
    def verdict(self) -> str:
        return VERDICT[max(self.lum_class, self.red_class)]


# ---------------------------------------------------------------------------
# 분석기 본체
# ---------------------------------------------------------------------------

class FlashAnalyzer:
    WINDOW_W, WINDOW_H = 96, 72        # 판정 창 크기 (고정)

    def __init__(self, mode: int = 1, config: FlashConfig | None = None):
        if mode in (3, 4):
            raise NotImplementedError("인터레이스 필드 모드(3, 4)는 구현하지 않았습니다.")
        if mode not in (1, 2):
            raise ValueError("mode는 1(25fps) 또는 2(30fps)")
        self.cfg = config or FlashConfig()
        self.fps30 = (mode == 2)
        self.N = 30 if self.fps30 else 25                         # §4 1초 창
        self.M, self.M4 = (150, 120) if self.fps30 else (125, 100)  # §9 5초 창

        s = self.cfg.scale
        self.W = (s * 192 // 10) // 2
        self.H = (s * 144 // 10) // 2
        self.roi_area = self.W * self.H
        self.kA = 25344.0 / self.roi_area

        self.frames: list[FrameData] = []
        self.f = 0                                                # 처리한 프레임 수
        shape = (self.H, self.W)
        self.ring_lum = TransitionRing(self.N, shape)
        self.ring_red = TransitionRing(self.N, shape)
        self.ext_flag = np.zeros(self.M, dtype=np.int8)
        self.ext_cnt = np.zeros(self.M, dtype=np.int64)
        # 직전 프레임 휘도 버퍼: 분석 단계에서만 갱신되므로 첫 분석 프레임에서는 0입니다
        self.prevY = np.zeros(shape, dtype=np.int64)

        # 단계 사이에 넘겨지는 값
        self.S_lum = 0.0
        self.S_red = 0.0
        self.F_lum = np.zeros(31, dtype=bool)
        self.F_red = np.zeros(31, dtype=bool)

    # -- 입력 -------------------------------------------------------------
    def process_frame(self, rgb: np.ndarray) -> FrameResult:
        """rgb: (H, W, 3) uint8, RGB 순서."""
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        return self.process_frame_bgr(bgr)

    def process_frame_bgr(self, bgr: np.ndarray) -> FrameResult:
        small = downscale(bgr, self.W, self.H)
        self.frames.append(FrameData.from_bgr(small))
        if len(self.frames) > 9:
            self.frames.pop(0)

        a_lum = a_red = 0.0                       # 매 프레임 0으로 초기화
        if len(self.frames) == 9:                 # 9프레임이 모여야 분석 실행
            a_lum, a_red = self._analyze()

        res = self._score(a_lum, a_red)
        self.f += 1
        return res

    # -- 깜빡임 분석 (§3 ~ §6) ------------------------------------------
    def _analyze(self):
        cfg, fr = self.cfg, self.frames
        cur, prev = fr[-1], fr[-2]
        g_lum, g_red = block_gates(cur, prev)

        # 휘도 전환 (§4): 직전 휘도는 버퍼에서 읽음 (첫 분석 프레임에서는 0)
        d = cur.Y - self.prevY
        self.prevY = cur.Y.copy()
        c_lum = self.ring_lum.update(self.f, d, g_lum)

        # 적색 전환: 현재 또는 직전 프레임이 포화 적색이어야 함
        sat_now = (cur.red_ratio() > cfg.red_saturation_pct) | (prev.red_ratio() > cfg.red_saturation_pct)
        d_red = 3 * (cur.red_q() - prev.red_q())
        c_red = self.ring_red.update(self.f, d_red, g_red & sat_now)

        # 위험 픽셀 (§5)
        Ys = np.stack([fr_.Y for fr_ in fr])                   # (9, H, W)
        D_lum = (c_lum > cfg.transitions_sec) & (Ys.min(axis=0) < 160)
        D_red = (c_red > cfg.transitions_sec)                  # 적색은 밝기 조건 없음

        # 에너지 (§6)
        P = self.H * self.W
        E_lum = np.zeros(P, dtype=np.int64)
        idx = np.flatnonzero(D_lum)
        if idx.size:
            E_lum[idx] = energy(Ys.reshape(9, P)[:, idx], cfg.flash_lum_delta)

        E_red = np.zeros(P, dtype=np.int64)
        idx = np.flatnonzero(D_red)
        if idx.size:
            rs = np.stack([fr_.red_energy_signal() for fr_ in fr]).reshape(9, P)[:, idx]
            sat9 = np.stack([fr_.red_ratio() for fr_ in fr]).reshape(9, P)[:, idx].max(axis=0) > cfg.red_saturation_pct
            E_red[idx] = np.where(sat9, energy(rs, cfg.flash_lum_delta), 0)

        E_lum = E_lum.reshape(self.H, self.W)
        E_red = E_red.reshape(self.H, self.W)

        # 판정 창 선택과 보조 정보 F(k) (§5, §6.6)
        self.S_lum, self.F_lum, n_lum = self._window(D_lum, E_lum, c_lum, 1.5 * 64 / 9)
        self.S_red, self.F_red, n_red = self._window(D_red, E_red, c_red, 64 / 9)

        # 위험 면적 비율: 선택된 창의 위험 픽셀 수 / 창 면적(96x72)
        win_area = self.WINDOW_W * self.WINDOW_H
        a_lum = 100.0 * n_lum / win_area
        a_red = 100.0 * n_red / win_area
        return a_lum, a_red

    def _window(self, D, E, cnt, coef):
        """96x72 창을 옮기며 위험도가 가장 큰 창을 고르고, 그 창 안의 F(k)를 구합니다."""
        ww, wh = min(self.WINDOW_W, self.W), min(self.WINDOW_H, self.H)
        A = int(np.rint(self.WINDOW_W * self.WINDOW_H * self.cfg.flash_area_pct / 100.0))
        iD = np.pad(D.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        iE = np.pad(E.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        best_f, best_n, best_S, by0, bx0 = 0.0, 0, 0.0, 0, 0
        for y0 in range(self.H - wh + 1):
            for x0 in range(self.W - ww + 1):
                y1, x1 = y0 + wh, x0 + ww
                n = int(iD[y1, x1] - iD[y0, x1] - iD[y1, x0] + iD[y0, x0])
                S = float(iE[y1, x1] - iE[y0, x1] - iE[y1, x0] + iE[y0, x0])
                # 지금까지의 최선보다 위험 픽셀도 에너지도 많지 않으면 건너뜀
                if not (n > best_n or S > best_S):
                    continue
                x = S * coef * self.kA / 1e6
                f = risk_curve(x, n, A)
                if f > best_f or (f == best_f and n > best_n):
                    best_f, best_n, best_S, by0, bx0 = f, n, S, y0, x0
        sub = cnt[by0:by0 + wh, bx0:bx0 + ww]
        F = np.zeros(31, dtype=bool)
        for k in range(1, 31):
            F[k] = (sub >= k).sum() > A
        return best_S, F, best_n

    # -- 점수와 판정 (§7 ~ §9) ------------------------------------------
    def _score(self, a_lum, a_red) -> FrameResult:
        x_lum = self.S_lum * 1.5 * 64 / 9 * self.kA / 1e6
        x_red = self.S_red * 64 / 9 * self.kA / 1e6

        f_lum = risk_curve(x_lum, a_lum, float(self.cfg.flash_area_pct))
        f_red = risk_curve(x_red, a_red, 25.0)            # 적색은 고정 실수 25.0
        R_lum = int(rint_even(f_lum)) / 10.0
        R_red = int(rint_even(f_red)) / 10.0
        cL, cR = classify(R_lum), classify(R_red)

        # §9 확장 경고: 최근 5초 링 버퍼
        w = ((cL == 1 and cR < 2) or (cL < 2 and cR == 1))
        cur, prv = self.f % self.M, (self.f - 1) % self.M
        old = self.ext_flag[cur]
        pc = self.ext_cnt[prv]
        if w:
            cnt = min(pc + 1, 255) if old == 0 else pc
        else:
            cnt = pc - 1 if (old != 0 and pc > 0) else pc
        self.ext_flag[cur], self.ext_cnt[cur] = int(w), cnt
        extended = bool(cnt > self.M4 and w)

        return FrameResult(
            frame=self.f,
            lum_score=R_lum, red_score=R_red,
            lum_class=cL, red_class=cR,
            extended=extended,
            lum_x=x_lum, red_x=x_red,
            lum_area_pct=a_lum, red_area_pct=a_red,
            lum_f=f_lum, red_f=f_red,
            lum_k_count=int(self.F_lum.sum()), red_k_count=int(self.F_red.sum()),
        )


# ---------------------------------------------------------------------------
# 명령줄 실행: 영상 파일 -> 프레임별 결과 CSV
#   프레임을 받은 그대로 하나씩 넣습니다(프레임레이트 변환 없음).
# ---------------------------------------------------------------------------

def main():
    import argparse
    import csv

    import cv2                                   # 영상 읽기에만 필요하므로 여기서 불러옴

    ap = argparse.ArgumentParser(description="영상 깜빡임 분석")
    ap.add_argument("video")
    ap.add_argument("csv_out")
    ap.add_argument("--mode", type=int, choices=(1, 2), default=None,
                    help="1 = 25fps 규칙, 2 = 30fps 규칙 (기본: 영상 fps로 결정)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mode = args.mode or (2 if fps >= 27.5 else 1)
    fa = FlashAnalyzer(mode=mode)
    print(f"fps={fps:.2f} -> mode {mode}")

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    n_ext = 0
    with open(args.csv_out, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["frame", "time_s", "verdict", "lum_R", "red_R", "extended",
                    "lum_x", "red_x", "lum_area_pct", "red_area_pct", "lum_f", "red_f"])
        while True:
            ok, bgr = cap.read()                 # OpenCV는 BGR 순서로 줍니다
            if not ok:
                break
            r = fa.process_frame_bgr(bgr)
            counts[r.verdict] += 1
            n_ext += r.extended
            w.writerow([r.frame, f"{r.frame / fps:.3f}", r.verdict, r.lum_score, r.red_score,
                        int(r.extended), f"{r.lum_x:.3f}", f"{r.red_x:.3f}",
                        f"{r.lum_area_pct:.2f}", f"{r.red_area_pct:.2f}",
                        f"{r.lum_f:.3f}", f"{r.red_f:.3f}"])
    cap.release()
    print(counts, "extended:", n_ext)


if __name__ == "__main__":
    main()
