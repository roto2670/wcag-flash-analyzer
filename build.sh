#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# build.sh — Nuitka를 사용한 PEAT Analyzer exe 빌드
# 
# 사용법:
#   chmod +x build.sh
#   ./build.sh
#
# 요구사항:
#   - Python 3.8+ (venv 활성화된 상태)
#   - pip install nuitka ordered-set
#   - C 컴파일러 (macOS: Xcode CLT, Windows: MSVC/MinGW, Linux: gcc)
#
# Windows에서 exe 빌드 시:
#   python -m nuitka --standalone --onefile --enable-plugin=pyqt5 \
#       --include-module=cv2 --include-module=numpy \
#       --include-module=pyqtgraph --include-module=peat \
#       --include-module=worker \
#       --noinclude-custom-mode=yt_dlp:bytecode \
#       --windows-console-mode=disable \
#       --output-filename=PEAT_Analyzer.exe \
#       gui.py
#
# ※ yt_dlp는 반드시 bytecode로 포함할 것 (--noinclude-custom-mode=yt_dlp:bytecode).
#   C 컴파일하면 사이트별 추출기 ~1800개 모듈이 전부 컴파일 대상이 되어 빌드가
#   수 분씩 걸리고, 초대형 생성 파일인 lazy_extractors에서 컴파일러 OOM으로
#   빌드가 실패한다. exe 안에서는 subprocess 경로(sys.executable -m yt_dlp)가
#   동작하지 않아 항상 라이브러리 폴백을 타므로, 완전히 빼면(-nofollow-import-to)
#   YouTube 다운로드 기능이 죽는다.
# ──────────────────────────────────────────────────────────────────

set -e

echo "=== PEAT Analyzer — Nuitka 빌드 시작 ==="

# venv 활성화 확인
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[!] venv가 활성화되지 않았습니다. 활성화합니다..."
    source venv/bin/activate
fi

# nuitka 설치 확인
if ! python -m nuitka --version > /dev/null 2>&1; then
    echo "[*] Nuitka 설치 중..."
    pip install nuitka ordered-set
fi

echo "[*] 빌드 옵션:"
echo "    - standalone (Python 미설치 환경에서 실행 가능)"
echo "    - onefile (단일 실행 파일)"
echo "    - PyQt5 플러그인 활성화"
echo ""

# 플랫폼별 분기
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS — .app 번들 또는 단일 바이너리
    echo "[*] macOS 빌드..."
    python -m nuitka \
        --standalone \
        --onefile \
        --enable-plugin=pyqt5 \
        --include-module=cv2 \
        --include-module=numpy \
        --include-module=pyqtgraph \
        --include-module=peat \
        --include-module=worker \
        --noinclude-custom-mode=yt_dlp:bytecode \
        --macos-create-app-bundle \
        --macos-app-name="PEAT Analyzer" \
        --output-filename=PEAT_Analyzer \
        gui.py

    echo ""
    echo "=== 빌드 완료 ==="
    echo "결과: ./PEAT_Analyzer.app (또는 ./PEAT_Analyzer.bin)"

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    echo "[*] Windows 빌드..."
    python -m nuitka \
        --standalone \
        --onefile \
        --enable-plugin=pyqt5 \
        --include-module=cv2 \
        --include-module=numpy \
        --include-module=pyqtgraph \
        --include-module=peat \
        --include-module=worker \
        --noinclude-custom-mode=yt_dlp:bytecode \
        --windows-console-mode=disable \
        --output-filename=PEAT_Analyzer.exe \
        gui.py

    echo ""
    echo "=== 빌드 완료 ==="
    echo "결과: ./PEAT_Analyzer.exe"

else
    # Linux
    echo "[*] Linux 빌드..."
    python -m nuitka \
        --standalone \
        --onefile \
        --enable-plugin=pyqt5 \
        --include-module=cv2 \
        --include-module=numpy \
        --include-module=pyqtgraph \
        --include-module=peat \
        --include-module=worker \
        --noinclude-custom-mode=yt_dlp:bytecode \
        --output-filename=PEAT_Analyzer \
        gui.py

    echo ""
    echo "=== 빌드 완료 ==="
    echo "결과: ./PEAT_Analyzer"
fi

echo ""
echo "[i] 빌드된 파일은 대상 OS에서 Python 없이 실행 가능합니다."
echo "[i] Windows exe 빌드는 Windows 환경에서 실행해야 합니다."
