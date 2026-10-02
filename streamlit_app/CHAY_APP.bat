@echo off
chcp 65001 >nul
title Soi Ảnh - Thật hay AI

echo ============================================
echo    Soi Anh - That hay AI
echo ============================================
echo.

:: ── Kiem tra Python ──
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Chua cai Python!
    echo.
    echo Huong dan:
    echo   1. Vao https://www.python.org/downloads/
    echo   2. Tai phien ban moi nhat (3.10 tro len^)
    echo   3. Khi cai, TICK vao "Add Python to PATH"
    echo   4. Chay lai file nay
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

:: ── Kiem tra 64-bit ──
python -c "import struct; exit(0 if struct.calcsize('P')*8==64 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo [LOI] Can Python 64-bit! Phien ban hien tai la 32-bit.
    echo Tai lai Python 64-bit tai https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo [OK] Python 64-bit
echo.

:: ── Tao virtual environment ──
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Dang tao moi truong ao (.venv^)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [LOI] Khong tao duoc moi truong ao.
        pause
        exit /b 1
    )
    echo [OK] Da tao .venv
) else (
    echo [1/3] Moi truong ao da ton tai.
)
echo.

:: ── Kich hoat venv ──
call .venv\Scripts\activate.bat

:: ── Cai thu vien ──
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/3] Dang cai thu vien (lan dau mat 5-10 phut, can Internet^)...
    echo      Vui long cho...
    echo.
    pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [LOI] Cai thu vien that bai!
        echo Kiem tra:
        echo   - May co ket noi Internet khong?
        echo   - Co bi chan mang (firewall/proxy^) khong?
        echo.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Da cai xong tat ca thu vien.
) else (
    echo [2/3] Thu vien da duoc cai san.
)
echo.

:: ── Kiem tra file mo hinh ──
set MISSING=0
if not exist "models\cnn_model.keras" if not exist "models\cnn_model.h5" (
    echo [CANH BAO] Thieu file: models\cnn_model.keras
    set MISSING=1
)
if not exist "models\svm.joblib" (
    echo [CANH BAO] Thieu file: models\svm.joblib
    set MISSING=1
)
if not exist "models\logistic_regression.joblib" (
    echo [CANH BAO] Thieu file: models\logistic_regression.joblib
    set MISSING=1
)
if not exist "models\random_forest.joblib" (
    echo [CANH BAO] Thieu file: models\random_forest.joblib
    set MISSING=1
)
if %MISSING%==1 (
    echo.
    echo Mot so mo hinh bi thieu. App van chay nhung se khong du 4 mo hinh.
    echo Dam bao thu muc "models" co du file khi copy qua may moi.
    echo.
)

:: ── Chay app ──
echo [3/3] Dang khoi dong ung dung...
echo.
echo ============================================
echo    Trinh duyet se tu mo tai:
echo    http://localhost:8501
echo.
echo    Nhan Ctrl+C de dung ung dung.
echo ============================================
echo.
python -m streamlit run app.py --server.headless true
pause
