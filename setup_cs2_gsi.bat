@echo off
setlocal EnableDelayedExpansion

echo CS2 GSI Setup
echo =============

set "CFG_CONTENT="uri" "http://127.0.0.1:3000/""
set "CFG_CONTENT=!CFG_CONTENT!!LF!"

set FOUND=0

for %%D in (C D E F G) do (
    set "TRY=%%D:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"
    if exist "!TRY!" (
        set "CFG_DIR=!TRY!"
        set FOUND=1
        goto :found
    )
    set "TRY=%%D:\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"
    if exist "!TRY!" (
        set "CFG_DIR=!TRY!"
        set FOUND=1
        goto :found
    )
)

:found
if %FOUND%==0 (
    echo [ERROR] Could not find CS2 installation automatically.
    echo.
    echo Please enter the path to your CS2 cfg folder manually.
    echo Example: C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg
    echo.
    set /p CFG_DIR="CFG folder path: "
    if not exist "!CFG_DIR!" (
        echo [ERROR] Path does not exist: !CFG_DIR!
        pause
        exit /b 1
    )
)

echo [OK] Found cfg folder: %CFG_DIR%

set "OUT=%CFG_DIR%\gamestate_integration_gsi.cfg"

(
echo "GSI Recorder"
echo {
echo     "uri" "http://127.0.0.1:3000/"
echo     "timeout" "5.0"
echo     "buffer"  "0.1"
echo     "throttle" "0.1"
echo     "heartbeat" "5.0"
echo     "data"
echo     {
echo         "map"            "1"
echo         "round"          "1"
echo         "player_id"      "1"
echo         "player_state"   "1"
echo         "player_weapons" "1"
echo         "player_match_stats" "1"
echo         "bomb"           "1"
echo         "allplayers_id"  "0"
echo     }
echo }
) > "%OUT%"

if exist "%OUT%" (
    echo [OK] Created: %OUT%
) else (
    echo [ERROR] Failed to write config file. Try running as Administrator.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  LAST STEP: Add launch option in Steam
echo ============================================================
echo  1. Open Steam ^> Library ^> Right-click CS2 ^> Properties
echo  2. Under "Launch Options" add:  -gamestateintegration
echo  3. Close and restart CS2
echo ============================================================
echo.
echo Setup complete. Run run.bat before starting a match.
echo.
pause
