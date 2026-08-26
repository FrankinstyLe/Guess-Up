@echo off
REM Double-click this at the fair. Keeps the window open if anything fails.
cd /d "%~dp0"
python run_kiosk.py %*
if errorlevel 1 (
    echo.
    echo The kiosk exited with an error. Things to check, in order:
    echo   1. python scripts/fetch_models.py     ^(models present?^)
    echo   2. python scripts/build_gallery.py    ^(galleries built?^)
    echo   3. python run_kiosk.py --selftest     ^(what does it say?^)
    echo.
    pause
)
