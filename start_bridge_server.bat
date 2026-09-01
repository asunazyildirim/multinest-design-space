@echo off
REM ---------------------------------------------------------------------
REM Starts the Windows half of the MAGNUS bridge. Double-click it, or run
REM it from any terminal -- it does not care where it is invoked from.
REM
REM It exists because two things about this machine make the obvious
REM command fail:
REM
REM   * "python" on PATH is the Microsoft Store stub in WindowsApps, which
REM     has no pywin32 and cannot talk to HYSYS. The interpreter that works
REM     is Anaconda's, and this script finds it.
REM   * the project path contains spaces and a non-ASCII folder name, so
REM     the cd has to be quoted every time.
REM
REM Anything you type after the script name is passed through to
REM hysys_bridge_server.py, so --fake and the rest still work:
REM
REM     start_bridge_server.bat --fake      (transport test, no HYSYS)
REM ---------------------------------------------------------------------

setlocal

REM Run from this script's own directory -- %~dp0 keeps its trailing slash.
cd /d "%~dp0"

REM Candidate interpreters, best first. The Store stub in WindowsApps is
REM deliberately not among them.
REM
REM The hysyspy environment comes first because that is the one this project
REM is actually driven from -- "conda activate hysyspy" before running the
REM samplers. Anaconda's base has pywin32 too and works, but if the two ever
REM drift apart it should be the environment the rest of the work uses that
REM wins, not whichever one happens to be found first.
set "PY="
for %%I in (
    "%USERPROFILE%\anaconda3\envs\hysyspy\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\envs\hysyspy\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "C:\ProgramData\Anaconda3\python.exe"
) do if not defined PY if exist %%I set "PY=%%~I"

if not defined PY (
    echo.
    echo ERROR: could not find an Anaconda Python.
    echo.
    echo Looked in:
    echo     %USERPROFILE%\anaconda3\python.exe
    echo     %USERPROFILE%\miniconda3\python.exe
    echo     C:\ProgramData\Anaconda3\python.exe
    echo.
    echo Find yours with "where python" in an Anaconda Prompt, then either
    echo edit this file or run the server directly:
    echo     "C:\path\to\python.exe" hysys_bridge_server.py
    echo.
    pause
    exit /b 1
)

echo python : %PY%
echo folder : %CD%
echo.

REM Check the imports before announcing anything. A missing pywin32 and a
REM HYSYS that is simply not open produce very different fixes, and the
REM error further down would not distinguish them.
"%PY%" -c "import pythoncom, win32com.client" 2>nul
if errorlevel 1 (
    echo ERROR: this Python has no pywin32, so it cannot reach HYSYS.
    echo Install it with:  "%PY%" -m pip install pywin32
    echo.
    pause
    exit /b 1
)

"%PY%" hysys_bridge_server.py %*

REM Ctrl-C and crashes both land here. Without the pause a double-clicked
REM window would close before the traceback could be read.
echo.
pause
