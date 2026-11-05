@echo off
setlocal
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"
pushd "%PROJECT_ROOT%"

if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
  echo لم يتم العثور على بيئة Python داخل مجلد المشروع:
  echo %PROJECT_ROOT%\.venv
  echo أنشئ البيئة وثبّت requirements-web.txt أولاً.
  pause
  popd
  exit /b 1
)

start "سعّر - الخادم" "%PROJECT_ROOT%\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8000
popd
endlocal
