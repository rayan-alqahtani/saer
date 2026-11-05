param([ValidateSet('gradio','rest')][string]$Mode = 'gradio')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$mainPython = Join-Path $projectRoot '.runtime/clean-main/Scripts/python.exe'
$imagePython = Join-Path $projectRoot '.runtime/clean-image/Scripts/python.exe'
if (!(Test-Path -LiteralPath $mainPython) -or !(Test-Path -LiteralPath $imagePython)) {
    throw 'أنشئ بيئتي التشغيل المستقلتين كما في README أولاً.'
}
$previousDirectory = Get-Location
try {
    Set-Location -LiteralPath $projectRoot
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:SAER_IMAGE_PYTHONPATH -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = '1'
    $env:SAER_APP_ROOT = $projectRoot
    $env:LOCAL_ASSETS_DIR = $projectRoot
    $env:SAER_IMAGE_PYTHON = $imagePython
    $env:SAER_RUNTIME_CACHE = Join-Path $projectRoot '.runtime-cache/delivery'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    $env:PYTHONIOENCODING = 'utf-8'
    if (!$env:FRONTEND_ORIGINS) {
        $env:FRONTEND_ORIGINS = 'http://127.0.0.1:5173,https://saer-rmsavg637.pages.dev'
    }
    & $mainPython tools/restore_public_encoders.py
    if ($LASTEXITCODE -ne 0) { throw 'المشفرات العامة غير مكتملة؛ راجع أمر الاستعادة.' }
    if ($Mode -eq 'gradio') {
        & $mainPython -m uvicorn app:app --app-dir hf_backend --host 127.0.0.1 --port 7863
    } else {
        & $mainPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
    }
    if ($LASTEXITCODE -ne 0) { throw 'توقف خادم التشغيل بخطأ.' }
} finally {
    Set-Location -LiteralPath $previousDirectory
}
