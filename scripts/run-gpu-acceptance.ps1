<# One-command GPU preflight / optional bilingual acceptance.
   Does not install software, change system CUDA, or download models unless
   -EnglishSample and -ChineseSample are both supplied and preflight succeeds.
#>
[CmdletBinding()]
param(
  [string] $Python = 'python',
  [string] $EnglishSample,
  [string] $ChineseSample,
  [string] $ReportDir = 'gpu-reports'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$reportPath = (Join-Path (Resolve-Path $ReportDir).Path 'preflight.json')
Write-Host '[1/3] Inspecting installed tools and ASR modules...'
& $Python -m transcription.doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment inspection failed.' }

Write-Host '[2/3] Checking actual CUDA tensor execution (no CPU fallback)...'
& $Python -m transcription.gpu_diagnostics $reportPath
if ($LASTEXITCODE -ne 0) {
  throw "CUDA probe failed; see $reportPath. Model inference was NOT started."
}

if (-not $EnglishSample -and -not $ChineseSample) {
  Write-Host "CUDA preflight passed. No audio supplied; model tests intentionally not started."
  exit 0
}
if (-not $EnglishSample -or -not $ChineseSample) {
  throw 'Provide both -EnglishSample and -ChineseSample (or neither).'
}
if (-not (Test-Path -LiteralPath $EnglishSample -PathType Leaf)) {
  throw "English audio does not exist: $EnglishSample"
}
if (-not (Test-Path -LiteralPath $ChineseSample -PathType Leaf)) {
  throw "Chinese audio does not exist: $ChineseSample"
}

Write-Host '[3/3] Real large-v3 -> Paraformer -> large-v3 GPU model acceptance...'
$env:TRANSCRIPTION_ENABLE_HARDWARE = '1'
$env:TRANSCRIPTION_EN_SAMPLE = (Resolve-Path -LiteralPath $EnglishSample).Path
$env:TRANSCRIPTION_ZH_SAMPLE = (Resolve-Path -LiteralPath $ChineseSample).Path
$env:TRANSCRIPTION_HARDWARE_DEVICE = 'gpu'
$env:TRANSCRIPTION_GPU_REPORT = Join-Path (Resolve-Path $ReportDir).Path 'gpu-acceptance-report.json'
try {
  & $Python -m pytest tests/hardware/test_real_models.py -m hardware -v
  if ($LASTEXITCODE -ne 0) { throw 'GPU model acceptance failed; see diagnostics report.' }
} finally {
  Remove-Item Env:TRANSCRIPTION_ENABLE_HARDWARE -ErrorAction SilentlyContinue
  Remove-Item Env:TRANSCRIPTION_EN_SAMPLE -ErrorAction SilentlyContinue
  Remove-Item Env:TRANSCRIPTION_ZH_SAMPLE -ErrorAction SilentlyContinue
  Remove-Item Env:TRANSCRIPTION_HARDWARE_DEVICE -ErrorAction SilentlyContinue
  Remove-Item Env:TRANSCRIPTION_GPU_REPORT -ErrorAction SilentlyContinue
}
Write-Host "GPU acceptance passed. Reports saved in $ReportDir"
