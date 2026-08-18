Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting React Frontend on port 5173..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location -Path "$PSScriptRoot\frontend"

npm run dev
