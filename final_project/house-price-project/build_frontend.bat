@echo off
title Build Frontend Production Bundle
echo ===================================================
echo Building Frontend Production Bundle (tsc + vite)...
echo ===================================================
cd /d "%~dp0frontend"

npm run build
pause
