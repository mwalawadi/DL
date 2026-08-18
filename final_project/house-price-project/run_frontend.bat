@echo off
title House Price - Frontend (React + Vite)
echo ===================================================
echo Starting React Vite Frontend on port 5173...
echo ===================================================
cd /d "%~dp0frontend"

if not exist ".env" (
    if exist ".env.example" (
        echo Creating frontend\.env from .env.example...
        copy ".env.example" ".env" >nul
    )
)

npm run dev
pause
