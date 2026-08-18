@echo off
title House Price Prediction App Launcher
echo ===================================================
echo Launching House Price Prediction Backend and Frontend...
echo ===================================================

:: Ensure .env exists for backend and frontend
if not exist "%~dp0backend\.env" (
    if exist "%~dp0backend\.env.example" (
        echo Initializing backend\.env from .env.example...
        copy "%~dp0backend\.env.example" "%~dp0backend\.env" >nul
    )
)
if not exist "%~dp0frontend\.env" (
    if exist "%~dp0frontend\.env.example" (
        echo Initializing frontend\.env from .env.example...
        copy "%~dp0frontend\.env.example" "%~dp0frontend\.env" >nul
    )
)

start "House Price - Backend" cmd /c "%~dp0run_backend.bat"
start "House Price - Frontend" cmd /c "%~dp0run_frontend.bat"

echo.
echo Both services launched in separate windows!
echo - Backend API:  http://localhost:8000
echo - Swagger Docs: http://localhost:8000/docs
echo - Frontend UI:  http://localhost:5173
echo.
timeout /t 5
