@echo off
title House Price Prediction App Launcher
echo ===================================================
echo Launching House Price Prediction Backend and Frontend...
echo ===================================================

start "House Price - Backend" cmd /c "%~dp0run_backend.bat"
start "House Price - Frontend" cmd /c "%~dp0run_frontend.bat"

echo Both services launched in separate windows!
echo - Backend:  http://localhost:8000/docs
echo - Frontend: http://localhost:5173
timeout /t 5
