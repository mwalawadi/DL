@echo off
title House Price - Backend (FastAPI)
echo ===================================================
echo Starting FastAPI Backend Server on port 8000...
echo ===================================================
cd /d "%~dp0backend"

if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found at ..\.venv
)

uvicorn app.main:app --reload --port 8000
pause
