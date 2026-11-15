@echo off
rem Semantic Sonar - Windows launcher (requires Node.js 18+ from https://nodejs.org)
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js was not found. Install it from https://nodejs.org and re-run.
  pause
  exit /b 1
)
start "" http://localhost:8123
node serve.mjs 8123
pause
